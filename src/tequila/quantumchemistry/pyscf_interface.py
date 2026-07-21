from tequila import TequilaException, QubitWaveFunction, BitNumbering
from tequila.quantumchemistry.qc_base import QuantumChemistryBase
from tequila.quantumchemistry.encodings import JordanWigner
from tequila.quantumchemistry import ParametersQC, NBodyTensor
import pyscf
from pyscf import fci

from .chemistry_tools import OrbitalData

import numpy
import typing
import warnings


def _merge_alpha_beta_strs(alpha_str, beta_str, norb):
    """
    Merge alpha and beta bitstrings into a single string and compute the resulting phase.

    Args:
        alpha_str (int): Bitstring representing alpha electrons.
        beta_str (int): Bitstring representing beta electrons.
        norb (int): Number of orbitals.

    Returns:
        tuple:
            merged_str (int): Interleaved bitstring.
            phase (int): Phase factor (+1 or -1) from fermionic antisymmetry.
    """
    # Interleave the alpha and beta strings
    alpha_str_b = bin(alpha_str)[2:].zfill(norb)
    beta_str_b = bin(beta_str)[2:].zfill(norb)
    merged_str = "".join([alpha_str_b[i] + beta_str_b[i] for i in range(norb)])

    # Position of filled orbitals
    set_bits_beta = [i for i in range(norb) if (beta_str >> i) & 1]
    phase = (-1) ** sum([(alpha_str & 2**i - 1).bit_count() for i in set_bits_beta])
    return int(merged_str, 2), phase


class OpenVQEEPySCFException(TequilaException):
    pass


class QuantumChemistryPySCF(QuantumChemistryBase):
    def __init__(
        self, parameters: ParametersQC, transformation: typing.Union[str, typing.Callable] = None, *args, **kwargs
    ):
        orbitals = None
        if "one_body_integrals" not in kwargs:
            geometry = parameters.get_geometry()
            pyscf_geomstring = ""
            for atom in geometry:
                pyscf_geomstring += "{} {} {} {};".format(atom[0], atom[1][0], atom[1][1], atom[1][2])

            if "point_group" in kwargs:
                point_group = kwargs["point_group"]
            else:
                point_group = None

            mol = pyscf.gto.Mole()
            mol.atom = pyscf_geomstring
            mol.basis = parameters.basis_set
            mol.charge = parameters.charge

            if point_group is not None:
                if point_group.lower() != "c1":
                    mol.symmetry = True
                    mol.symmetry_subgroup = point_group
                else:
                    mol.symmetry = False
            else:
                mol.symmetry = True

            mol.build(parse_arg=False)

            # solve restricted HF
            mf = pyscf.scf.RHF(mol)
            mf.verbose = False
            if "verbose" in kwargs:
                mf.verbose = kwargs["verbose"]

            mf.kernel()

            self.irreps = pyscf.symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff).tolist()

            orbital_energies = mf.mo_energy

            orbitals = [
                OrbitalData(idx_total=idx, irrep=irr, energy=energy)
                for idx, (irr, energy) in enumerate(zip(self.irreps, orbital_energies))
            ]

            for irr in {o.irrep for o in orbitals}:
                for i, o in enumerate([o for o in orbitals if o.irrep == irr]):
                    o.idx_irrep = i

            # compute mo integrals
            mo_coeff = mf.mo_coeff
            h_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
            g_ao = mol.intor("int2e", aosym="s1")
            S = mol.intor_symmetric("int1e_ovlp")
            g_ao = NBodyTensor(elems=g_ao, ordering="mulliken")

            self.pyscf_molecule = mol
            self.point_group = mol.symmetry_subgroup

            kwargs["overlap_integrals"] = S
            kwargs["two_body_integrals"] = g_ao
            kwargs["one_body_integrals"] = h_ao
            kwargs["orbital_coefficients"] = mo_coeff
            kwargs["orbital_type"] = "hf"

            if "nuclear_repulsion" not in kwargs:
                kwargs["nuclear_repulsion"] = mol.energy_nuc()

        super().__init__(parameters=parameters, transformation=transformation, orbitals=orbitals, *args, **kwargs)

    def compute_fci(self, get_wfn=False, ci0=None, use_hcb=False, **kwargs):
        c, h1, h2 = self.get_integrals(ordering="chem")
        norb = self.n_orbitals
        nelec = self.n_electrons

        if ci0 is not None:
            ci0 = self.create_ci_vector(ci0, use_hcb=use_hcb)

        e, fcivecs = fci.direct_spin1.kernel(
            h1, h2.elems, norb, nelec, max_cycle=2000, max_space=100, ci0=ci0, **kwargs
        )

        if get_wfn:
            if not ("nroots" in kwargs and kwargs["nroots"] > 1):
                fcivecs = [fcivecs]
                e = [e]
            wfns = []
            energies = [x + c for x in e]
            for fcivec in fcivecs:
                if use_hcb:
                    alpha_strs = fci.cistring.make_strings(range(norb), nelec // 2)
                    wfn_dim = 2**norb
                    wfn = numpy.zeros(wfn_dim)
                    for i, alpha_str in enumerate(alpha_strs):
                        alpha_str_b = bin(alpha_str)[2:].zfill(norb)
                        merged_str = int(alpha_str_b, 2)
                        wfn[merged_str] = fcivec[i, i]
                else:
                    alpha_strs = fci.cistring.make_strings(range(norb), nelec // 2)
                    beta_strs = alpha_strs.copy()
                    wfn_dim = 2 ** (2 * norb)
                    wfn = numpy.zeros(wfn_dim)
                    for i, alpha_str in enumerate(alpha_strs):
                        for j, beta_str in enumerate(beta_strs):
                            if not self.transformation.up_then_down:
                                merged_str, phase = _merge_alpha_beta_strs(alpha_str, beta_str, norb)
                                wfn[merged_str] = phase * fcivec[i, j]
                            else:
                                alpha_str_b = bin(alpha_str)[2:].zfill(norb)
                                beta_str_b = bin(beta_str)[2:].zfill(norb)
                                merged_str_b = alpha_str_b + beta_str_b
                                merged_str = int(merged_str_b, 2)
                                wfn[merged_str] = fcivec[i, j]
                wfns.append(QubitWaveFunction.from_array(wfn, numbering=BitNumbering.LSB))
            if not ("nroots" in kwargs and kwargs["nroots"] > 1):
                return energies[0], wfns[0]
            return energies, wfns

        return e + c

    def create_ci_vector(self, wfn, use_hcb=False):
        """
        Reduces a full Fock space CI vector to the (na, nb) vector
        required by fci.direct_spin1.

        Args:
            wfn (np.ndarray): 1D vector of length 2**(2*norb).

        Returns:
            ci_vector (np.ndarray): 2D vector of shape (na, nb).
        """
        if not isinstance(wfn, numpy.ndarray):
            if hasattr(wfn, "to_array"):
                wfn = wfn.to_array()
            else:
                raise TequilaException(
                    f"create_ci_vector received wfn object of type {type(wfn)} but needs numpy.ndarray or tq.QubitWaveFunction\nwfn={wfn}"
                )

        wfn = numpy.real_if_close(wfn).astype(numpy.float64)

        if type(self.transformation).__name__ != "JordanWigner":
            warnings.warn("!!!guess wavefunctions for pyscf need to be in JordanWigner encoding!!!")

        norb = self.n_orbitals
        nelec = self.n_electrons
        neleca = (nelec + 1) // 2
        nelecb = nelec // 2

        expected_len = 2 ** (2 * norb) if not use_hcb else 2**norb
        if wfn.size != expected_len:
            raise ValueError(
                f"Input vector has length {wfn.size}, "
                f"but expected full Fock space length {expected_len} (for {norb} orbitals)."
            )

        na = fci.cistring.num_strings(norb, neleca)
        nb = fci.cistring.num_strings(norb, nelecb)
        ci_vector = numpy.zeros((na, nb), dtype=wfn.dtype)

        for i in range(expected_len):
            if use_hcb:
                alpha_str = bin(i)[2:].zfill(norb)
                beta_str = alpha_str
            else:
                state_binary_str = bin(i)[2:].zfill(2 * norb)
                if not self.transformation.up_then_down:
                    alpha_str = state_binary_str[0::2]
                    beta_str = state_binary_str[1::2]
                else:
                    alpha_str = state_binary_str[0 : len(state_binary_str) // 2]
                    beta_str = state_binary_str[len(state_binary_str) // 2 : len(state_binary_str)]

            # Only keep states that are particle-number conserving
            if alpha_str.count("1") == neleca and beta_str.count("1") == nelecb:
                alpha_int = int(alpha_str, 2)
                beta_int = int(beta_str, 2)

                # Compute the phase as in _merge_alpha_beta_strs(alpha_str, beta_str, norb)
                set_bits_beta = [i for i in range(norb) if (beta_int >> i) & 1]
                try:
                    phase = (-1) ** sum([(alpha_int & 2**i - 1).bit_count() for i in set_bits_beta])
                except AttributeError as error:
                    phase = (-1) ** sum([bin(alpha_int & 2**i - 1).count("1") for i in set_bits_beta])

                coeff = wfn[i]
                row_idx = fci.cistring.str2addr(norb, neleca, alpha_int)
                col_idx = fci.cistring.str2addr(norb, nelecb, beta_int)
                ci_vector[row_idx, col_idx] = phase * coeff

        return ci_vector

    def compute_energy(self, method: str, *args, **kwargs) -> float:
        method = method.lower()

        if method == "hf":
            return self._get_hf(do_not_solve=False, **kwargs).e_tot
        elif method == "mp2":
            return self._run_mp2(**kwargs).e_tot
        elif method == "cisd":
            hf = self._get_hf(do_not_solve=False, **kwargs)
            return self._run_cisd(hf=hf, **kwargs).e_tot
        elif method == "ccsd":
            return self._run_ccsd(**kwargs).e_tot
        elif method == "ccsd(t)":
            ccsd = self._run_ccsd(**kwargs)
            return ccsd.e_tot + self._compute_perturbative_triples_correction(ccsd=ccsd, **kwargs)
        elif method == "fci":
            return self.compute_fci(**kwargs)
        else:
            raise TequilaException("unknown method: {}".format(method))

    def _get_hf(self, do_not_solve=True, **kwargs):
        c, h1, h2 = self.get_integrals(ordering="mulliken")
        norb = self.n_orbitals
        nelec = self.n_electrons

        mo_coeff = numpy.eye(norb)
        mo_occ = numpy.zeros(norb)
        mo_occ[: nelec // 2] = 2

        pyscf_mol = pyscf.gto.M(verbose=0, parse_arg=False)
        pyscf_mol.nelectron = nelec
        pyscf_mol.incore_anyway = True  # ensure that custom integrals are used
        pyscf_mol.energy_nuc = lambda *args: c

        hf = pyscf.scf.RHF(pyscf_mol)
        hf.get_hcore = lambda *args: h1
        hf.get_ovlp = lambda *args: numpy.eye(norb)
        hf._eri = pyscf.ao2mo.restore(8, h2.elems, norb)

        if do_not_solve:
            hf.mo_coeff = mo_coeff
            hf.mo_occ = mo_occ
        else:
            hf.kernel(numpy.diag(mo_occ))

        return hf

    def _run_ccsd(self, hf=None, **kwargs):
        from pyscf import cc

        if hf is None:
            hf = self._get_hf()
        ccsd = cc.RCCSD(hf)
        ccsd.kernel()
        return ccsd

    def _compute_perturbative_triples_correction(self, ccsd=None, **kwargs) -> float:
        if ccsd is None:
            ccsd = self._run_ccsd(**kwargs)
        ecorr = ccsd.ccsd_t()
        return ecorr

    def _run_cisd(self, hf=None, **kwargs):
        from pyscf import ci

        if hf is None:
            hf = self._get_hf(**kwargs)
        cisd = ci.RCISD(hf)
        cisd.kernel()
        return cisd

    def _run_mp2(self, hf=None, **kwargs):
        from pyscf import mp

        if hf is None:
            hf = self._get_hf(**kwargs)
        mp2 = mp.MP2(hf)
        mp2.kernel()
        return mp2

    def __str__(self):
        base = super().__str__()
        try:
            if hasattr(self, "pyscf_molecule"):
                base += "{:15} : {} ({})\n".format(
                    "point_group", self.pyscf_molecule.groupname, self.pyscf_molecule.topgroup
                )
            if hasattr(self, "irreps"):
                base += "{:15} : {}\n".format("irreps", self.irreps)
        except Exception:
            return base
        return base


if __name__ == "__main__":
    pass
