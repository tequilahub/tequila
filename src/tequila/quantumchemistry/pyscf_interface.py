from tequila import TequilaException, TequilaWarning
from openfermion import MolecularData
from tequila.quantumchemistry.qc_base import ParametersQC, QuantumChemistryBase, NBodyTensor

import pyscf

import numpy, typing, warnings


class OpenVQEEPySCFException(TequilaException):
    pass


class QuantumChemistryPySCF(QuantumChemistryBase):
    def __init__(self, parameters: ParametersQC,
                 transformation: typing.Union[str, typing.Callable] = None,
                 *args, **kwargs):

        super().__init__(parameters=parameters, transformation=transformation, *args, **kwargs)

    @classmethod
    def from_tequila(cls, molecule, transformation=None, *args, **kwargs):
        c, h1, h2 = molecule.get_integrals(two_body_ordering="openfermion")
        return cls(nuclear_repulsion=c,
                          one_body_integrals=h1,
                          two_body_integrals=h2,
                          n_electrons=molecule.n_electrons,
                          transformation=transformation,
                          parameters=molecule.parameters, *args, **kwargs)

    def do_make_molecule(self, molecule=None, nuclear_repulsion=None, one_body_integrals=None, two_body_integrals=None,
                         *args, **kwargs) -> MolecularData:
        if molecule is None:
            if one_body_integrals is not None and two_body_integrals is not None:
                if nuclear_repulsion is None:
                    warnings.warn("PySCF Interface: No constant part (nuclear repulsion)", TequilaWarning)
                    nuclear_repulsion = 0.0
                molecule = super().do_make_molecule(nuclear_repulsion=nuclear_repulsion,
                                                    one_body_integrals=one_body_integrals,
                                                    two_body_integrals=two_body_integrals, *args, **kwargs)
            else:

                # determine if the molecular structure was given over a file already
                # if not, write the file
                geometry = self.parameters.get_geometry()
                pyscf_geomstring = ""
                for atom in geometry:
                    pyscf_geomstring += "{} {} {} {};".format(atom[0],atom[1][0],atom[1][1],atom[1][2])

                if "point_group" in kwargs:
                    point_group = kwargs["point_group"]
                else:
                    point_group = None

                mol = pyscf.gto.Mole()
                mol.atom = pyscf_geomstring
                mol.basis = self.parameters.basis_set

                if point_group is not None:
                    if point_group.lower() != "c1":
                        mol.symmetry = True
                        mol.symmetry_subgroup = point_group
                    else:
                        mol.symmetry = False
                else:
                    mol.symmetry=True

                mol.build()

                # solve restricted HF
                mf = pyscf.scf.RHF(mol)
                mf.kernel()
                self.irreps=mf.get_irrep_nelec()

                # compute mo integrals
                mo_coeff = mf.mo_coeff
                h_ao = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
                h = numpy.einsum('pi,pq,qj->ij', mo_coeff, h_ao, mo_coeff)
                g = pyscf.ao2mo.kernel(mol, mo_coeff)
                g = pyscf.ao2mo.restore(1, numpy.asarray(g), mo_coeff.shape[1])

                # Mulliken to Openfermion convention
                g = NBodyTensor(elems=g, ordering="mulliken").reorder(to="openfermion").elems

                self.pyscf_molecule=mol
                self.point_group = mol.symmetry_subgroup
                molecule=super().do_make_molecule(one_body_integrals=h, two_body_integrals=g, nuclear_repulsion=mol.energy_nuc())

        return molecule

    def compute_fci(self, *args, **kwargs):
        from pyscf import fci
        c, h1, h2 = self.get_integrals(two_body_ordering="mulliken")
        norb = self.n_orbitals
        nelec = self.n_electrons
        e, fcivec = fci.direct_spin1.kernel(h1, h2, norb, nelec, **kwargs)
        return e + c

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
        c, h1, h2 = self.get_integrals(two_body_ordering="mulliken")
        norb = self.n_orbitals
        nelec = self.n_electrons

        mo_coeff = numpy.eye(norb)
        mo_occ = numpy.zeros(norb)
        mo_occ[:nelec // 2] = 2

        pyscf_mol = pyscf.gto.M()
        pyscf_mol.nelectron = nelec
        pyscf_mol.incore_anyway = True  # ensure that custom integrals are used
        pyscf_mol.energy_nuc = lambda *args: c

        hf = pyscf.scf.RHF(pyscf_mol)
        hf.get_hcore = lambda *args: h1
        hf.get_ovlp = lambda *args: numpy.eye(norb)
        hf._eri = pyscf.ao2mo.restore(8, h2, norb)

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
                base += "{:15} : {} ({})\n".format("point_group", self.pyscf_molecule.groupname, self.pyscf_molecule.topgroup)
            if hasattr(self, "irreps"):
                base += "{:15} : {}\n".format("irreps", self.irreps)
        except:
            return base
        return base