from tequila import TequilaException
from openfermion import MolecularData

from tequila.quantumchemistry.qc_base import ParametersQC, QuantumChemistryBase, ClosedShellAmplitudes, Amplitudes

import copy
import numpy
import typing

from dataclasses import dataclass

__HAS_PSI4_PYTHON__ = False
try:
    import psi4

    __HAS_PSI4_PYTHON__ = True
except ModuleNotFoundError:
    __HAS_PSI4_PYTHON__ = False


class TequilaPsi4Exception(TequilaException):
    pass


class OpenVQEEPySCFException(TequilaException):
    pass


@dataclass
class Psi4Results:
    variables: dict = None  # psi4 variables dictionary, storing all computed values
    filename: str = None  # psi4 output file
    wfn: typing.Union[
        psi4.core.Wavefunction, psi4.core.CCWavefunction, psi4.core.CIWavefunction] = None  # psi4 wavefunction


class QuantumChemistryPsi4(QuantumChemistryBase):
    def __init__(self, parameters: ParametersQC, transformation: typing.Union[str, typing.Callable] = None, *args, **kwargs):

        self.energies = {}  # history to avoid recomputation
        self.logs = {}  # store full psi4 output

        super().__init__(parameters=parameters, transformation=transformation, *args, **kwargs)

    def do_make_molecule(self) -> MolecularData:
        molecule = MolecularData(**self.parameters.molecular_data_param)
        energy = self.compute_energy(method="hf")
        wfn = self.logs['hf'].wfn
        if wfn.nirrep() != 1:
            wfn = wfn.c1_deep_copy(wfn.basisset())
        molecule.one_body_integrals = self.compute_one_body_integrals(wfn)
        molecule.two_body_integrals = self.compute_two_body_integrals(wfn)
        molecule.hf_energy = self.logs['hf'].variables['HF TOTAL ENERGY']
        molecule.nuclear_repulsion = self.logs['hf'].variables['NUCLEAR REPULSION ENERGY']
        molecule.canonical_orbitals = numpy.asarray(wfn.Ca())
        molecule.overlap_integrals = numpy.asarray(wfn.S())
        molecule.n_orbitals = molecule.canonical_orbitals.shape[0]
        molecule.n_qubits = 2 * molecule.n_orbitals
        molecule.orbital_energies = numpy.asarray(wfn.epsilon_a())
        molecule.fock_matrix = numpy.asarray(wfn.Fa())
        molecule.save()
        return molecule

    def compute_one_body_integrals(self, ref_wfn=None):
        if ref_wfn is None:
            self.compute_energy(method="hf")
            ref_wfn = self.logs['hf'].wfn
        if ref_wfn.nirrep() != 1:
            wfn = ref_wfn.c1_deep_copy(ref_wfn.basisset())
        else:
            wfn = ref_wfn
        Ca = numpy.asarray(wfn.Ca())
        h = wfn.H()
        h = numpy.einsum("xy, yi -> xi", h, Ca, optimize='optimize')
        h = numpy.einsum("xj, xi -> ji", Ca, h, optimize='optimize')
        return h

    def compute_two_body_integrals(self, ref_wfn=None):
        if ref_wfn is None:
            if 'hf' not in self.logs:
                self.compute_energy(method="hf")
            ref_wfn = self.logs['hf'].wfn

        if ref_wfn.nirrep() != 1:
            wfn = ref_wfn.c1_deep_copy(ref_wfn.basisset())
        else:
            wfn = ref_wfn

        mints = psi4.core.MintsHelper(wfn.basisset())

        # Molecular orbitals (coeffs)
        Ca = wfn.Ca()
        h = numpy.asarray(mints.ao_eri())
        h = numpy.einsum("psqr", h, optimize='optimize')  # meet openfermion conventions
        h = numpy.einsum("wxyz, wi -> ixyz", h, Ca, optimize='optimize')
        h = numpy.einsum("wxyz, xi -> wiyz", h, Ca, optimize='optimize')
        h = numpy.einsum("wxyz, yi -> wxiz", h, Ca, optimize='optimize')
        h = numpy.einsum("wxyz, zi -> wxyi", h, Ca, optimize='optimize')
        return h

    def compute_ccsd_amplitudes(self):
        return self.compute_amplitudes(method='ccsd')

    def _run_psi4(self, options: dict = None, method=None, return_wfn=True, point_group=None, filename: str = None):

        psi4.core.clean()
        defaults = {'basis': self.parameters.basis_set,
                    'e_convergence': 1e-8,
                    'd_convergence': 1e-8}
        if options is None:
            options = {}
        options = {**defaults, **options}
        psi4.set_options(options)

        if filename is None:
            filename = "{}_{}.out".format(self.parameters.filename, method)

        psi4.core.set_output_file(filename)

        mol = psi4.geometry(self.parameters.get_geometry_string())

        if point_group is not None:
            mol.reset_point_group(point_group.lower())

        energy, wfn = psi4.energy(name=method.lower(), return_wfn=return_wfn, molecule=mol)

        self.energies[method.lower()] = energy
        self.logs[method.lower()] = Psi4Results(filename=filename, variables=copy.deepcopy(psi4.core.variables()),
                                                wfn=wfn)

        return energy, wfn

    def compute_amplitudes(self, method: str, options: dict = None, filename: str = None) -> typing.Union[
        Amplitudes, ClosedShellAmplitudes]:
        if __HAS_PSI4_PYTHON__:
            energy, wfn = self._run_psi4(method=method, options=options, point_group='c1', filename=filename)
            all_amplitudes = wfn.get_amplitudes()
            closed_shell = isinstance(wfn.reference_wavefunction(), psi4.core.RHF)
            if closed_shell:
                return ClosedShellAmplitudes(**{k: v.to_array() for k, v in all_amplitudes.items()})
            else:
                return Amplitudes(**{k: v.to_array() for k, v in all_amplitudes.items()})

        else:
            raise TequilaPsi4Exception("Can't find the psi4 python module, let your environment know the path to psi4")

    def compute_energy(self, method: str = "fci", options=None, *args, **kwargs):
        if method.lower() in self.energies:
            return self.energies[method.lower()]
        if __HAS_PSI4_PYTHON__:
            return self._run_psi4(method=method, options=options, *args, **kwargs)[0]
        else:
            raise TequilaPsi4Exception("Can't find the psi4 python module, let your environment know the path to psi4")
