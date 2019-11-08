from openvqe import OpenVQEException, numpy, typing
from openvqe import dataclass
from openfermion import MolecularData

from openfermionpsi4 import run_psi4
from openvqe.quantumchemistry.qc_base import ParametersQC, QuantumChemistryBase, Amplitudes
from openfermionpsi4._psi4_conversion_functions import parse_psi4_ccsd_amplitudes

import pyscf


class OpenVQEEPySCFException(OpenVQEException):
    pass


@dataclass
class ParametersPsi4:
    run_scf: bool = True
    run_mp2: bool = False
    run_cisd: bool = False
    run_ccsd: bool = False
    run_fci: bool = False
    verbose: bool = False
    tolerate_error: bool = False
    delete_input: bool = True
    delete_output: bool = True
    memory: int = 8000


class QuantumChemistryPsi4(QuantumChemistryBase):
    def __init__(self, parameters: ParametersQC, transformation: typing.Union[str, typing.Callable] = None,
                 parameters_psi4: ParametersPsi4 = None):
        if parameters_psi4 is None:
            self.parameters_psi4 = ParametersPsi4()
        else:
            self.parameters_psi4 = parameters_psi4

        super().__init__(parameters=parameters, transformation=transformation)

    def do_make_molecule(self, molecule=None) -> MolecularData:
        if molecule is None:
            molecule = MolecularData(**self.parameters.molecular_data_param)
        return run_psi4(molecule, **self.parameters_psi4.__dict__)

    def compute_ccsd_amplitudes(self):
        filename = self.parameters.filename
        if ".out" not in self.parameters.filename:
            filename += ".out"

        from os import path, access, R_OK
        file_exists = path.isfile("./" + filename) and access("./" + filename, R_OK)

        if not file_exists or not self.parameters_psi4.run_ccsd:
            self.parameters_psi4.run_ccsd = True
            self.parameters_psi4.delete_output = False
            self.make_molecule()

        return self.parse_ccsd_amplitudes(filename)

    def parse_ccsd_amplitudes(self, filename: str) -> Amplitudes:

        singles, doubles = parse_psi4_ccsd_amplitudes(number_orbitals=self.n_orbitals * 2,
                                                      n_alpha_electrons=self.n_electrons // 2,
                                                      n_beta_electrons=self.n_electrons // 2,
                                                      psi_filename=filename)

        tmp1 = Amplitudes.from_ndarray(array=singles, closed_shell=False)
        tmp2 = Amplitudes.from_ndarray(array=doubles, closed_shell=False)
        return Amplitudes(data={**tmp1.data, **tmp2.data}, closed_shell=False)
