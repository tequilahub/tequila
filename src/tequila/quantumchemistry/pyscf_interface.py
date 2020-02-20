from tequila import TequilaException, numpy, typing
from tequila import dataclass
from openfermion import MolecularData

from openfermionpyscf import run_pyscf
from tequila.quantumchemistry.qc_base import ParametersQC, QuantumChemistryBase, OldAmplitudes


class OpenVQEEPySCFException(TequilaException):
    pass


@dataclass
class ParametersPySCF:
    run_scf: bool = True
    run_mp2: bool = False
    run_cisd: bool = False
    run_ccsd: bool = False
    run_fci: bool = False
    verbose: bool = False


class QuantumChemistryPySCF(QuantumChemistryBase):
    def __init__(self, parameters: ParametersQC, transformation: typing.Union[str, typing.Callable] = None,
                 parameters_pyscf: ParametersPySCF = None):
        if parameters_pyscf is None:
            self.parameters_pyscf = ParametersPySCF()
        else:
            self.parameters_pyscf = parameters_pyscf

        super().__init__(parameters=parameters, transformation=transformation)

    def do_make_molecule(self, molecule=None) -> MolecularData:
        if molecule is None:
            molecule = MolecularData(**self.parameters.molecular_data_param)
        return run_pyscf(molecule, **self.parameters_pyscf.__dict__)

    def compute_ccsd_amplitudes(self):
        ## Checks if it is already calculated
        molecule = self.make_molecule()
        if molecule.ccsd_energy == None:
            self.parameters_pyscf.run_ccsd = True
            molecule = self.make_molecule()

        return self.parse_ccsd_amplitudes(molecule)

    def parse_ccsd_amplitudes(self, molecule) -> OldAmplitudes:

        singles = molecule._ccsd_single_amps
        doubles = molecule._ccsd_double_amps

        tmp1 = OldAmplitudes.from_ndarray(array=singles, closed_shell=False)
        tmp2 = OldAmplitudes.from_ndarray(array=doubles, closed_shell=False)
        return OldAmplitudes(data={**tmp1.data, **tmp2.data}, closed_shell=False)
