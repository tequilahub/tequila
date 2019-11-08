from openvqe import OpenVQEException, numpy, typing
from openvqe import dataclass
from openfermion import MolecularData

from openfermionpyscf import run_pyscf
from openvqe.quantumchemistry.qc_base import ParametersQC, QuantumChemistryBase, Amplitudes

import pyscf


class OpenVQEEPySCFException(OpenVQEException):
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

    def make_molecule(self) -> MolecularData:
        """
        Creates a molecule in openfermion format by running psi4 and extracting the data
        Will check for previous outputfiles before running
        :param parameters: An instance of ParametersQC, which also holds an instance of ParametersPsi4 via parameters.psi4
        The molecule will be saved in parameters.filename, if this file exists before the call the molecule will be imported from the file
        :return: the molecule in openfermion.MolecularData format
        """
        molecule = MolecularData(**self.parameters.molecular_data_param)
        # try to load

        do_compute = True
        if self.parameters.filename:
            try:
                import os
                if os.path.exists(self.parameters.filename):
                    molecule.load()
                    do_compute = False
            except OSError:
                do_compute = True

        if do_compute:
            molecule = run_pyscf(molecule, **self.parameters_pyscf.__dict__)

        molecule.save()
        return molecule

    def compute_ccsd_amplitudes(self):
        ## Checks if it is already calculated
        molecule = self.make_molecule()
        if molecule.cisd_energy  == None:
            self.parameters_pyscf = ParametersPySCF(run_ccsd=True) 
            molecule = self.make_molecule()

        return self.parse_ccsd_amplitudes(molecule)

    def parse_ccsd_amplitudes(self, molecule) -> Amplitudes:

        singles = molecule._ccsd_single_amps
        doubles = molecule._ccsd_double_amps

        tmp1 = Amplitudes.from_ndarray(array=singles, closed_shell=False)
        tmp2 = Amplitudes.from_ndarray(array=doubles, closed_shell=False)
        return Amplitudes(data={**tmp1.data, **tmp2.data}, closed_shell=False)
