from openvqe import OpenVQEException, numpy, typing
from openvqe import dataclass
from openfermion import MolecularData
from openfermionpsi4._psi4_conversion_functions import parse_psi4_ccsd_amplitudes
from openfermionpsi4 import run_psi4
from openvqe.quantumchemistry.qc_base import ParametersQC, QuantumChemistryBase, Amplitudes

import psi4


class OpenVQEEPsi4Exception(OpenVQEException):
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
            molecule = run_psi4(molecule, **self.parameters_psi4.__dict__)

        molecule.save()
        return molecule

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
