"""
Interface to get
Quantum Chemistry Hamiltonians for OpenVQE
Derived class of HamiltonianBase: Overwrites the get_hamiltonian function
"""

from openvqe.parameters import ParametersQC
from openfermion.hamiltonians import MolecularData
from .hamiltonian_base import HamiltonianBase
import openfermion
import openfermionpsi4


class HamiltonianQC(HamiltonianBase):

    def __init__(self, parameters: ParametersQC):
        """
        Constructor will run psi4 and create the molecule which is stored as member variable
        :param parameters: instance of ParametersQC which holds all necessary parameters
        """
        assert (isinstance(parameters, ParametersQC))
        self.parameters = parameters
        self.molecule = self.make_molecule()

    def get_hamiltonian(self) -> openfermion.InteractionOperator:
        """
        Note that the Qubit Hamiltonian can be created over the call method which is already implemented in the baseclass
        :return: The fermionic Hamiltonian as InteractionOperator structure
        """
        # fast return if possible
        if self.molecule is None:
            self.molecule = self.make_molecule(self.parameters)
        return self.molecule.get_molecular_hamiltonian()

    def make_molecule(self) -> openfermion.MolecularData:
        """
        Creates a molecule in openfermion format by running psi4 and extracting the data
        Will check for previous outputfiles before running
        :param parameters: An instance of ParametersQC, which also holds an instance of ParametersPsi4 via parameters.psi4
        The molecule will be saved in parameters.filename, if this file exists before the call the molecule will be imported from the file
        :return: the molecule in openfermion.MolecularData format
        """
        molecule = MolecularData(geometry=self.parameters.get_geometry(),
                                 basis=self.parameters.basis_set,
                                 multiplicity=self.parameters.multiplicity,
                                 charge=self.parameters.charge,
                                 description=self.parameters.description,
                                 filename=self.parameters.filename)

        # try to load
        do_compute = True
        if self.parameters.filename:
            try:
                import os
                if os.path.exists(self.parameters.filename):
                    molecule.load()
                    do_compute = False
            except OSError:
                print("loading molecule from file=", self.parameters.filename, " failed. Try to recompute")
                do_compute = True

        if do_compute:
            molecule = openfermionpsi4.run_psi4(molecule,
                                                run_scf=self.parameters.psi4.run_scf,
                                                run_mp2=self.parameters.psi4.run_mp2,
                                                run_cisd=self.parameters.psi4.run_cisd,
                                                run_ccsd=self.parameters.psi4.run_ccsd,
                                                run_fci=self.parameters.psi4.run_fci,
                                                verbose=self.parameters.psi4.verbose,
                                                tolerate_error=self.parameters.psi4.tolerate_error,
                                                delete_input=self.parameters.psi4.delete_input,
                                                delete_output=self.parameters.psi4.delete_input,
                                                memory=self.parameters.psi4.memory)

        molecule.save()

        return molecule

    def verify(self) -> bool:
        from openvqe import OvqeTypeError
        """
        Overwritten verify function to check specificly for ParametersQC type
        :return:
        """
        # do some verification specificaly for this class

        # check if the molecule was initialized
        if not isinstance(self.molecule, openfermion.MolecularData):
            raise OvqeTypeError(attr=type(self).__name__ + ".molecule", type=type(self.molecule),
                                expected=type(openfermion.MolecularData))

        # do the standard checks for the baseclass
        return self._verify(ParameterType=ParametersQC)
