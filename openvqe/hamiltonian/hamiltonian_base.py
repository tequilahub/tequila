"""
Base Class for OpenVQE Hamiltonians
"""

from openvqe import ParameterError
from openvqe import ParametersHamiltonian
import openfermion


class HamiltonianBase:

    def __init__(self, parameters: ParametersHamiltonian):
        assert (isinstance(parameters, ParametersHamiltonian))
        self.parameters = parameters

    def __call__(self) -> openfermion.QubitOperator:
        """
        :return: Gives back the Qubit Operator
        """

        if self.parameters.jordan_wigner():
            return openfermion.jordan_wigner(openfermion.get_fermion_operator(self.get_hamiltonian()))
        elif self.parameters.bravyi_kitaev():
            return openfermion.bravyi_kitaev(openfermion.get_fermion_operator(self.get_hamiltonian()))
        else:
            raise ParameterError(
                type(self).__name__ + ": Unknown parameter for transformation type, transformation=",
                self.parameters.transformation)

    def greet(self):
        print("This is the " + type(self).__name__ + " class")

    def get_hamiltonian(self):
        """
        Compute the Fermionic Hamiltonian which will be transformed
        by the class __call__ function
        This function should be overwritten by classes which take this class as base
        :return: the fermionic Hamiltonian
        """
        raise NotImplementedError(
            "You try to call get_hamiltonian from the HamiltonianBase class. This function needs to be overwritten by subclasses")
