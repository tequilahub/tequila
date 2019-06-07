"""
Base Class for OpenVQE Hamiltonians
Implements all functions which are needed by as good as all derived classes
"""

from openvqe import OvqeParameterError, OvqeException, OvqeTypeError
from openvqe import ParametersHamiltonian
import openfermion


class HamiltonianBase:

    def __init__(self, parameters: ParametersHamiltonian):
        """
        Default constructor of the baseclass, initializes the parameters
        :param parameters: Parameters of type or derived from ParametersHamiltonian
        """
        assert (isinstance(parameters, ParametersHamiltonian))
        self.parameters = parameters

    def __call__(self) -> openfermion.QubitOperator:
        """
        Calls the self.get_hamiltonian() function and transforms it to a qubit operator
        The transformation is specified in the parameters
        :return: Gives back the Qubit Operator
        """

        self.verify()

        if self.parameters.jordan_wigner():
            return openfermion.jordan_wigner(openfermion.get_fermion_operator(self.get_hamiltonian()))
        elif self.parameters.bravyi_kitaev():
            return openfermion.bravyi_kitaev(openfermion.get_fermion_operator(self.get_hamiltonian()))
        else:
            raise OvqeParameterError(parameter_name="transformation", parameter_class=type(self.parameters),
                                     parameter_value=self.parameters.transformation,
                                     called_from=type(self).__name__ + ".__call__()")

    def greet(self):
        print("Hello from the " + type(self).__name__ + " class")

    def get_hamiltonian(self):
        """
        Compute the Fermionic Hamiltonian which will be transformed
        by the class' __call__ function
        This function should be overwritten by classes which take this class as base
        :return: the fermionic Hamiltonian
        """
        raise NotImplementedError(
            "You try to call get_hamiltonian from the HamiltonianBase class. This function needs to be overwritten by subclasses")

    def verify(self) -> bool:
        """
        check if the instance is sane, should be overwritten by derived classes
        :return: true if sane, raises exception if not
        """
        return self._verify(ParameterType=ParametersHamiltonian)

    def _verify(self, ParameterType: type) -> bool:
        """
        Actual verify function
        :return: true if sane, raises exception if not
        """

        # check if verify was called correctly
        if not isinstance(ParameterType, type):
            raise OvqeException(
                "Wrong input type for " + type(self).__name__ + "._verify"
            )
        # check if the parameters are of the correct type
        if not isinstance(self.parameters, ParameterType):
            # raise OpenVQEException(
            #     "parameters attribute of instance of class " + type(
            #         self).__name__ + " should be of type " + ParameterType.__name__ + " but is of type " + type(
            #         self.parameters).__name__)
            raise OvqeTypeError(attr=type(self).__name__ + ".parameters", type=type(self.parameters),
                            expected=ParameterType)

        return True
