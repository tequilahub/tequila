"""
Base Class for OpenVQE Hamiltonians
Implements all functions which are needed by as good as all derived classes
"""

from dataclasses import dataclass
from openvqe.openvqe_abc import OpenVQEModule, OpenVQEParameters, parametrized
from openvqe import OpenVQEParameterError, OpenVQEException, OpenVQETypeError
import openfermion


class HamiltonianBase(OpenVQEModule):

    def __call__(self) -> openfermion.QubitOperator:
        """
        Calls the self.get_hamiltonian()
        :return: Gives back the Qubit Operator
        """
        return self.hamiltonian

    @property
    def hamiltonian(self):
        return self._hamiltonian


    @hamiltonian.setter
    def hamiltonian(self, other: openfermion.QubitOperator):
        self._hamiltonian = other
        return self

    def n_qubits(self):
        """
        Needs to be overwritten by specialization
        :return: the number of qubits this hamiltonian needs
        """
        raise OpenVQEException(type(
            self).__name__ + ": forgot to overwrite n_qubits() function or you are calling the BaseClass which you shall not do")

    def verify(self) -> bool:
        """
        check if the instance is sane, should be overwritten by derived classes
        :return: true if sane, raises exception if not
        """
        return self._verify()
