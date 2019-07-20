from openvqe.abc import OpenVQEModule, OpenVQEParameters, parametrized
from openvqe.exceptions import OpenVQEException
from openvqe.ansatz.backend_handler import get_backend_hander
from openvqe.hamiltonian.hamiltonian_base import HamiltonianBase
from dataclasses import dataclass


@dataclass
class ParametersAnsatz(OpenVQEParameters):
    """
    Enter general parameters which hold for all types of VQE ansaetze
    """

    # have to be assigned
    backend: str = "cirq"


@parametrized(ParametersAnsatz)
class AnsatzBase(OpenVQEModule):
    """
    Base Class for the VQE Ansatz
    Derive all specializations from this Base Class
    """

    def __post_init__(self, hamiltonian: HamiltonianBase = None, qubits=None):
        self.hamiltonian = hamiltonian
        self.backend_handler = get_backend_hander(backend=self.parameters.backend, n_qubits=hamiltonian.n_qubits(),
                                                  qubits=qubits)
        self.parameters.n_qubits = hamiltonian.n_qubits()
        self.verify()

    def __call__(self, angles):
        """
        :param angles: The angles which parametrize the circuit
        :return: the circuit in the correct format for the simulator backend secified by self.parameters.backend
        """
        return self.construct_circuit(angles=angles)

    def construct_circuit(self, angles):
        """
        Construct the circuit specifified by the given ansatz
        this function should be overwritten by specializations of this baseclass
        :param angles:
        :return: the circuit in the correct format for the simulator backend secified by self.parameters.backend
        """
        raise OpenVQEException(type(self).__name__ + ": You tried to call the ABC directly")

    def greet(self):
        print("Hello from the " + type(self).__name__ + " class")
