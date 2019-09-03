"""
Default Hamiltonian to play around with
Uses OpenFermion Structure
Has no special features
"""
from openvqe.hamiltonian import HamiltonianBase
from openfermion import QubitOperator


class QubitHamiltonian(HamiltonianBase):
    axis_to_string = {0: "x", 1: "y", 2: "z"}
    string_to_axis = {"x": 0, "y": 1, "z": 2}

    def index(self, ituple):
        return ituple[0]

    def pauli(selfs, ituple):
        return ituple[1]

    def __init__(self, hamiltonian: QubitOperator = None):
        if isinstance(hamiltonian, str):
            self.init_from_string(string=hamiltonian)
        else:
            self._hamiltonian = hamiltonian

    def init_from_string(self, string):
        self._hamiltonian = QubitOperator(string.upper(), 1.0)

    def __add__(self, other):
        return QubitHamiltonian(hamiltonian=self.hamiltonian + other.hamiltonian)

    def __iadd__(self, other):
        self.hamiltonian += other.hamiltonian
        return self

    def __mul__(self, other):
        return QubitHamiltonian(hamiltonian=self.hamiltonian * other.hamiltonian)

    def __imul__(self, other):
        self.hamiltonian *= other.hamiltonian
        return self

    def __rmul__(self, other):
        return QubitHamiltonian(hamiltonian=self.hamiltonian * other)

    def __pow__(self, power):
        return QubitHamiltonian(hamiltonian=self.hamiltonian ** power)

    def normalize(self):
        self._hamiltonian.renormalize()

    @property
    def n_qubits(self):
        n_qubits = 0
        for key, value in self.hamiltonian.terms.items():
            indices = [self.index(k) for k in key]
            n_qubits = max(n_qubits, max(indices))
        return n_qubits + 1


"""
Convenience initialization
Using PX, PY, PZ notation to not confuse with circuits
"""


def pauli(qubit, type):
    if type in QubitHamiltonian.axis_to_string:
        type = QubitHamiltonian.axis_to_string(type)
    else:
        type = type.upper()
    return QubitHamiltonian(type + str(qubit))


def PX(qubit):
    return QubitHamiltonian("X" + str(qubit))


def PY(qubit):
    return QubitHamiltonian("Y" + str(qubit))


def PZ(qubit):
    return QubitHamiltonian("Z" + str(qubit))
