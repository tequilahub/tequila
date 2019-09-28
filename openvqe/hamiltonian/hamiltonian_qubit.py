"""
Default Hamiltonian to play around with
Uses OpenFermion Structure
Has no special features
"""
from openvqe.hamiltonian import HamiltonianBase
from openvqe.hamiltonian.paulistring import PauliString
from openvqe.tools.convenience import number_to_string
from openfermion import QubitOperator
from openvqe import BitString
from typing import List


class QubitHamiltonian(HamiltonianBase):
    axis_to_string = {0: "x", 1: "y", 2: "z"}
    string_to_axis = {"x": 0, "y": 1, "z": 2}

    def index(self, ituple):
        return ituple[0]

    def pauli(selfs, ituple):
        return ituple[1]

    def __init__(self, hamiltonian: QubitOperator = None):
        if isinstance(hamiltonian, str):
            self._hamiltonian = self.init_from_string(string=hamiltonian)._hamiltonian
        elif hamiltonian is None:
            self._hamiltonian = QubitOperator.identity()
        else:
            self._hamiltonian = hamiltonian

        assert (isinstance(self._hamiltonian, QubitOperator))

    def __repr__(self):
        result = ""
        for ps in self.paulistrings:
            result += str(ps)
        return result

    def items(self):
        return self._hamiltonian.terms.items()

    def keys(self):
        return self._hamiltonian.terms.keys()

    def values(self):
        return self._hamiltonian.terms.values()

    @classmethod
    def init_zero(cls):
        return QubitHamiltonian(hamiltonian=QubitOperator("", 0.0))

    @classmethod
    def init_unit(cls):
        return QubitHamiltonian(hamiltonian=QubitOperator.identity())

    @classmethod
    def init_from_string(cls, string):
        return QubitHamiltonian(hamiltonian=QubitOperator(string.upper(), 1.0))

    @classmethod
    def init_from_paulistring(cls, ps: PauliString):
        return QubitHamiltonian(hamiltonian=QubitOperator(term=ps.key_openfermion(), coefficient=ps.coeff))

    def __add__(self, other):
        return QubitHamiltonian(hamiltonian=self.hamiltonian + other.hamiltonian)

    def __sub__(self, other):
        return QubitHamiltonian(hamiltonian=self.hamiltonian - other.hamiltonian)

    def __iadd__(self, other):
        self.hamiltonian += other.hamiltonian
        return self

    def __isub__(self, other):
        self.hamiltonian -= other.hamiltonian
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

    def __eq__(self, other):
        return self.hamiltonian == other.hamiltonian

    def is_hermitian(self):
        for v in self.values():
            if v.imag != 0.0:
                return False
        return True

    def is_antihermitian(self):
        for v in self.values():
            if v.real != 0.0:
                return False
        return True

    def conjugate(self):
        conj_hamiltonian = QubitOperator("", 0)
        for key, value in self._hamiltonian.terms.items():
            sign = 1
            for term in key:
                p = self.pauli(term)
                if p.lower() == "y":
                    sign *= -1
            conj_hamiltonian.terms[key] = sign * value.conjugate()

        return QubitHamiltonian(hamiltonian=conj_hamiltonian)

    def transpose(self):
        trans_hamiltonian = QubitOperator("", 0)
        for key, value in self._hamiltonian.terms.items():
            sign = 1
            for term in key:
                p = self.pauli(term)
                if p.lower() == "y":
                    sign *= -1
            trans_hamiltonian.terms[key] = sign * value

        return QubitHamiltonian(hamiltonian=trans_hamiltonian)

    def dagger(self):
        dag_hamiltonian = QubitOperator("", 0)
        for key, value in self._hamiltonian.terms.items():
            dag_hamiltonian.terms[key] = value.conjugate()

        return QubitHamiltonian(hamiltonian=dag_hamiltonian)

    def normalize(self):
        self._hamiltonian.renormalize()

    @property
    def n_qubits(self):
        n_qubits = 0
        for key, value in self.hamiltonian.terms.items():
            indices = [self.index(k) for k in key]
            n_qubits = max(n_qubits, max(indices))
        return n_qubits + 1

    @property
    def paulistrings(self):
        """
        :return: the Hamiltonian as list of PauliStrings
        """
        return [PauliString.init_from_openfermion(key=k, coeff=v) for k, v in self.items()]

    @paulistrings.setter
    def paulistrings(self, other):
        """
        Reassign with OpenVQE PauliString format
        :param other: list of PauliStrings
        :return: self for chaining
        """
        new_hamiltonian = QubitOperator.identity()
        for ps in other:
            tmp = QubitOperator(term=ps.key_openfermion(), value=ps.coeff)
            new_hamiltonian += tmp
        self._hamiltonian = new_hamiltonian
        return self


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


def PI(qubit):
    return QubitHamiltonian.init_unit()


def Qp(qubit):
    return 0.5 * (PI(qubit=qubit) + PZ(qubit=qubit))


def Qm(qubit):
    return 0.5 * (PI(qubit=qubit) - PZ(qubit=qubit))


def Sp(qubit):
    return 0.5 * (PX(qubit=qubit) + 1.j * PY(qubit=qubit))


def Sm(qubit):
    return 0.5 * (PX(qubit=qubit) - 1.j * PY(qubit=qubit))


def decompose_transfer_operator(ket: BitString, bra: BitString, qubits: List[int] = None) -> QubitHamiltonian:
    """
    Decompose |ket><bra| into paulistrings
    """

    opmap = {
        (0, 0): Qp,
        (0, 1): Sp,
        (1, 0): Sm,
        (1, 1): Qm
    }

    if isinstance(bra, int):
        bra = BitString.from_int(integer=bra, nbits=len(qubits))
    if isinstance(ket, int):
        ket = BitString.from_int(integer=ket, nbits=len(qubits))

    b_arr = bra.array
    k_arr = ket.array
    assert (len(b_arr) == len(k_arr))
    n_qubits = len(k_arr)

    if qubits is None:
        qubits = range(n_qubits)

    assert (n_qubits <= len(qubits))

    result = QubitHamiltonian.init_unit()
    for q, b in enumerate(b_arr):
        k = k_arr[q]
        result *= opmap[(k, b)](qubit=qubits[q])

    return result



