import numbers
import typing
import numpy

from openfermion import QubitOperator
from tequila.tools import number_to_string
from functools import reduce

class PauliString:
    """
    Convenient DataClass for single PauliStrings
    Internal Storage is a dictionary where keys are particle-numbers and values the primitive paulis
    i.e. X(1)Y(2)Z(5) is {1:'x', 2:'y', 5:'z'}
    additional a coefficient can be stored
    iteration is then over the dimension
    """

    @property
    def qubits(self):
        accumulate = [k for k in self.keys()]
        return sorted(list(set(accumulate)))

    def key_openfermion(self):
        """
        Convert into key to store in Hamiltonian
        Same key syntax than openfermion
        :return: The key for the openfermion dataformat
        """
        key = []
        for k, v in self._data.items():
            key.append((k, v))
        return tuple(key)

    def __repr__(self):
        result = ""
        if self._coeff is not None:
            result = number_to_string(self.coeff)
        for k, v in self._data.items():
            result += str(v) + "(" + str(k) + ")"
        return result

    def __init__(self, data=None, coeff=None):
        """
        Initialize the class
        :param data: Dictionary which holds the paulistring with dimensions as keys
        i.e. X(0)Y(1)Z(3)X(20) is { 0:'x', 1:'y', 3:'z', 20:'x' }
        :param coeff:
        """
        if data is None:
            self._data = {}
        else:
            # stores the paulistring as dictionary
            # keys are the dimensions
            # values are x,y,z
            self._data = data
        self._coeff = coeff

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    @classmethod
    def from_string(cls, string:str, coeff=None):
        """
        :param string: Format is for example: X(0)Y(100)Z(2)
        :param coeff: coefficient
        :return: new instance with the data given by the string
        """
        data = dict()
        string = string.strip()
        for part in string.split(')'):
            part = part.strip()
            if part == "":
                break
            pauli_dim = part.split('(')
            data[int(pauli_dim[1])] = pauli_dim[0].upper()
        return PauliString(data=data, coeff=coeff)


    @classmethod
    def from_openfermion(cls, key, coeff=None):
        """
        Initialize a PauliString from OpenFermion data
        :param key: The pauli-string in OpenFermion format i.e. a list of tuples
        [(0,X),(1,X),(2,Z)] -> X(0)X(1)X(Z)
        :param coeff: The coefficient for this paulistring
        :return:
        """
        data = {}
        for term in key:
            index = term[0]
            pauli = term[1].upper()
            data[index] = pauli
        return PauliString(data=data, coeff=coeff)

    @property
    def coeff(self):
        if self._coeff is None:
            return 1
        else:
            return self._coeff

    @coeff.setter
    def coeff(self, other):
        self._coeff = other
        return self

    def __eq__(self, other):
        return self._data == other._data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]

    def naked(self):
        """
        :return: naked paulistring without coefficient
        """
        return PauliString(data=self._data, coeff=None)


"""
Explicit matrix forms for the Pauli operators for the tomatrix method
"""
import numpy as np

pauli_matrices = {
    'I': numpy.array([[1, 0], [0, 1]], dtype=numpy.complex),
    'Z': numpy.array([[1, 0], [0, -1]], dtype=numpy.complex),
    'X': numpy.array([[0, 1], [1, 0]], dtype=numpy.complex),
    'Y': numpy.array([[0, -1j], [1j, 0]], dtype=numpy.complex)
}

class QubitHamiltonian:
    """
    Default QubitHamiltonian
    Uses OpenFermion Structures for arithmetics
    """

    #convenience
    axis_to_string = {0: "x", 1: "y", 2: "z"}
    string_to_axis = {"x": 0, "y": 1, "z": 2}

    @property
    def hamiltonian(self) -> QubitOperator:
        """
        :return: The underlying OpenFermion QubitOperator
        """
        return self._hamiltonian

    @property
    def qubits(self):
        """
        :return: All Qubits the Hamiltonian acts on
        """
        accumulate = []
        for ps in self.paulistrings:
            accumulate += ps.qubits
        return sorted(list(set(accumulate)))

    @hamiltonian.setter
    def hamiltonian(self, other: QubitOperator) -> QubitOperator:
        self._hamiltonian = other

    def index(self, ituple):
        return ituple[0]

    def pauli(selfs, ituple):
        return ituple[1]

    def __init__(self, hamiltonian: typing.Union[QubitOperator,str, numbers.Number] = None):
        """
        Initialize from string or from a preexisting OpenFermion QubitOperator instance
        :param hamiltonian: string or openfermion.QubitOperator
        if string: Same conventions as openfermion
        if None: The Hamiltonian is initialized as identity operator
        if Number: initialized as scaled unit operator
        """
        if isinstance(hamiltonian, str):
            self._hamiltonian = self.init_from_string(string=hamiltonian)._hamiltonian
        elif hamiltonian is None:
            self._hamiltonian = QubitOperator.identity()
        elif isinstance(hamiltonian, numbers.Number):
            self._hamiltonian = hamiltonian*QubitOperator.identity()
        else:
            self._hamiltonian = hamiltonian

        assert (isinstance(self._hamiltonian, QubitOperator))

    def __len__(self):
        return len(self._hamiltonian.terms)

    def __repr__(self):
        result = ""
        for ps in self.paulistrings:
            result += str(ps)
        return result

    def __getitem__(self, item):
        return self._hamiltonian.terms[item]

    def __setitem__(self, key, value):
        self._hamiltonian.terms[key] = value
        return self

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
        return self._hamiltonian == other._hamiltonian

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

    def to_matrix(self):
        """
        Returns the Hamiltonian as a dense matrix.

        Returns a dense 2**N x 2**N matrix representation of this
        QubitHamiltonian. Watch for memory usage when N is >12!
        
        :return: numpy.ndarray(2**N, 2**N) with type numpy.complex
        """
        nq = self.n_qubits
        I = numpy.eye(2,dtype=numpy.complex)
        Hm = numpy.zeros((2**nq, 2**nq), dtype=numpy.complex)

        for key,val in self.items():
            term = [I] * nq

            for ind, op in key:
                term[ind] = pauli_matrices[op]

            Hm += val * reduce(numpy.kron, term)
        return Hm

    @property
    def n_qubits(self):
        max_index = 0
        for key, value in self.items():
            indices = [self.index(k) for k in key]
            if len(indices) > 0: # for the case that there is a 1 in the operator
                max_index = max(max_index, max(indices))
        return max_index + 1

    @property
    def paulistrings(self):
        """
        :return: the Hamiltonian as list of PauliStrings
        """
        return [PauliString.from_openfermion(key=k, coeff=v) for k, v in self.items()]

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
