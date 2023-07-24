import numbers
import typing

from tequila.tools import number_to_string
from tequila.utils import to_float
from tequila import TequilaException

from openfermion import QubitOperator
from functools import reduce

from collections import namedtuple

BinaryPauli = namedtuple("BinaryPauli", "coeff, binary")

"""
Explicit matrix forms for the Pauli operators for the tomatrix method
For sparse matrices use the openfermion tool
get the openfermion object with hamiltonian.hamiltonian
"""
import numpy as np

pauli_matrices = {
    'I': np.array([[1, 0], [0, 1]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex)
}


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
        """
        :return: The qubits on which the PauliString acts non-trivial as list
        """
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

        self._all_z = all([x.lower() == "z" for x in data.values()])

    def trace_out_qubits(self, qubits, states=None):
        """
        See trace_out_qubits in QubitHamiltonian
        Parameters
        ----------
        qubits
            qubits to trace out
        states
            states a|0> + b|1> as list of tuples of the a,b coefficients. Default is just |0>.
        Returns
        -------
            traced out PauliString
        """
        # |psi> = a|0> + b|1>
        # <psi|op|psi> = |a|**2<0|op|0> + (a*)*b<1|op|0> + (b*)*a<0|op|1> + |b|**2<1|op|1>

        if states is None:
            states = [(1.0, 0.0)]*len(qubits)

        def make_coeff_vec(state_tuple):
            return np.asarray([np.abs(state_tuple[0])**2, state_tuple[0].conjugate()*state_tuple[1], state_tuple[1].conjugate()*state_tuple[0] ,np.abs(state_tuple[1])**2])

        factor=1.0
        for q, state in zip(qubits, states):
            if q in self.keys():
                matrix = pauli_matrices[self[q].upper()].reshape([4])
                vec = make_coeff_vec(state)
                factor *= vec.dot(matrix)
                if factor == 0.0:
                    break

        new_data = {k:v for k,v in self.items() if k not in qubits}
        return PauliString(data=new_data, coeff=self.coeff*factor)


    def map_qubits(self, qubit_map: dict):
        """

        E.G.  X(1)Y(2) --> X(3)Y(1) with qubit_map = {1:3, 2:1}

        Parameters
        ----------
        qubit_map
            a dictionary which maps old to new qubits

        Returns
        -------
        the PauliString with mapped qubits

        """

        mapped = {qubit_map[k]: v for k, v in self._data.items()}
        return PauliString(data=mapped, coeff=self.coeff)

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    @classmethod
    def from_string(cls, string: str, coeff=None):
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
            string = pauli_dim[0].upper()
            if not string in ['X', 'Y', 'Z']:
                raise TequilaException("PauliString.from_string initialization failed, unknown pauliterm: " + string)
            data[int(pauli_dim[1])] = string

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
        """
        :return: The coefficient of this paulistring
        """
        if self._coeff is None:
            return 1.0
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
        Convenience function to strip coefficients from the PauliStrings and avoid having two coefficients
        :return: naked paulistring without the coefficient
        """
        return PauliString(data=self._data, coeff=None)

    def binary(self, n_qubits: int = None):
        if len(self._data.keys()) == 0:
            maxq = 1
        else:
            maxq = max(self._data.keys()) + 1

        if n_qubits is None:
            n_qubits = maxq

        if n_qubits < maxq:
            raise TequilaException(
                "PauliString acts on qubit number larger than n_qubits given\n PauliString=" + self.__repr__() + ", n_qubits=" + str(
                    n_qubits))

        binary = np.zeros(2 * n_qubits)
        for k, v in self._data.items():
            if v.upper() == "X":
                binary[k] = 1
            elif v.upper() == "Y":
                binary[k] = 1
                binary[n_qubits + k] = 1
            elif v.upper() == "Z":
                binary[n_qubits + k] = 1
            else:
                raise TequilaException("Unknown Pauli: %" + str(v))
        return BinaryPauli(coeff=self.coeff, binary=binary)

    def is_all_z(self):
        return self._all_z


class QubitHamiltonian:
    """
    Default QubitHamiltonian
    Uses OpenFermion Structures for arithmetics
    """

    # convenience
    axis_to_string = {0: "x", 1: "y", 2: "z"}
    string_to_axis = {"x": 0, "y": 1, "z": 2}

    @classmethod
    def from_openfermion(cls, qubit_operator: QubitOperator):
        return QubitHamiltonian(qubit_operator=qubit_operator)

    def to_openfermion(self) -> QubitOperator:
        return self.qubit_operator

    @property
    def qubit_operator(self) -> QubitOperator:
        """
        :return: The underlying OpenFermion QubitOperator
        """
        return self._qubit_operator

    @property
    def qubits(self):
        """
        :return: All Qubits the Hamiltonian acts on
        """
        accumulate = []
        for ps in self.paulistrings:
            accumulate += ps.qubits
        return sorted(list(set(accumulate)))

    @qubit_operator.setter
    def qubit_operator(self, other: QubitOperator) -> QubitOperator:
        self._qubit_operator = other

    def index(self, ituple):
        return ituple[0]

    def pauli(selfs, ituple):
        return ituple[1]

    def __call__(self, wfn):
        if hasattr(wfn, "apply_qubitoperator"):
            return wfn.apply_qubitoperator(self)
        else:
            raise TequilaException("Not sure what to do here with {} and {} ...".format(self, wfn))

    def __init__(self, qubit_operator: typing.Union[QubitOperator, str, numbers.Number] = None):
        """
        Initialize from string or from a preexisting OpenFermion QubitOperator instance
        :param qubit_operator: string or openfermion.QubitOperator
        if string: Same conventions as openfermion
        if None: The Hamiltonian is initialized as identity operator
        if Number: initialized as scaled unit operator
        """
        if isinstance(qubit_operator, str):
            self._qubit_operator = self.from_string(string=qubit_operator)._qubit_operator
        elif qubit_operator is None:
            self._qubit_operator = QubitOperator.zero()
        elif isinstance(qubit_operator, numbers.Number):
            self._qubit_operator = qubit_operator * QubitOperator.identity()
        else:
            self._qubit_operator = qubit_operator

        assert (isinstance(self._qubit_operator, QubitOperator))

    def trace_out_qubits(self, qubits, states: list=None, *args, **kwargs):
        """
        Tracing out qubits with the assumption that they are in the |0> (default) or |1> state

        Parameters
        ----------
        qubits
            qubits to trace out
        states
            states of the qubits as list of individual tq.QubitWaveFunction (default is all in |0>)
        Returns
        -------
            traced out Hamiltonian
        """

        if states is None:
            states = [(1.0,0.0)]*len(qubits)
        else:
            assert len(states) == len(qubits)
            # states should be given as list of individual tq.QubitWaveFunctions
            states = [tuple(s.to_array()) for s in states]

        reduced_ps = [ps.trace_out_qubits(qubits=qubits, states=states) for ps in self.paulistrings]
        return self.from_paulistrings(ps=reduced_ps).simplify(*args, **kwargs)

    def count_measurements(self):
        if self.is_all_z():
            return 1
        else:
            return len(self)

    def __len__(self):
        return len(self.paulistrings)

    def __repr__(self):
        result = ""
        for ps in self.paulistrings:
            result += str(ps)
        return result

    def __getitem__(self, item):
        return self._qubit_operator.terms[item]

    def __setitem__(self, key, value):
        self._qubit_operator.terms[key] = value
        return self

    def items(self):
        return self._qubit_operator.terms.items()

    def keys(self):
        return self._qubit_operator.terms.keys()

    def values(self):
        return self._qubit_operator.terms.values()

    @classmethod
    def zero(cls):
        return QubitHamiltonian(qubit_operator=QubitOperator("", 0.0))

    @classmethod
    def unit(cls):
        return QubitHamiltonian(qubit_operator=QubitOperator.identity())

    @classmethod
    def from_string(cls, string, openfermion_format=False):
        """
        stringify your hamiltonian as str(H.hamiltonian) to get the openfermion stringification
        :param string: Hamiltonian as string
        :param openfermion_format: use the openfermion string format
        :return: QubitHamiltonian
        """
        if string.strip() == "":
            return cls.zero()
        elif openfermion_format:
            return QubitHamiltonian(qubit_operator=QubitOperator(string))
        else:
            H = QubitHamiltonian.zero()
            string = string.replace(" ", "")
            string = string.replace("*", "")
            string = string.replace("e-", "@")
            string = string.replace("+-", "-")
            string = string.replace("-+", "-")
            string = string.replace("-", "+-")
            string = string.replace("X", " X")
            string = string.replace("Y", " Y")
            string = string.replace("Z", " Z")
            string += " "
            terms = string.split('+')
            for term in terms:
                if term.strip() == "":
                    continue

                coeff = term.split(" ")[0]
                if coeff.strip() == "" or coeff[0] in ["X", "Y", "Z"]:
                    coeff = '1.0'
                    ps = term
                else:
                    ps = term.replace(coeff, " ").replace(" ", "")

                try:
                    if "i" in coeff:
                        coeff = coeff.replace("i", "j")
                    if "@" in coeff:
                        coeff = coeff.replace("@", "e-")
                    coeff = complex(coeff)
                except Exception as E:
                    raise Exception("failed to convert coefficient : {}".format(coeff))

                if coeff.imag == 0.0:
                    coeff = float(coeff.real)

                H += cls.from_paulistrings(ps=PauliString.from_string(string=ps, coeff=coeff))
            return H.simplify()

    @classmethod
    def from_paulistrings(cls, ps: typing.List[PauliString]):
        if isinstance(ps, PauliString):
            return cls.from_paulistrings(ps=[ps])
        else:
            H = cls.zero()
            for x in ps:
                H += QubitHamiltonian(qubit_operator=QubitOperator(term=x.key_openfermion(), coefficient=x.coeff))
            return H.simplify()

    def __add__(self, other):
        if isinstance(other, numbers.Number):
            return QubitHamiltonian(qubit_operator=self.qubit_operator + other * self.unit().qubit_operator)
        else:
            return QubitHamiltonian(qubit_operator=self.qubit_operator + other.qubit_operator)

    def __sub__(self, other):
        if isinstance(other, numbers.Number):
            return QubitHamiltonian(qubit_operator=self.qubit_operator - other * self.unit().qubit_operator)
        else:
            return QubitHamiltonian(qubit_operator=self.qubit_operator - other.qubit_operator)

    def __iadd__(self, other):
        if isinstance(other, numbers.Number):
            self.qubit_operator += other * self.unit().qubit_operator
        else:
            self.qubit_operator += other.qubit_operator
        return self

    def __isub__(self, other):
        if isinstance(other, numbers.Number):
            self.qubit_operator -= other * self.unit().qubit_operator
        else:
            self.qubit_operator -= other.qubit_operator
        return self

    def __mul__(self, other):
        if hasattr(other, "apply_qubitoperator"):
            # actually an apply operation
            return other.apply_qubitoperator(operator=self)
        elif isinstance(other, numbers.Number):
            return QubitHamiltonian(qubit_operator=self.qubit_operator * other)
        else:
            return QubitHamiltonian(qubit_operator=self.qubit_operator * other.qubit_operator)

    def __imul__(self, other):
        if isinstance(other, numbers.Number):
            self.qubit_operator *= other
        else:
            self.qubit_operator *= other.qubit_operator
        return self

    def __rmul__(self, other):
        assert isinstance(other, numbers.Number)
        return QubitHamiltonian(qubit_operator=self.qubit_operator * other)

    def __radd__(self, other):
        return self.__add__(other=other)

    def __rsub__(self, other):
        return self.__neg__().__add__(other=other)

    def __pow__(self, power):
        return QubitHamiltonian(qubit_operator=self.qubit_operator ** power)

    def __neg__(self):
        return self.__mul__(other=-1.0)

    def __eq__(self, other):
        return self._qubit_operator == other._qubit_operator

    def is_hermitian(self):
        try:
            for k, v in self.qubit_operator.terms.items():
                self.qubit_operator.terms[k] = to_float(v)
            return True
        except TypeError:
            return False

    def simplify(self, threshold=0.0):
        simplified = {}
        for k, v in self.qubit_operator.terms.items():
            if not np.isclose(v, 0.0, atol=threshold):
                simplified[k] = v
        self._qubit_operator.terms = simplified
        return self

    def split(self, *args, **kwargs) -> tuple:
        """
        Returns
        -------
            Hermitian and anti-Hermitian part as tuple
        """
        hermitian = QubitHamiltonian.zero()
        anti_hermitian = QubitHamiltonian.zero()
        for k, v in self.qubit_operator.terms.items():
            hermitian.qubit_operator.terms[k] = v.real
            anti_hermitian.qubit_operator.terms[k] = 1.j * v.imag

        return hermitian.simplify(), anti_hermitian.simplify()

    def is_antihermitian(self):
        for v in self.values():
            if v.real != 0.0:
                return False
        return True

    def conjugate(self):
        conj_hamiltonian = QubitOperator("", 0)
        for key, value in self._qubit_operator.terms.items():
            sign = 1
            for term in key:
                p = self.pauli(term)
                if p.lower() == "y":
                    sign *= -1
            conj_hamiltonian.terms[key] = sign * value.conjugate()

        return QubitHamiltonian(qubit_operator=conj_hamiltonian)

    def transpose(self):
        trans_hamiltonian = QubitOperator("", 0)
        for key, value in self._qubit_operator.terms.items():
            sign = 1
            for term in key:
                p = self.pauli(term)
                if p.lower() == "y":
                    sign *= -1
            trans_hamiltonian.terms[key] = sign * value

        return QubitHamiltonian(qubit_operator=trans_hamiltonian)

    def dagger(self):
        dag_hamiltonian = QubitOperator("", 0)
        for key, value in self._qubit_operator.terms.items():
            dag_hamiltonian.terms[key] = value.conjugate()

        return QubitHamiltonian(qubit_operator=dag_hamiltonian)

    def normalize(self):
        self._qubit_operator.renormalize()
        return self

    def to_matrix(self, ignore_unused_qubits=True):
        """
        Returns the Hamiltonian as a dense matrix.

        Returns a dense 2**N x 2**N matrix representation of this
        QubitHamiltonian. Watch for memory usage when N is >12!
        
        Args:
            ignore_unused_qubits: If no non-trivial operator is defined on a qubits this qubit will be ignored in the matrix construction.
                Take for example X(1). 
                If False the operator X(1) will get mapped to X(0)
                and the function will return the matrix for X(0)
                otherwise the function will return the matrix 1 \otimes X(1)

        :return: np.ndarray(2**N, 2**N) with type complex

        """
        qubits = self.qubits
        if ignore_unused_qubits:
            nq = len(qubits)
        else:
            nq = max(qubits)+1

        I = np.eye(2, dtype=complex)
        Hm = np.zeros((2 ** nq, 2 ** nq), dtype=complex)

        for key, val in self.items():
            term = [I] * nq

            for ind, op in key:
                ind = qubits.index(ind)
                term[ind] = pauli_matrices[op]

            Hm += val * reduce(np.kron, term)
        return Hm

    @property
    def n_qubits(self):
        return len(self.qubits)

    @property
    def paulistrings(self):
        """
        :return: the Hamiltonian as list of PauliStrings
        """
        return [PauliString.from_openfermion(key=k, coeff=v) for k, v in self.items()]

    @paulistrings.setter
    def paulistrings(self, other):
        """
        Reassign with Tequila PauliString format
        :param other: list of PauliStrings
        :return: self for chaining
        """
        new_hamiltonian = QubitOperator.identity()
        for ps in other:
            tmp = QubitOperator(term=ps.key_openfermion(), value=ps.coeff)
            new_hamiltonian += tmp
        self._qubit_operator = new_hamiltonian
        return self

    def map_qubits(self, qubit_map: dict):
        """

        E.G.  X(1)Y(2) --> X(3)Y(1) with qubit_map = {1:3, 2:1}

        Parameters
        ----------
        qubit_map
            a dictionary which maps old to new qubits

        Returns
        -------
        the Hamiltonian with mapped qubits

        """

        mapped_terms = {}

        for k, v in self.qubit_operator.terms.items():
            mk = tuple([(qubit_map[x[0]], x[1]) for x in k])
            mapped_terms[mk] = v

        mapped = QubitOperator.zero()
        mapped.terms = mapped_terms
        return QubitHamiltonian(qubit_operator=mapped)

    def is_all_z(self):
        """
        Returns
        -------
            returns True if all non-unit paulis in the hamiltonian are Z
        """
        for p in self.paulistrings:
            if not p.is_all_z():
                return False
        return True
