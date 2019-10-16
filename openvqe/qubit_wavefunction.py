import copy
import typing

from openvqe import numpy
from openvqe import BitNumbering, BitString, initialize_bitstring, OpenVQEException
from openvqe.hamiltonian import QubitHamiltonian, PauliString
from openvqe.keymap import KeyMapLSB2MSB, KeyMapMSB2LSB
from openvqe.tools import number_to_string


class QubitWaveFunction:
    """
    Store Wavefunction as dictionary of comp. basis state and complex numbers
    Use the same structure for Measurments results with int instead of complex numbers (counts)
    """

    numbering = BitNumbering.MSB

    def apply_keymap(self, keymap, initial_state: BitString = None):
        self.n_qubits = keymap.n_qubits
        mapped_state = dict()
        for k, v in self.state.items():
            mapped_state[keymap(input_state=k, initial_state=initial_state)] = v

        self.state = mapped_state
        return self

    @property
    def n_qubits(self) -> int:
        if self._n_qubits is None:
            return self.min_qubits()
        else:
            return max(self._n_qubits, self.min_qubits())

    def min_qubits(self) -> int:
        if len(self.state) > 0:
            maxk = max(self.state.keys())
            return maxk.nbits
        else:
            return 0

    @n_qubits.setter
    def n_qubits(self, n_qubits):
        if n_qubits is not None:
            self._n_qubits = max(n_qubits, self.min_qubits())
        return self

    @property
    def state(self):
        if self._state is None:
            return dict()
        else:
            return self._state

    @state.setter
    def state(self, other: typing.Dict[BitString, complex]):
        assert (isinstance(other, dict))
        self._state = other

    def __init__(self, state: typing.Dict[BitString, complex] = None):
        if state is None:
            self._state = dict()
        else:
            self._state = state
        self._n_qubits = None

    def items(self):
        return self.state.items()

    def keys(self):
        return self.state.keys()

    def values(self):
        return self.state.values()

    def __getitem__(self, item):
        return self.state[item]

    def __setitem__(self, key, value):
        self._state[key] = value
        return self

    def __len__(self):
        return len(self.state)

    @classmethod
    def from_array(cls, arr: numpy.ndarray, keymap=None, threshold: float = 1.e-6,
                   numbering: BitNumbering = BitNumbering.MSB):
        assert (len(arr.shape) == 1)
        state = dict()
        maxkey = len(arr) - 1
        maxbit = initialize_bitstring(integer=maxkey, numbering_in=numbering, numbering_out=cls.numbering).nbits
        for ii, v in enumerate(arr):
            i = initialize_bitstring(integer=ii, nbits=maxbit, numbering_in=numbering, numbering_out=cls.numbering)
            if not numpy.isclose(abs(v), 0, atol=threshold):
                key = i if keymap is None else keymap(i)
                state[key] = v
        result = QubitWaveFunction(state)

        if cls.numbering != numbering:
            if cls.numbering == BitNumbering.MSB:
                result.apply_keymap(keymap=KeyMapLSB2MSB())
            else:
                result.apply_keymap(keymap=KeyMapMSB2LSB())

        return result

    @classmethod
    def from_int(cls, i: int, coeff=1):
        if isinstance(i, BitString):
            return QubitWaveFunction(state={i: coeff})
        else:
            return QubitWaveFunction(state={BitString.from_int(integer=i): coeff})

    def __repr__(self):
        result = str()
        for k, v in self.items():
            result += number_to_string(number=v) + "|" + str(k.binary) + "> "
        return result

    def __eq__(self, other):
        if len(self.state) != len(other.state):
            return False
        for k, v in self.state.items():
            if k not in other.state:
                return False
            elif not numpy.isclose(complex(v), complex(other.state[k]), atol=1.e-6):
                return False

        return True

    def __add__(self, other):
        result = QubitWaveFunction(state=copy.deepcopy(self._state))
        for k, v in other.items():
            if k in result._state:
                result._state[k] += v
            else:
                result._state[k] = v
        return result

    def __sub__(self, other):
        return self + -1.0*other

    def __iadd__(self, other):
        for k, v in other.items():
            if k in self._state:
                self._state[k] += v
            else:
                self._state[k] = v
        return self

    def __rmul__(self, other):
        result = QubitWaveFunction(state=copy.deepcopy(self._state))
        for k, v in result._state.items():
            result._state[k] *= other
        return result

    def inner(self, other):
        # currently very slow and not optimized in any way
        result = 0.0
        for k, v in self.items():
            if k in other._state:
                result += v.conjugate() * other._state[k]
        return result

    def normalize(self):
        """
        Inplace operation
        :return: Normalizes the wavefunction/countrate
        """
        norm2 = self.inner(other=self)
        self = 1.0/numpy.sqrt(norm2)*self
        return self

    def compute_expectationvalue(self, operator:QubitHamiltonian) -> float:
        tmp = self.apply_qubitoperator(operator=operator)
        return self.inner(other=tmp)

    def apply_qubitoperator(self, operator: QubitHamiltonian):
        """
        Inefficient function which computes the action of a QubitHamiltonian on this wfn
        :param operator: QubitOperator
        :return: resulting Qubitwavefunction
        """
        result = QubitWaveFunction()
        for ps in operator.paulistrings:
            result += self.apply_paulistring(paulistring=ps)
        return result

    def apply_paulistring(self, paulistring: PauliString):
        """
        Inefficient function which computes action of a single paulistring
        :param paulistring: PauliString
        :return: Expectation Value
        """
        result = QubitWaveFunction()
        for k, v in self.items():
            arr = k.array
            c = v
            for idx, p in paulistring.items():
                if p.lower() == "x":
                    arr[idx] = (arr[idx] + 1) % 2
                elif p.lower() == "y":
                    c *= 1.0j*(-1)**(arr[idx])
                    arr[idx] = (arr[idx] + 1) % 2
                elif p.lower() == "z":
                    c *= (-1)**(arr[idx])
                else:
                    raise OpenVQEException("unknown pauli: " + str(p))
            result[BitString.from_array(array=arr)] = c
        return paulistring.coeff*result