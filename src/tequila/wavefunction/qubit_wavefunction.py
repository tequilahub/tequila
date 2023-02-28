from __future__ import annotations
import copy
import typing
from typing import Dict, Union
import numpy
import numbers

from tequila.utils.bitstrings import BitNumbering, BitString, initialize_bitstring
from tequila import TequilaException
from tequila.utils.keymap import KeyMapLSB2MSB, KeyMapMSB2LSB
from tequila.tools import number_to_string

# from __future__ import annotations # can use that in python 3.7+ to get rid of string type hints

if typing.TYPE_CHECKING:
    # don't need those structures, just for convenient type hinting
    from tequila.hamiltonian.qubit_hamiltonian import QubitHamiltonian, PauliString


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
            mapped_key=keymap(input_state=k, initial_state=initial_state)
            if mapped_key in mapped_state:
                mapped_state[mapped_key] += v
            else:
                mapped_state[mapped_key] = v

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
    def state(self, other: Dict[BitString, complex]):
        assert (isinstance(other, dict))
        self._state = other

    def __init__(self, state: Dict[BitString, complex] = None, n_qubits=None):
        if state is None:
            self._state = dict()
        elif isinstance(state, int):
            self._state = self.from_int(i=state, n_qubits=n_qubits).state
        elif isinstance(state, str):
            self._state = self.from_string(string=state, n_qubits=n_qubits).state
        elif isinstance(state, numpy.ndarray) or isinstance(state, list):
            self._state = self.from_array(arr=state, n_qubits=n_qubits).state
        elif hasattr(state, "state"):
            self._state = state.state
        else:
            self._state = state
        self._n_qubits = n_qubits

    def items(self):
        return self.state.items()

    def keys(self):
        return self.state.keys()

    def values(self):
        return self.state.values()

    @staticmethod
    def convert_bitstring(key: Union[BitString, numbers.Integral], n_qubits):
        if isinstance(key, numbers.Integral):
            return BitString.from_int(integer=key, nbits=n_qubits)
        elif isinstance(key, str):
            return BitString.from_binary(binary=key, nbits=n_qubits)
        else:
            return key

    def __getitem__(self, item: BitString):
        key = self.convert_bitstring(item, self.n_qubits)
        return self.state[key]

    def __call__(self, key, *args, **kwargs) -> numbers.Number:
        """
        Like getitem but returns zero if key is not there

        Parameters
        ----------
        key: bitstring (or int or str)
        Returns
        -------
            Return the amplitude or measurement occurence of a bitstring
        """
        ckey = self.convert_bitstring(key, self.n_qubits)
        if ckey in self.state:
            return self.state[ckey]
        else:
            return 0.0



    def __setitem__(self, key: BitString, value: numbers.Number):
        self._state[self.convert_bitstring(key, self.n_qubits)] = value
        return self

    def __contains__(self, item: BitString):
        return self.convert_bitstring(item, self.n_qubits) in self.keys()

    def __len__(self):
        return len(self.state)

    @classmethod
    def from_array(cls, arr: numpy.ndarray, keymap=None, threshold: float = 1.e-6,
                   numbering: BitNumbering = BitNumbering.MSB, n_qubits: int = None):
        arr = numpy.asarray(arr)
        assert (len(arr.shape) == 1)
        state = dict()
        maxkey = len(arr) - 1
        maxbit = initialize_bitstring(integer=maxkey, numbering_in=numbering, numbering_out=cls.numbering).nbits
        for ii, v in enumerate(arr):
            i = initialize_bitstring(integer=ii, nbits=maxbit, numbering_in=numbering, numbering_out=cls.numbering)
            if not numpy.isclose(abs(v), 0.0, atol=threshold):
                key = i if keymap is None else keymap(i)
                state[key] = v
        result = QubitWaveFunction(state, n_qubits=n_qubits)

        if cls.numbering != numbering:
            if cls.numbering == BitNumbering.MSB:
                result.apply_keymap(keymap=KeyMapLSB2MSB())
            else:
                result.apply_keymap(keymap=KeyMapMSB2LSB())

        return result

    @classmethod
    def from_int(cls, i: int, coeff=1, n_qubits: int = None):
        if isinstance(i, BitString):
            return QubitWaveFunction(state={i: coeff}, n_qubits=n_qubits)
        else:
            return QubitWaveFunction(state={BitString.from_int(integer=i, nbits=n_qubits): coeff}, n_qubits=n_qubits)

    @classmethod
    def from_string(cls, string: str, n_qubits: int = None):
        """
        Complex values like (x+iy)|...> will currently not work, you need to type Real and imaginary separately
        Or improve this constructor :-)
        e.g instead of (0.5+1.0j)|0101> do 0.5|0101> + 1.0j|0101>
        :param paths:
        :param string:
        :return:
        """
        try:
            state = dict()
            string = string.replace(" ", "")
            string = string.replace("*", "")
            string = string.replace("+-", "-")
            string = string.replace("-+", "-")
            terms = (string + "terminate").split('>')
            for term in terms:
                if term == 'terminate':
                    break
                tmp = term.split("|")
                coeff = tmp[0]
                if coeff == '':
                    coeff = 1.0
                else:
                    coeff = complex(coeff)
                basis_state = BitString.from_binary(binary=tmp[1])

                state[basis_state] = coeff
        except ValueError:
            raise TequilaException("Failed to initialize QubitWaveFunction from string:" + string + "\n"
                                                                                                    "did you try complex values?\n"
                                                                                                    "currently you need to type real and imaginary parts separately\n"
                                                                                                    "e.g. instead of (0.5+1.0j)|0101> do 0.5|0101> + 1.0j|0101>")
        except:
            raise TequilaException("Failed to initialize QubitWaveFunction from string:" + string)
        return QubitWaveFunction(state=state, n_qubits=n_qubits)

    def __repr__(self):
        result = str()
        for k, v in self.items():
            result += number_to_string(number=v) + "|" + str(k.binary) + "> "
        return result

    def __eq__(self, other):
        raise TequilaException("Wavefunction equality is not well-defined. Consider using inner"
                               + " product equality, wf1.isclose(wf2).")

    def isclose(self :  'QubitWaveFunction',
                other : 'QubitWaveFunction',
                rtol : float=1e-5,
                atol : float=1e-8) -> bool:
        """Return whether this wavefunction is similar to the target wavefunction."""
        over1 = complex(self.inner(other))
        over2 = numpy.sqrt(complex(self.inner(self) * other.inner(other)))
        # Explicit casts to complex() is required if self or other are sympy
        # wavefunction with sympy-typed amplitudes

        # Check if the two numbers are equal.
        return numpy.isclose(over1, over2, rtol=rtol, atol=atol)

    def __add__(self, other):
        result = QubitWaveFunction(state=copy.deepcopy(self._state))
        for k, v in other.items():
            if k in result._state:
                result._state[k] += v
            else:
                result._state[k] = v
        return result

    def __sub__(self, other):
        return self + -1.0 * other

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
        NOT AN Inplace operation
        :return: Normalizes the wavefunction/countrate
        """
        norm2 = self.inner(other=self)
        normalized = 1.0 / numpy.sqrt(norm2) * self
        return normalized

    def compute_expectationvalue(self, operator: 'QubitHamiltonian') -> numbers.Real:
        tmp = self.apply_qubitoperator(operator=operator)
        E = self.inner(other=tmp)
        if hasattr(E, "imag") and numpy.isclose(E.imag, 0.0, atol=1.e-6):
            return float(E.real)
        else:
            return E

    def apply_qubitoperator(self, operator: 'QubitHamiltonian'):
        """
        Inefficient function which computes the action of a QubitHamiltonian on this wfn
        :param operator: QubitOperator
        :return: resulting Qubitwavefunction
        """
        result = QubitWaveFunction()
        for ps in operator.paulistrings:
            result += self.apply_paulistring(paulistring=ps)
        result = result.simplify()
        return result

    def apply_paulistring(self, paulistring: 'PauliString'):
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
                    c *= 1.0j * (-1) ** (arr[idx])
                    arr[idx] = (arr[idx] + 1) % 2
                elif p.lower() == "z":
                    c *= (-1) ** (arr[idx])
                else:
                    raise TequilaException("unknown pauli: " + str(p))
            result[BitString.from_array(array=arr)] = c
        return paulistring.coeff * result

    def to_array(self):
        result = numpy.zeros(shape=2 ** self.n_qubits, dtype=complex)
        for k, v in self.items():
            result[int(k)] = v
        return result

    def simplify(self, threshold = 1.e-8):
        state = {}
        for k, v in self.state.items():
            if not numpy.isclose(v, 0.0, atol=threshold):
                state[k] = v
        return QubitWaveFunction(state=state)

