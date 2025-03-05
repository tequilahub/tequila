from __future__ import annotations

import typing
from copy import deepcopy
from math import log2
from typing import Union, Generator

import numpy
import numpy as np
import numpy.typing as npt
import numbers

import sympy

from tequila.utils.bitstrings import BitString, reverse_int_bits
from tequila import TequilaException, BitNumbering, initialize_bitstring
from tequila.utils.keymap import KeyMapABC

if typing.TYPE_CHECKING:
    # Don't need those structures, just for convenient type hinting
    from tequila.hamiltonian.qubit_hamiltonian import QubitHamiltonian, PauliString


class QubitWaveFunction:
    """
    Represents a wavefunction.
    Amplitudes are either stored in a Numpy array for dense wavefunctions, or a dictionary for sparse wavefunctions.
    Does not enforce normalization.
    """

    def __init__(self, n_qubits: int, numbering: BitNumbering = BitNumbering.MSB, dense: bool = False,
                 init_state: bool = True) -> None:
        """
        Initialize a QubitWaveFunction with all amplitudes set to zero.

        :param n_qubits: Number of qubits.
        :param numbering: Whether the first qubit is the most or least significant.
        :param dense: Whether to store the amplitudes in a Numpy array instead of a dictionary.
        :param init_state: Whether to initialize the state array.
            If False, set_state must be called immediately after the constructor.
        """
        self._n_qubits: int = n_qubits
        self._numbering = numbering
        self._dense = dense
        if init_state:
            self._state = np.zeros(2 ** self._n_qubits, dtype=complex) if dense else dict()
        else:
            self._state = None

    @classmethod
    def from_wavefunction(cls, wfn: QubitWaveFunction, keymap: KeyMapABC = None, n_qubits: int = None,
                          initial_state: BitString = None) -> QubitWaveFunction:
        """
        Create a copy of a wavefunction.

        :param wfn: The wavefunction to copy.
        :param keymap: A keymap to apply to the wavefunction.
        :param n_qubits: Number of qubits of the new wavefunction.
            Must not be None if keymap is not None.
        :param initial_state: Initial state to pass to the keymap.
        :return: The copied wavefunction.
        """
        if keymap is not None:
            result = QubitWaveFunction(n_qubits, numbering=wfn._numbering, dense=wfn._dense)
            # Change amplitudes to sympy objects
            if wfn._dense and wfn._state.dtype == object:
                result = sympy.Integer(1) * result
            for index, coeff in wfn.raw_items():
                key = initialize_bitstring(index, wfn._n_qubits, numbering_in=wfn._numbering, numbering_out=keymap.numbering)
                key = keymap(key, initial_state)
                result[key] += coeff
            return result
        else:
            return deepcopy(wfn)

    @classmethod
    def from_array(cls, array: npt.NDArray[complex], numbering: BitNumbering = BitNumbering.MSB,
                   copy: bool = True) -> QubitWaveFunction:
        """
        Create a dense wavefunction from a Numpy array.

        :param array: Array of amplitudes.
        :param numbering: Whether the first qubit is the most or least significant.
        :param copy: Whether to copy the array or use it directly.
            If False, the array must not be modified after the constructor.
        :return: The created wavefunction.
        """
        if not log2(len(array)).is_integer():
            raise ValueError(f"Array length must be a power of 2, received {len(array)}")
        n_qubits = int(log2(len(array)))
        result = QubitWaveFunction(n_qubits, numbering, dense=True, init_state=False)
        result.set_state(array, copy)
        return result

    @classmethod
    def from_basis_state(cls, n_qubits: int, basis_state: Union[int, BitString],
                         numbering: BitNumbering = BitNumbering.MSB) -> QubitWaveFunction:
        """
        Create a sparse wavefunction that is a basis state.

        :param n_qubits: Number of qubits.
        :param basis_state: Index of the basis state.
        :param numbering: Whether the first qubit is the most or least significant.
        :return: The created wavefunction.
        """
        if 2 ** n_qubits <= basis_state:
            raise ValueError(f"Number of qubits {n_qubits} insufficient for basis state {basis_state}")
        if isinstance(basis_state, BitString):
            basis_state = reverse_int_bits(basis_state.integer,
                                           basis_state.nbits) if numbering != basis_state.numbering else basis_state.integer
        result = QubitWaveFunction(n_qubits, numbering)
        result[basis_state] = 1.0
        return result

    @classmethod
    def from_string(cls, string: str, numbering: BitNumbering = BitNumbering.MSB) -> QubitWaveFunction:
        """
        Create a sparse wavefunction from a string.

        :param string: String representation of the wavefunction.
        :param numbering: Whether the first qubit is the most or least significant.
        :return: The created wavefunction.
        """
        try:
            string = string.replace(" ", "")
            string = string.replace("*", "")
            terms = string.split(">")[:-1]
            n_qubits = len(terms[0].split("|")[-1])
            result = QubitWaveFunction(n_qubits, numbering)
            for term in terms:
                coeff, index = term.split("|")
                coeff = complex(coeff) if coeff != "" else 1.0
                index = int(index, 2)
                result[index] = coeff
            return result
        except ValueError:
            raise TequilaException(f"Failed to initialize QubitWaveFunction from string:\n\"{string}\"\n")

    @classmethod
    def convert_from(cls, n_qubits: int, val: Union[QubitWaveFunction, int, str, numpy.ndarray]):
        """
        Convert a value to a QubitWaveFunction.
        Accepts QubitWaveFunction, int, str, and numpy.ndarray.

        :param n_qubits: Number of qubits.
        :param val: Value to convert.
        :return: The converted value.
        """
        if isinstance(val, QubitWaveFunction):
            return val
        elif isinstance(val, int):
            return cls.from_basis_state(n_qubits=n_qubits, basis_state=val)
        elif isinstance(val, str):
            return cls.from_string(val)
        elif isinstance(val, numpy.ndarray):
            return cls.from_array(val)
        else:
            raise TequilaException(f"Cannot initialize QubitWaveFunction from type {type(val)}")

    @property
    def n_qubits(self) -> int:
        """
        Returns number of qubits in the wavefunction.
        """
        return self._n_qubits

    @property
    def numbering(self) -> BitNumbering:
        """
        Returns the bit numbering of the wavefunction.
        """
        return self._numbering

    @property
    def dense(self) -> bool:
        """
        Returns whether the wavefunction is dense.
        """
        return self._dense

    def to_array(self, out_numbering: BitNumbering = BitNumbering.MSB, copy: bool = True) -> npt.NDArray[complex]:
        """
        Returns array of amplitudes.

        :param out_numbering: Whether the first qubit is the most or least significant in the output array indices.
            For dense wavefunctions, this operation is significantly cheaper when this is the same as the numbering
            of the wavefunction.
        :param copy: Whether to copy the array or use it directly for dense Wavefunctions.
            If False, changes to the array or wavefunction will affect each other.
        :return: Array of amplitudes.
        """
        if self._dense and self._numbering == out_numbering:
            return self._state.copy() if copy else self._state
        else:
            result = np.zeros(2 ** self._n_qubits, dtype=complex)
            for k, v in self.raw_items():
                if self._numbering != out_numbering:
                    k = reverse_int_bits(k, self._n_qubits)
                result[k] = v
            return result

    def set_state(self, value: npt.NDArray[complex], copy: bool = True) -> None:
        """
        Sets the state to an array.
        After this call, the wavefunction will be dense.

        :param value: Array of amplitudes. Length must be 2 ** n_qubits.
        :param copy: Whether to copy the array or use it directly.
            If False, changes to the array or wavefunction will affect each other.
        """
        if len(value) != 2 ** self._n_qubits:
            raise ValueError(f"Wavefunction of {self._n_qubits} qubits must have {2 ** self._n_qubits} amplitudes, "
                             f"received {len(value)}")
        self._dense = True
        if copy:
            self._state = value.copy()
        else:
            self._state = value

    def __getitem__(self, key: Union[int, BitString]) -> complex:
        if isinstance(key, BitString):
            key = reverse_int_bits(key.integer, key.nbits) if self._numbering != key.numbering else key.integer
        return self._state[key] if self._dense else self._state.get(key, 0)

    def __setitem__(self, key: Union[int, BitString], value: complex) -> None:
        if isinstance(key, BitString):
            key = reverse_int_bits(key.integer, key.nbits) if self._numbering != key.numbering else key.integer
        self._state[key] = value

    def __contains__(self, item: Union[int, BitString]) -> bool:
        if isinstance(item, BitString):
            item = reverse_int_bits(item.integer, item.nbits) if self._numbering != item.numbering else item.integer
        return abs(self[item]) > 1e-6

    def raw_items(self) -> Generator[tuple[int, complex]]:
        """Returns a generator of non-zero amplitudes with integer indices."""
        return ((k, v) for k, v in (enumerate(self._state) if self._dense else self._state.items()))

    def items(self) -> Generator[tuple[BitString, complex]]:
        """Returns a generator of non-zero amplitudes with BitString indices."""
        return ((initialize_bitstring(k, self._n_qubits, self._numbering), v)
                for k, v in self.raw_items()
                if isinstance(v, sympy.Basic) or abs(v) > 1e-6)

    def keys(self) -> Generator[BitString]:
        """Returns a generator of BitString indices of non-zero amplitudes."""
        return (k for k, v in self.items())

    def values(self) -> Generator[complex]:
        """Returns a generator of non-zero amplitudes."""
        return (v for k, v in self.items())

    def __eq__(self, other) -> bool:
        if not isinstance(other, QubitWaveFunction):
            return False

        raise TequilaException("Wavefunction equality is not well-defined. Consider using isclose.")

    def isclose(self: QubitWaveFunction,
                other: QubitWaveFunction,
                rtol: float = 1e-5,
                atol: float = 1e-8) -> bool:
        """
        Check if two wavefunctions are close, up to a global phase.

        :param other: The other wavefunction.
        :param rtol: Relative tolerance.
        :param atol: Absolute tolerance.
        :return: Whether the wavefunctions are close.
        """
        inner = self.inner(other)
        self_norm = self.norm()
        other_norm = other.norm()
        cosine_similarity = inner / (self_norm * other_norm)

        return (np.isclose(abs(cosine_similarity), 1.0, rtol, atol)
                and np.isclose(self_norm, other_norm, rtol, atol))

    def __add__(self, other: QubitWaveFunction) -> QubitWaveFunction:
        if self._dense and other._dense and self._numbering == other._numbering:
            return QubitWaveFunction.from_array(self._state + other._state, self.numbering, copy=False)
        else:
            result = QubitWaveFunction.from_wavefunction(self)
            result += other
            return result

    def __iadd__(self, other: QubitWaveFunction) -> QubitWaveFunction:
        if self._dense and other._dense and self._numbering == other._numbering:
            self._state += other._state
        else:
            for k, v in other.raw_items():
                if self._numbering != other._numbering:
                    k = reverse_int_bits(k, self._n_qubits)
                self[k] += v
        return self

    def __sub__(self, other: QubitWaveFunction) -> QubitWaveFunction:
        if self._dense and other._dense and self._numbering == other._numbering:
            return QubitWaveFunction.from_array(self._state - other._state, self.numbering, copy=False)
        else:
            result = QubitWaveFunction.from_wavefunction(self)
            result -= other
            return result

    def __isub__(self, other: QubitWaveFunction) -> QubitWaveFunction:
        if self._dense and other._dense and self._numbering == other._numbering:
            self._state -= other._state
        else:
            for k, v in other.raw_items():
                if self._numbering != other._numbering:
                    k = reverse_int_bits(k, self._n_qubits)
                self[k] -= v
        return self

    def __rmul__(self, other: complex) -> QubitWaveFunction:
        if self._dense:
            return QubitWaveFunction.from_array(other * self._state, self.numbering, copy=False)
        else:
            result = QubitWaveFunction.from_wavefunction(self)
            result *= other
            return result

    def __imul__(self, other: complex) -> QubitWaveFunction:
        if self._dense:
            self._state *= other
        else:
            for k, v in self.raw_items():
                self[k] = other * v
        return self

    def inner(self, other: QubitWaveFunction) -> complex:
        """Returns the inner product with another wavefunction."""
        if self._dense and other._dense and self._numbering == other._numbering:
            return np.inner(self._state.conjugate(), other._state)
        else:
            result = 0
            for k, v in self.raw_items():
                if self._numbering != other._numbering:
                    k = reverse_int_bits(k, self._n_qubits)
                result += v.conjugate() * other[k]
            if isinstance(result, sympy.Basic):
                result = complex(result)
            return result

    def norm(self) -> float:
        """Returns the norm of the wavefunction."""
        return np.sqrt(self.inner(self))

    def normalize(self, inplace: bool = False) -> QubitWaveFunction:
        """
        Normalizes the wavefunction.

        :param inplace: Whether to normalize the wavefunction in place or return a new one.
        :return: The normalized wavefunction.
        """
        norm = self.norm()
        if inplace:
            self *= 1.0 / norm
            return self
        else:
            return (1.0 / norm) * QubitWaveFunction.from_wavefunction(self)

    # It would be nice to call this __len__, however for some reason this causes infinite loops
    # when multiplying wave functions with some types of numbers from the right sight, likely
    # because the __mul__ implementation of the number tries to perform some sort of array
    # operation.
    def length(self):
        return sum(1 for (k, v) in self.raw_items() if abs(v) > 1e-6)

    def __repr__(self):
        result = str()
        for index, coeff in self.items():
            index = index.integer
            if self.numbering == BitNumbering.LSB:
                index = reverse_int_bits(index, self._n_qubits)
            if np.isclose(coeff.imag, 0.0):
                result += f"{coeff.real:+2.4f} |{index:0{self._n_qubits}b}> "
            else:
                result += f"({coeff.real:+2.4f} + {coeff.imag:+2.4f}i) |{index:0{self._n_qubits}b}> "
        # If the wavefunction contains no states
        if not result:
            result = "empty wavefunction"
        return result

    def compute_expectationvalue(self, operator: QubitHamiltonian) -> numbers.Real:
        tmp = self.apply_qubitoperator(operator=operator)
        E = self.inner(other=tmp)
        if hasattr(E, "imag") and np.isclose(E.imag, 0.0, atol=1.e-6):
            return float(E.real)
        else:
            return E

    def apply_qubitoperator(self, operator: QubitHamiltonian) -> QubitWaveFunction:
        """
        Inefficient function which computes the action of a QubitHamiltonian on this wfn
        :param operator: QubitOperator
        :return: resulting Qubitwavefunction
        """
        result = QubitWaveFunction(self.n_qubits, self._numbering)
        for ps in operator.paulistrings:
            result += self.apply_paulistring(paulistring=ps)
        return result

    def apply_paulistring(self, paulistring: PauliString) -> QubitWaveFunction:
        """
        Inefficient function which computes action of a single paulistring
        :param paulistring: PauliString
        :return: Expectation Value
        """
        result = QubitWaveFunction(self._n_qubits, self._numbering)
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
