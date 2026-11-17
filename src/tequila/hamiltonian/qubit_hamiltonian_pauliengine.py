import numbers
import typing

import numpy as np

import pauliengine


from pauliengine.qubit_hamiltonian import (
    Expression,
    PauliStringComplex,
    PauliStringSymbolic,
    QubitHamiltonianComplex,
    QubitHamiltonianSymbolic,
    _complex_coeff_to_expression,
)

from openfermion import QubitOperator

from tequila import TequilaException
from tequila.utils import to_float
from tequila.hamiltonian.qubit_hamiltonian import PauliString


def _to_expression(value):
    if isinstance(value, Expression):
        return value
    if isinstance(value, str):
        return Expression(value)
    return _complex_coeff_to_expression(complex(value))


def _terms_to_pe(terms):
    if any(isinstance(v, (Expression, str)) for v in terms.values()):
        ps = [PauliStringSymbolic(_to_expression(v), {q: p.upper() for q, p in k}) for k, v in terms.items()]
        if not ps:
            return QubitHamiltonianComplex.zero()
        return QubitHamiltonianSymbolic(ps)
    ps = [(complex(v), {q: p.upper() for q, p in k}) for k, v in terms.items()]
    if not ps:
        return pauliengine.QubitHamiltonian.zero()
    return pauliengine.QubitHamiltonian(ps)


def _real_if_possible(coeff):
    """
    pauliengine always hands out complex coefficients, while openfermion keeps
    real ones as floats. Downstream consumers rely on that (the spex backend for
    instance feeds coefficients into pybind11 signatures that only accept floats),
    so mirror the openfermion behaviour here.
    """
    if isinstance(coeff, complex) and coeff.imag == 0.0:
        return coeff.real
    return coeff


def _pe_to_terms(pe_hamiltonian):
    terms = {}
    for coeff, ops in pe_hamiltonian.to_dictionary():
        key = tuple(sorted((q, p.upper()) for q, p in ops.items()))
        terms[key] = _real_if_possible(coeff)
    return terms


def _promote_pe(pe_hamiltonian):
    if isinstance(pe_hamiltonian, QubitHamiltonianSymbolic):
        return pe_hamiltonian
    ps = [
        PauliStringSymbolic(_complex_coeff_to_expression(complex(c)), dict(ops))
        for c, ops in pe_hamiltonian.to_dictionary()
    ]
    if not ps:
        return QubitHamiltonianSymbolic.zero()
    return QubitHamiltonianSymbolic(ps)


def _scalar_sym_pe(value):
    return QubitHamiltonianSymbolic([PauliStringSymbolic(_to_expression(value), {})])


class QubitHamiltonian:
    axis_to_string = {0: "x", 1: "y", 2: "z"}
    string_to_axis = {"x": 0, "y": 1, "z": 2}

    def __init__(self, qubit_operator: typing.Union[QubitOperator, str, numbers.Number] = None):
        self._pe = None
        self._terms = None
        if qubit_operator is None:
            self._terms = {}
        elif isinstance(qubit_operator, str):
            other = self.from_string(string=qubit_operator)
            self._pe = other._pe
            self._terms = other._terms
        elif isinstance(qubit_operator, numbers.Number):
            self._terms = {tuple(): qubit_operator}
        elif isinstance(qubit_operator, QubitHamiltonian):
            self._pe = qubit_operator._pe
            self._terms = dict(qubit_operator._terms) if qubit_operator._terms is not None else None
        elif isinstance(qubit_operator, QubitOperator):
            self._terms = dict(qubit_operator.terms)
        elif isinstance(qubit_operator, (PauliStringComplex, PauliStringSymbolic)):
            self._pe = pauliengine.QubitHamiltonian([qubit_operator])
        else:
            self._pe = pauliengine.QubitHamiltonian(qubit_operator)

    @classmethod
    def _from_pe(cls, pe_hamiltonian):
        result = cls.__new__(cls)
        result._pe = pe_hamiltonian
        result._terms = None
        return result

    @classmethod
    def _from_terms(cls, terms):
        result = cls.__new__(cls)
        result._pe = None
        result._terms = terms
        return result

    def _get_pe(self):
        if self._pe is None:
            self._pe = _terms_to_pe(self._terms)
        return self._pe

    def _get_terms(self):
        if self._terms is None:
            self._terms = _pe_to_terms(self._pe)
        return self._terms

    def is_symbolic(self):
        if self._pe is not None and isinstance(self._pe, QubitHamiltonianSymbolic):
            return True
        if self._terms is not None and any(isinstance(v, (Expression, str)) for v in self._terms.values()):
            return True
        return False

    def _needs_symbolic(self, other):
        if self.is_symbolic():
            return True
        if isinstance(other, (str, Expression)):
            return True
        return isinstance(other, QubitHamiltonian) and other.is_symbolic()

    def _symbolic_other(self, other):
        if isinstance(other, (numbers.Number, str, Expression)):
            return _scalar_sym_pe(other)
        return _promote_pe(other._get_pe())

    def subs(self, variables):
        if not self.is_symbolic():
            return self._from_terms(dict(self._get_terms()))
        sub_map = {str(k): complex(v) for k, v in variables.items()}
        result = self._get_pe().subs(sub_map)
        terms = _pe_to_terms(result)
        try:
            terms = {k: _real_if_possible(PauliStringSymbolic.to_complex(v)) for k, v in terms.items()}
        except RuntimeError:
            pass
        return self._from_terms(terms)

    def diff(self, variable):
        if not self.is_symbolic():
            return self.zero()
        return self._from_pe(self._get_pe().diff(str(variable)))

    def __getstate__(self):
        terms = {}
        for k, v in self._get_terms().items():
            if isinstance(v, Expression):
                terms[k] = ("__symbolic__", str(v))
            else:
                terms[k] = v
        return {"terms": terms}

    def __setstate__(self, state):
        terms = {}
        for k, v in state["terms"].items():
            if isinstance(v, tuple) and len(v) == 2 and v[0] == "__symbolic__":
                terms[k] = Expression(v[1])
            else:
                terms[k] = v
        self._terms = terms
        self._pe = None

    @classmethod
    def from_openfermion(cls, qubit_operator: QubitOperator):
        return cls(qubit_operator=qubit_operator)

    def to_openfermion(self) -> QubitOperator:
        if self.is_symbolic():
            raise TequilaException(
                "to_openfermion() is not supported for symbolic Hamiltonians, evaluate with subs(...) first"
            )
        result = QubitOperator()
        result.terms = dict(self._get_terms())
        return result

    @property
    def qubit_operator(self) -> QubitOperator:
        return self.to_openfermion()

    @qubit_operator.setter
    def qubit_operator(self, other: QubitOperator):
        self._terms = dict(other.terms)
        self._pe = None

    @property
    def qubits(self):
        accumulate = []
        for ps in self.paulistrings:
            accumulate += ps.qubits
        return sorted(list(set(accumulate)))

    def index(self, ituple):
        return ituple[0]

    def pauli(selfs, ituple):
        return ituple[1]

    def __call__(self, wfn):
        if hasattr(wfn, "apply_qubitoperator"):
            if self.is_symbolic():
                raise TequilaException(
                    "applying a symbolic Hamiltonian to a wavefunction is not supported, evaluate with subs(...) first"
                )
            return wfn.apply_qubitoperator(self)
        else:
            raise TequilaException("Not sure what to do here with {} and {} ...".format(self, wfn))

    def trace_out_qubits(self, qubits, states: list = None, *args, **kwargs):
        if self.is_symbolic():
            raise TequilaException(
                "trace_out_qubits() is not supported for symbolic Hamiltonians, evaluate with subs(...) first"
            )
        if states is None:
            states = [(1.0, 0.0)] * len(qubits)
        else:
            assert len(states) == len(qubits)
            states = [tuple(s.to_array()) for s in states]

        reduced_ps = [ps.trace_out_qubits(qubits=qubits, states=states) for ps in self.paulistrings]
        return self.from_paulistrings(ps=reduced_ps).simplify(*args, **kwargs)

    def count_measurements(self):
        if self.is_all_z():
            return 1
        else:
            return len(self)

    def __len__(self):
        return len(self._get_terms())

    def __repr__(self):
        result = ""
        for ps in self.paulistrings:
            result += str(ps)
        return result

    def __getitem__(self, item):
        return self._get_terms()[item]

    def __setitem__(self, key, value):
        self._get_terms()[key] = value
        self._pe = None
        return self

    def items(self):
        return self._get_terms().items()

    def keys(self):
        return self._get_terms().keys()

    def values(self):
        return self._get_terms().values()

    @classmethod
    def zero(cls):
        return cls._from_terms({tuple(): 0.0})

    @classmethod
    def unit(cls):
        return cls._from_terms({tuple(): 1.0})

    @classmethod
    def from_string(cls, string, openfermion_format=False):
        if string.strip() == "":
            return cls.zero()
        elif openfermion_format:
            return cls(qubit_operator=QubitOperator(string))
        else:
            H = cls.zero()
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
            terms = string.split("+")
            for term in terms:
                if term.strip() == "":
                    continue

                coeff = term.split(" ")[0]
                if coeff.strip() == "" or coeff[0] in ["X", "Y", "Z"]:
                    coeff = "1.0"
                    ps = term
                else:
                    ps = term.replace(coeff, " ").replace(" ", "")

                try:
                    if "i" in coeff:
                        coeff = coeff.replace("i", "j")
                    if "@" in coeff:
                        coeff = coeff.replace("@", "e-")
                    coeff = complex(coeff)
                except Exception:
                    raise Exception("failed to convert coefficient : {}".format(coeff))

                if coeff.imag == 0.0:
                    coeff = float(coeff.real)

                H += cls.from_paulistrings(ps=PauliString.from_string(string=ps, coeff=coeff))
            return H.simplify()

    @classmethod
    def from_paulistrings(cls, ps: typing.List[PauliString]):
        if isinstance(ps, PauliString):
            return cls.from_paulistrings(ps=[ps])
        try:
            terms = {}
            for x in ps:
                coeff = x.coeff
                complex(coeff)
                key = tuple(sorted((q, p.upper()) for q, p in x.items()))
                terms[key] = terms.get(key, 0.0) + coeff
            terms = {k: _real_if_possible(v) for k, v in terms.items()}
            return cls._from_terms(terms).simplify()
        except (TypeError, ValueError):
            pauli_strings = [
                PauliStringSymbolic(_to_expression(x.coeff), {q: p.upper() for q, p in x.items()}) for x in ps
            ]
            return cls._from_pe(QubitHamiltonianSymbolic(pauli_strings))

    def _merged_terms(self, other, sign):
        terms = dict(self._get_terms())
        if isinstance(other, numbers.Number):
            other_terms = {tuple(): other}
        else:
            other_terms = other._get_terms()
        for k, v in other_terms.items():
            new_value = terms.get(k, 0.0) + sign * v
            if abs(new_value) < 1.0e-12:
                terms.pop(k, None)
            else:
                terms[k] = new_value
        return terms

    def __add__(self, other):
        if self._needs_symbolic(other):
            return self._from_pe(_promote_pe(self._get_pe()) + self._symbolic_other(other))
        return self._from_terms(self._merged_terms(other, 1.0))

    def __sub__(self, other):
        if self._needs_symbolic(other):
            return self._from_pe(_promote_pe(self._get_pe()) - self._symbolic_other(other))
        return self._from_terms(self._merged_terms(other, -1.0))

    def __iadd__(self, other):
        if self._needs_symbolic(other):
            self._pe = _promote_pe(self._get_pe()) + self._symbolic_other(other)
            self._terms = None
        else:
            self._terms = self._merged_terms(other, 1.0)
            self._pe = None
        return self

    def __isub__(self, other):
        if self._needs_symbolic(other):
            self._pe = _promote_pe(self._get_pe()) - self._symbolic_other(other)
            self._terms = None
        else:
            self._terms = self._merged_terms(other, -1.0)
            self._pe = None
        return self

    def __mul__(self, other):
        if hasattr(other, "apply_qubitoperator"):
            if self.is_symbolic():
                raise TequilaException(
                    "applying a symbolic Hamiltonian to a wavefunction is not supported, evaluate with subs(...) first"
                )
            return other.apply_qubitoperator(operator=self)
        if self._needs_symbolic(other):
            if isinstance(other, numbers.Number):
                return self._from_pe(self._get_pe() * complex(other))
            return self._from_pe(_promote_pe(self._get_pe()) * self._symbolic_other(other))
        if isinstance(other, numbers.Number):
            return self._from_terms({k: v * other for k, v in self._get_terms().items()})
        return self._from_pe(self._get_pe() * other._get_pe())

    def __imul__(self, other):
        if self._needs_symbolic(other):
            if isinstance(other, numbers.Number):
                self._pe = self._get_pe() * complex(other)
            else:
                self._pe = _promote_pe(self._get_pe()) * self._symbolic_other(other)
            self._terms = None
        elif isinstance(other, numbers.Number):
            self._terms = {k: v * other for k, v in self._get_terms().items()}
            self._pe = None
        else:
            self._pe = self._get_pe() * other._get_pe()
            self._terms = None
        return self

    def __rmul__(self, other):
        assert isinstance(other, (numbers.Number, str, Expression))
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other=other)

    def __rsub__(self, other):
        return self.__neg__().__add__(other=other)

    def __pow__(self, power):
        return self._from_pe(self._get_pe() ** power)

    def __neg__(self):
        return self.__mul__(other=-1.0)

    def __eq__(self, other):
        if isinstance(other, QubitHamiltonian):
            if self.is_symbolic() or other.is_symbolic():
                return _promote_pe(self._get_pe()) == _promote_pe(other._get_pe())
            return self.to_openfermion() == other.to_openfermion()
        if isinstance(other, QubitOperator):
            return self.to_openfermion() == other
        return NotImplemented

    def is_hermitian(self):
        if self.is_symbolic():
            return bool(self._get_pe().is_hermitian())
        try:
            terms = self._get_terms()
            for k, v in terms.items():
                terms[k] = to_float(v)
            return True
        except TypeError:
            return False

    def simplify(self, threshold=0.0):
        if self.is_symbolic():
            return self
        simplified = {}
        for k, v in self._get_terms().items():
            if not np.isclose(v, 0.0, atol=threshold):
                simplified[k] = v
        self._terms = simplified
        self._pe = None
        return self

    def split(self, *args, **kwargs) -> tuple:
        if self.is_symbolic():
            raise TequilaException("split() is not supported for symbolic Hamiltonians, evaluate with subs(...) first")
        hermitian = {}
        anti_hermitian = {}
        for k, v in self._get_terms().items():
            v = complex(v)
            hermitian[k] = v.real
            anti_hermitian[k] = 1.0j * v.imag

        return self._from_terms(hermitian).simplify(), self._from_terms(anti_hermitian).simplify()

    def is_antihermitian(self):
        if self.is_symbolic():
            return bool(self._get_pe().is_antihermitian())
        for v in self.values():
            if v.real != 0.0:
                return False
        return True

    def _pe_representation_live(self):
        return self._terms is None or self.is_symbolic()

    def conjugate(self):
        if self._pe_representation_live():
            return self._from_pe(self._get_pe().conjugate())
        terms = {}
        for key, value in self._get_terms().items():
            sign = 1
            for term in key:
                if self.pauli(term).lower() == "y":
                    sign *= -1
            terms[key] = sign * value.conjugate()
        return self._from_terms(terms)

    def transpose(self):
        if self._pe_representation_live():
            return self._from_pe(self._get_pe().transpose())
        terms = {}
        for key, value in self._get_terms().items():
            sign = 1
            for term in key:
                if self.pauli(term).lower() == "y":
                    sign *= -1
            terms[key] = sign * value
        return self._from_terms(terms)

    def dagger(self):
        if self._pe_representation_live():
            return self._from_pe(self._get_pe().dagger())
        terms = {}
        for key, value in self._get_terms().items():
            terms[key] = value.conjugate()
        return self._from_terms(terms)

    def normalize(self):
        if self.is_symbolic():
            raise TequilaException(
                "normalize() is not supported for symbolic Hamiltonians, evaluate with subs(...) first"
            )
        norm = np.sqrt(sum(abs(v) ** 2 for v in self._get_terms().values()))
        self._terms = {k: v / norm for k, v in self._get_terms().items()}
        self._pe = None
        return self

    def to_matrix(self, ignore_unused_qubits=True):
        if self.is_symbolic():
            raise TequilaException(
                "to_matrix() is not supported for symbolic Hamiltonians, evaluate with subs(...) first"
            )
        return np.asarray(self._get_pe().to_matrix(ignore_unused_qubits=ignore_unused_qubits), dtype=complex)

    @property
    def n_qubits(self):
        return len(self.qubits)

    @property
    def paulistrings(self):
        return [PauliString.from_openfermion(key=k, coeff=v) for k, v in self.items()]

    @paulistrings.setter
    def paulistrings(self, other):
        self._pe = _terms_to_pe({ps.key_openfermion(): ps.coeff for ps in other})
        self._terms = None
        return self

    def map_qubits(self, qubit_map: dict):
        if self._pe_representation_live():
            return self._from_pe(self._get_pe().map_qubits(qubit_map))
        mapped_terms = {}
        for k, v in self._get_terms().items():
            mk = tuple(sorted((qubit_map[x[0]], x[1]) for x in k))
            mapped_terms[mk] = v
        return self._from_terms(mapped_terms)

    def is_all_z(self):
        for p in self.paulistrings:
            if not p.is_all_z():
                return False
        return True

    def constant(self):
        return self._get_terms().get(tuple(), 0.0)
