import typing
import copy
import numbers
from tequila.grouping.compile_groups import compile_commuting_parts
from tequila import TequilaException
from tequila.utils import JoinedTransformation
from tequila.hamiltonian import paulis
import numpy as onp
from tequila.autograd_imports import numpy as numpy
from tequila.objective.quantum_arg import QuantumArg, is_quantum_arg
import collections

def make_overlap(U0 = None, U1 = None, *args, **kwargs):
    """
    Function that calculates the overlap between two quantum states.

    Parameters
    ----------
    U0 : QCircuit tequila object, corresponding to the first state (will be the bra).

    U1 : QCircuit tequila object, corresponding to the second state (will be the ket).

    Returns
    -------
    Real and imaginary Tequila objectives to be simulated or compiled.

    """
    from tequila.circuit.circuit import find_unused_qubit
    from tequila.circuit.gates import H, X
    from tequila.hamiltonian import paulis

    ctrl = find_unused_qubit(U0=U0, U1=U1)

    # print('Control qubit:',ctrl)

    U_a = U0.add_controls([ctrl])  # this add the control by modifying the previous circuit
    U_b = U1.add_controls([ctrl])  # NonType object

    # bulding the circuit for the overlap evaluation
    circuit = H(target=ctrl)
    circuit += X(target=ctrl)
    circuit += U_a
    circuit += X(target=ctrl)
    circuit += U_b

    x = paulis.X(ctrl)
    y = paulis.Y(ctrl)
    Ex = ExpectationValue(H=x, U=circuit, *args, **kwargs)
    Ey = ExpectationValue(H=y, U=circuit, *args, **kwargs)

    return Ex, Ey


def make_transition(
    U0 = None, U1 = None, H = None, *args, **kwargs
):
    """
    Function that calculates the transition elements of an Hamiltonian operator
    between two different quantum states.

    Parameters
    ----------
    U0 : QCircuit tequila object, corresponding to the first state (will be the bra).

    U1 : QCircuit tequila object, corresponding to the second state (will be the ket).

    H : QubitHamiltonian tequila object

    Returns
    -------
    Real and imaginary Tequila objectives to be simulated or compiled.

    """
    from tequila.circuit.gates import PauliGate

    # want to measure: <U1|H|U0> -> \sum_k c_k <U1|U_k|U0>

    trans_real = 0
    trans_im = 0

    for ps in H.paulistrings:
        c_k = ps.coeff
        U_k = PauliGate(ps)
        objective_real, objective_im = make_overlap(U0=U0, U1=U1 + U_k, *args, **kwargs)

        trans_real += c_k.real * objective_real - c_k.imag * objective_im
        trans_im += c_k.real * objective_im + c_k.imag * objective_real

    return trans_real, trans_im

class BraKetImpl(QuantumArg):
    def __init__(
        self,
        ket,
        bra=None,
        operator=None,
        contraction: typing.Callable = None,
        shape: tuple = None,
        samples: int = None,
        *args,
        **kwargs
    ):
        if ket is None:
            raise TequilaException("BraKet requires a ket circuit")

        self.ket = ket
        self.bra = ket if bra is None else bra
        self.operator = operator
        self._contraction = contraction
        self._shape = shape
        self.samples = samples
        self._args = args
        self._kwargs = kwargs

    @property
    def U(self):
        return self.ket

    @property
    def H(self) -> tuple:
        if self.operator is None:
            return tuple()
        return (self.operator,)

    @property
    def is_overlap(self) -> bool:
        return self.operator is None

    @property
    def is_self_overlap(self) -> bool:
        return self.is_overlap and self.bra is self.ket

    @property
    def qubits(self):
        q = set(self.bra.qubits) | set(self.ket.qubits)
        if self.operator is not None:
            for ps in self.operator.paulistrings:
                q |= set(ps.qubits)
        return sorted(q)

    @property
    def is_compiled(self):
        return False

    def extract_variables(self):
        var = list(self.ket.extract_variables())
        if self.bra is not self.ket:
            var.extend(self.bra.extract_variables())

        seen, unique = set(), []
        for v in var:
            if v not in seen:
                seen.add(v)
                unique.append(v)
        return unique

    def map_variables(self, variables: dict, *args, **kwargs) -> "BraKetImpl":
        return BraKetImpl(
            ket=self.ket.map_variables(variables, *args, **kwargs),
            bra=self.bra.map_variables(variables, *args, **kwargs),
            operator=self.operator,
            contraction=self._contraction,
            shape=self._shape,
            samples=self.samples,
            *self._args,
            **self._kwargs,
        )

    def map_qubits(self, qubit_map: dict[int, int]) -> "BraKetImpl":
        return BraKetImpl(
            ket=self.ket.map_qubits(qubit_map),
            bra=self.bra.map_qubits(qubit_map),
            operator=self.operator.map_qubits(qubit_map) if self.operator else None,
            contraction=self._contraction,
            shape=self._shape,
            samples=self.samples,
            *self._args,
            **self._kwargs,
        )

    def compile(self):
        kwargs = dict(self._kwargs)

        if self.samples is not None:
            kwargs["samples"] = self.samples

        if self.bra is self.ket and self.operator is None:
            return Objective() + 1.0, Objective()

        if self.bra is self.ket and self.operator is not None:
            return (
                ExpectationValue(H=self.operator, U=self.ket, *self._args, **kwargs),
                Objective()
            )

        if self.operator is None:
            return make_overlap(
                U0=self.bra,
                U1=self.ket,
                *self._args,
                **kwargs
            )
        
        return make_transition(
            U0=self.bra,
            U1=self.ket,
            H=self.operator,
            *self._args,
            **kwargs
        )

    def count_measurements(self) -> int:
        if self.operator is None:
            return 2
        return sum(
            ps.count_measurements() if hasattr(ps, "count_measurements") else 1
            for ps in self.operator.paulistrings
        )

    def __call__(self, *args, **kwargs):
        raise TequilaException(
            "BraKetImpl is symbolic and should not be evaluated directly. "
            "Use tq.simulate(BraKetImpl(...)) for backend execution or BraKet(...)[0/1] for symbolic objectives."
        )

    def __repr__(self) -> str:
        if self.operator:
            return f"BraKetImpl(<bra|H|ket>, qubits={self.qubits})"
        return f"BraKetImpl(<bra|ket>, qubits={self.qubits})"

    def __str__(self) -> str:
        return self.__repr__()
    
def Fidelity(bra, ket, *args, **kwargs):
    """

    Convenience initialization of an tq.Objective that corresponds to the fidelity |<bra|ket>|^2 between two quantum states
    initialized by the circuits bra and ket

    Notes
    -----
    Initializes an ancilla free implementation using:
    |<A|B>|^2 = <A|B><B|A><0|VAUB|0><0|VBUA|0> = <P0>_{VBAU}
    VA = U^\dagger_A
    P0 = |0..0><0..0|
    see: https://arxiv.org/abs/2006.03075 Eq.3-4
    or https://arxiv.org/abs/2011.05938 Eq.57

    note, that the fidelity is symmetric: F(a,b) = F(b,a)
    so it does not matter what is bra and what is ket

    Parameters:
    ----------
    bra: QCircuit
    ket: QCircuit

    Returns:
    ----------
    A tq.Objective that evaluated to the fidelity between the two states

    """
    from tequila.objective.objective import ExpectationValue


    U = bra + ket.dagger()
    qubits = U.qubits
    P0 = paulis.Qp(qubits)
    return ExpectationValue(H=P0, U=U, *args, **kwargs)


def Overlap(bra, ket, *args, **kwargs):
    """

    Convenience initialization of an tq.Objective that corresponds to the overlap <bra|ket>
    initialized by the circuits bra and ket

    Notes
    -----
    the fidelity between two state |<bra|ket>|^2 can be computed more efficient with the function tq.Fidelity

    Parameters:
    ----------
    bra: QCircuit
    ket: QCircuit

    Returns:
    ----------
    A tuple of tq.Objective that evaluates to the real and imaginary part of the overlap between the two states

    """

    return BraKet(ket=ket, bra=bra, operator=None, *args, **kwargs)


def BraKet(ket, bra = None, operator = None, *args, **kwargs):
    from tequila.objective.objective import Objective

    if "H" in kwargs:
        if operator is None:
            operator = kwargs["H"]
        else:
            raise TequilaException(
                'BraKet inconsistency between operator and kwargs["H"] do not given H= ... and operator= ... in the same call'
            )
        kwargs.pop("H")

    if bra is None:
        bra = ket

    return Objective.BraKet(bra=bra, ket=ket, operator=operator, *args, **kwargs)

def RealBraKet(ket, bra=None, operator=None, *args, **kwargs):
    bk = BraKet(ket=ket, bra=bra, operator=operator, *args, **kwargs)
    return Objective(args=[bk], transformation=lambda z: z.real)


def ImagBraKet(ket, bra=None, operator=None, *args, **kwargs):
    bk = BraKet(ket=ket, bra=bra, operator=operator, *args, **kwargs)
    return Objective(args=[bk], transformation=lambda z: z.imag)

class ExpectationValueImpl(QuantumArg):
    """
    Implements the (uncompiled) Expectation Value as a class. Should not be called directly.

    Notes
    -----
    Though not obscured from the user, users should not initialize this class directly, since it lacks convenience
    functions for arithmetics. Instead, these are handled by the Objective class; initializing an Objective
    of a single Expectation Value is done with Objective.ExpectationValue or with tq.ExpectationValue.

    See Also
    --------
    Objective
    ExpectationValue

    Attributes
    ----------
    U:
        a QCircuit, for preparing a state
    H:
        a Hamiltonian, whose expectation value with the state prepared by U is to be determined.

    Methods
    -------
    extract_variables:
        wrapper over extract_variables for QCircuit.
    update_variables:
        wrapper over update_variables for QCircuit.
    info:
        return information about the ExpectationValue.
    """

    @property
    def U(self):
        if self._unitary is None:
            return None
        else:
            return self._unitary

    @property
    def H(self) -> list:
        if self._hamiltonian is None:
            return paulis.QubitHamiltonian.unit()
        else:
            return self._hamiltonian
        
    @property
    def is_compiled(self):
        return self._compiled_cache is not None

    def count_measurements(self):
        return sum([H.count_measurements() for H in self.H])

    def extract_variables(self) -> typing.Dict[str, numbers.Real]:
        """
        Wrapper over identically named function in QCircuit. Returns all variables from the underlying unitary.
        Returns
        -------

        """
        result = []
        if self.U is not None:
            result = self.U.extract_variables()
        return result

    def replace_variables(self, replacement):
        if self.U is not None:
            self.U.replace_variables(replacement)

    def __init__(self, U=None, H=None, contraction=None, shape=None, samples=None, *args, **kwargs):
        """

        Parameters
        ----------
        U: QCircuit, Optional:
            the unitary for state preparation in the expectation value.
        H: optional:
            the hamiltonian
        contraction:
            callable that should transform the output of calling the expectation value.
        shape: optional:
            the shape of the return value for the object
        """
        self._unitary = copy.deepcopy(U)
        if hasattr(H, "paulistrings"):
            self._hamiltonian = tuple([copy.deepcopy(H)])
        else:
            self._hamiltonian = tuple(H)
        self._contraction = contraction
        self._shape = shape
        self.samples = samples

        self._compiled_cache: tuple | None = None

    def compile(self) -> tuple["Objective", None]:
        if self._compiled_cache is not None:
            return self._compiled_cache
        
        real_obj = Objective(args=[self])
        self._compiled_cache = (real_obj, None)
        return self._compiled_cache
    
    def set_compiled(self, compiled_obj: "Objective"):
        self._compiled_cache = (compiled_obj, None)


    def map_qubits(self, qubit_map: dict):
        """

        Maps the qubit within the underlying Hamiltonians and Unitaries

        Parameters
        ----------
        qubit_map
            a dictionary which maps old to new qubits

        Returns
        -------
        the ExpectationValueImpl structure with mapped qubits

        """
        return ExpectationValueImpl(
            H=tuple([H.map_qubits(qubit_map=qubit_map) for H in self.H]),
            U=self.U.map_qubits(qubit_map=qubit_map),
            contraction=self._contraction,
            shape=self._shape,
        )

    def map_variables(self, variables: dict, *args, **kwargs):
        """

        Parameters
        ----------
        variables
            dictionary with old variable names as keys and new variable names or values as values
        Returns
        -------
        Circuit with changed variables

        """
        return ExpectationValueImpl(
            H=self.H,
            U=self.U.map_variables(variables=variables, *args, **kwargs),
            contraction=self._contraction,
            shape=self._shape,
        )

    def __call__(self, *args, **kwargs):
        raise TequilaException(
            "Tried to call uncompiled ExpectationValueImpl, compile your objective before calling with tq.compile(objective) or evaluate with tq.simulate(objective)"
        )

    def info(self, short=True, *args, **kwargs):
        if short:
            print(
                "Expectation Value with {qubits} active qubits and {paulis} paulistrings".format(
                    qubits=len(self.U.qubits), paulis=len(self.H)
                )
            )
        else:
            print("Hamiltonian:\n", str(self.H))
            print("\n", str(self.U))


def identity(x):
    """
    Returns input unchanged.

    Use: in place of lambda x: x, this function is used for smarter auto-diff in grad.
    Parameters
    ----------
    x:
        anything.

    Returns
    -------
    object
        input, returned unchanged.
    """
    return x


class Objective:
    """
    the class which represents mathematical manipulation of ExpectationValue and Variable objects. The core of tequila.

    Todo: Jakob, wanna write some nice examples here?

    Attributes:

    backend: str:
        a string; the backend to which the objective has been compiled, if any. If no expectationvalues, returns 'free'.

    """

    def __init__(self, args: typing.Iterable = None, transformation: typing.Callable = None):
        if args is None:
            self._args = tuple()
            self._transformation = lambda *x: 0.0
        else:
            self._args = tuple(args)
            self._transformation = transformation

    def map_qubits(self, qubit_map: dict):
        """

        Maps qubits for all quantum circuits and hamiltonians in the objective

        Parameters
        ----------
        qubit_map
            a dictionary which maps old to new qubits
            keys and values should be integers

        Returns
        -------
        the Objective with mapped qubits

        """
        mapped_args = []
        for arg in self.args:
            if hasattr(arg, "map_qubits"):
                mapped_args.append(arg.map_qubits(qubit_map=qubit_map))
            else:
                assert not is_quantum_arg(arg)  # failsave
                mapped_args.append(arg)  # for purely variable dependend arguments

        return Objective(args=mapped_args, transformation=self.transformation)

    def map_variables(self, variables, *args, **kwargs):
        """

        Parameters
        ----------
        variables
            dictionary with old variable names as keys and new variable names or values as values
        Returns
        -------
        Circuit with changed variables

        """

        variables = {assign_variable(k): assign_variable(v) for k, v in variables.items()}

        mapped_args = []
        for arg in self.args:
            if hasattr(arg, "map_variables"):
                mapped_args.append(arg.map_variables(variables=variables))
            else:
                mapped_args.append(arg)

        return Objective(args=mapped_args, transformation=self.transformation)

    @property
    def backend(self) -> str:
        """
        Checks if the objective is compiled and gives back the name of the backend if so
        Otherwise returns None
        If the objective has no expectationvalues it gives back 'free'
        """
        if self.has_expectationvalues():
            for arg in self.args:
                if hasattr(arg, "U"):
                    return str(type(arg))
        
        return "free"

    def extract_variables(self) -> list:
        """
        Extract all variables on which the objective depends
        :return: List of all Variables
        """
        variables = []
        for arg in self.args:
            if hasattr(arg, "extract_variables"):
                variables += arg.extract_variables()
            else:
                variables += []
        # remove duplicates without affecting ordering
        # allows better reproducibility for random initialization
        # won't work with set
        unique = []
        for i in variables:
            if i not in unique:
                unique.append(i)
        return unique

    def is_expectationvalue(self):
        """
        :return: bool: whether or not this objective is just a wrapped ExpectationValue
        """
        return len(self.args) == 1 and self._transformation is None and is_quantum_arg(self.args[0])

    def has_expectationvalues(self):
        """
        :return: bool: wether or not this objective has expectationvalues or is just a function of the variables
        """
        # testing if all arguments are only variables and give back the negative
        return any(is_quantum_arg(arg) for arg in self.args)
    
    def get_expectationvalues(self):
        return [arg for arg in self.args if is_quantum_arg(arg)]

    @classmethod
    def ExpectationValue(cls, U=None, H=None, samples=None, *args, **kwargs):
        """
        Initialize a wrapped expectationvalue directly as Objective
        """
        E = ExpectationValueImpl(H=H, U=U, samples=samples, *args, **kwargs)
        return Objective(args=[E])

    @classmethod
    def BraKet(cls, bra=None, ket=None, operator=None, samples=None, *args, **kwargs):
        BK = BraKetImpl(ket=ket, bra=bra, operator=operator, samples=samples, *args, **kwargs)
        return Objective(args=[BK])

    @property
    def transformation(self) -> typing.Callable:
        if self._transformation is None:
            return lambda x: x
        return self._transformation

    @property
    def transformations(self) -> typing.Callable:
        return [self.transformation]

    @property
    def args(self) -> typing.Tuple:
        if self._args is None:
            return tuple()
        return self._args

    @property
    def argsets(self):
        return [self.args]

    def _left_helper(self, op, other):
        """
        function for use by magic methods, which all have an identical structure, differing only by the
        external operator they call.
        left helper is responsible for all 'self # other' operations; right helper, for "other # self".

        Parameters
        ----------
        op: callable:
            the operation to be performed
        other:
            the right-hand argument of the operation to be performed

        Returns
        -------
        Objective:
            an Objective, who transform is the joined_transform of self with op, acting on self and other.
        """
        if isinstance(other, numbers.Number):
            t = lambda v: op(v, other)
            new = self.unary_operator(left=self, op=t)
        elif isinstance(other, Objective):
            new = self.binary_operator(left=self, right=other, op=op)
        elif is_quantum_arg(other):
            new = self.binary_operator(left=self, right=Objective(args=[other]), op=op)
        else:
            t = op
            nother = Objective(args=[assign_variable(other)])
            new = self.binary_operator(left=self, right=nother, op=t)
        return new

    def _right_helper(self, op, other):
        """
        see the doc of _left_helper above for explanation
        """
        if isinstance(other, numbers.Number):
            t = lambda v: op(other, v)
            new = self.unary_operator(left=self, op=t)
        elif isinstance(other, Objective):
            new = self.binary_operator(left=other, right=self, op=op)
        elif is_quantum_arg(other):
            new = self.binary_operator(left=Objective(args=[other]), right=self, op=op)
        else:
            t = op
            nother = Objective(args=[assign_variable(other)])
            new = self.binary_operator(left=nother, right=self, op=t)
        return new

    def __mul__(self, other):
        return self._left_helper(numpy.multiply, other)

    def __add__(self, other):
        return self._left_helper(numpy.add, other)

    def __sub__(self, other):
        return self._left_helper(numpy.subtract, other)

    def __truediv__(self, other):
        return self._left_helper(numpy.true_divide, other)

    def __neg__(self):
        return self.unary_operator(left=self, op=lambda v: numpy.multiply(v, -1))

    def __pow__(self, other):
        return self._left_helper(numpy.power, other)

    def __rpow__(self, other):
        return self._right_helper(numpy.power, other)

    def __rmul__(self, other):
        return self._right_helper(numpy.multiply, other)

    def __radd__(self, other):
        return self._right_helper(numpy.add, other)

    def __rsub__(self, other):
        return self._right_helper(numpy.subtract, other)

    def __rtruediv__(self, other):
        return self._right_helper(numpy.true_divide, other)

    def __invert__(self):
        new = Objective(args=[self])
        return new**-1

    @classmethod
    def unary_operator(cls, left, op):
        """
        Arithmetical function for unary operations.
        Generally, called by the magic methods of Objective itself.
        Parameters
        ----------
        left: Objective:
            the objective to which op will be applied

        op: Callable:
            an operation to apply to left

        Returns
        -------
        Objective:
            Objective representing op applied to objective left.

        """
        return Objective(args=left.args, transformation=lambda *args: op(left.transformation(*args)))

    @classmethod
    def binary_operator(cls, left, right, op):
        """
        Core arithmetical method for creating differentiable callables of two Tequila Objectives and or Variables.

        this function, usually called by the convenience magic-methods of Observable objects, constructs a new Objective
        whose Transformation  is the JoinedTransformation of the lower arguments and transformations
        of the left and right objects, alongside op (if they are or can be rendered as objectives).
        In case one of left or right is a number, calls unary_operator instead.

        Parameters
        ----------
        left:
            the left hand argument to op
        right:
            the right hand argument to op.
        op: callable:
            an operation; a function object.

        Returns
        -------
        Objective:
            an objective whose Transformation is op acting on left and right.
        """

        if isinstance(right, numbers.Number):
            if isinstance(left, Objective):
                return cls.unary_operator(left=left, op=lambda E: op(E, right))
            else:
                raise TequilaException(
                    "BinaryOperator method called on types " + str(type(left)) + "," + str(type(right))
                )
        elif isinstance(left, numbers.Number):
            if isinstance(right, Objective):
                return cls.unary_operator(left=right, op=lambda E: op(left, E))
            else:
                raise TequilaException(
                    "BinaryOperator method called on types " + str(type(left)) + "," + str(type(right))
                )
        else:
            split_at = len(left.args)
            return Objective(
                args=left.args + right.args,
                transformation=JoinedTransformation(
                    left=left.transformation, right=right.transformation, split=split_at, op=op
                ),
            )

    def wrap(self, op):
        """
        convenience function for doing unary_operator with non-arithmetical operations like sin, cosine, etc.
        Parameters
        ----------
        op: callable:
            an operation to perform on the output of self

        Returns
        -------
        Objective:
            an objective which is evaluated as op(self)
        """
        return self.unary_operator(self, op)

    def apply(self, op):
        """alias for wrap"""
        return self.wrap(op=op)

    def count_measurements(self):
        """
        Count all measurements necessary for this objective:
        Function will iterate to all unique expectation values and count the
        number of Pauli strings in the corresponding Hamiltonians
        Returns
        -------
        Number of measurements required for this objective
        Measurements can be on different circuits (with regards to gates, depth, size, qubits)
        """
        return sum(E.count_measurements() for E in list(set(self.get_expectationvalues())))

    def count_expectationvalues(self, unique=True):
        """
        Parameters
        ----------
        unique: bool:
            whether or not to count identical expectationvalues as distinct.

        Returns
        -------
        int:
            how many (possibly, how many unique) expectationvalues are contained within the objective.

        """
        if unique:
            return len(set(self.get_expectationvalues()))
        else:
            return len(self.get_expectationvalues())

    def __repr__(self):
        return "f({})".format(self.extract_variables())

    def __str__(self):
        variables = self.extract_variables()
        if len(variables) > 5:
            variables = len(variables)

        types = [type(E) for E in self.get_expectationvalues()]
        types = list(set(types))

        if ExpectationValueImpl in types:
            if len(types) == 1:
                types = "not compiled"
            else:
                types = "partially compiled to " + str([t for t in types if t is not ExpectationValueImpl])

        unique = self.count_expectationvalues(unique=True)
        measurements = self.count_measurements()
        return (
            "Objective with {} unique expectation values\n"
            "total measurements = {}\n"
            "variables          = {}\n"
            "types              = {}".format(unique, measurements, variables, types)
        )

    def __call__(self, variables=None, initial_state=0, *args, **kwargs):
        """
        Return the output of the calculation the objective represents.

        Parameters
        ----------
        variables: dict:
            dictionary instantiating all variables that may appear within the objective.
        initial_state: int or QubitWaveFunction:
            the initial state of the circuit
        args
        kwargs

        Returns
        -------
        float:
            the result of the calculation represented by this objective.
        """
        variables = format_variable_dictionary(variables)
        # failsafe
        check_variables = {k: k in variables for k in self.extract_variables()}
        if not all(list(check_variables.values())):
            raise TequilaException(
                "Objective did not receive all variables:\n"
                "You gave\n"
                " {}\n"
                " but the objective depends on\n"
                " {}\n"
                " missing values for\n"
                " {}".format(variables, self.extract_variables(), [k for k, v in check_variables.items() if not v])
            )

        # avoid multiple evaluations
        evaluated = {}
        ev_array = []

        for arg in self.args:
            if arg in evaluated:  #
                ev_array.append(evaluated[arg])
                continue

            if is_quantum_arg(arg):
                result = _evaluate_quantum_arg(
                    arg, variables=variables, initial_state=initial_state, *args, **kwargs
                )
            else:
                result = arg(variables=variables, initial_state=initial_state, *args, **kwargs)
            
            evaluated[arg] = result
            ev_array.append(result)
        
        result = self.transformation(*ev_array)
        result = onp.asarray(result)
        if result.shape == ():
            val = result.item()
            return val
            # return float(result)
        elif len(result) == 1:
            return result[0]
            # return float(result[0])
        else:
            return result

    def contract(self):
        """
        Exists only to be convient in optimizers, which all contract over VectrObjectives.
        Returns
        -------
        Objective:
            itself.
        """
        return self

    def __len__(self):
        return 1

    def is_translated(self):
        """
        check if the objective was already translated to a quantum backend
        """

        quantum_args = self.get_expectationvalues()
        if not quantum_args:
            return False
        else:
            return all(getattr(a, 'is_compiled', True) for a in quantum_args)

    def print_tree(self, indent=0, label="root"):
        prefix = "  " * indent
        jt = self._transformation

        # 1.  print the current node
        if isinstance(jt, JoinedTransformation):
            op_name = getattr(jt.op, '__name__', str(jt.op))
            print(f"{prefix}[{label}] JoinedTransformation(op={op_name})")
            left_args  = self.args[:jt.split]
            right_args = self.args[jt.split:]
            left_obj   = Objective(args=left_args,  transformation=jt.left)
            right_obj  = Objective(args=right_args, transformation=jt.right)
            left_obj.print_tree (indent + 1, "left")
            right_obj.print_tree(indent + 1, "right")
        else:
            fn_name = getattr(jt, '__name__', repr(jt))
            print(f"{prefix}[{label}] transform={fn_name}, args={len(self.args)}")

        # 2.  *** always *** recurse into the arguments
        for idx, arg in enumerate(self.args):
            if isinstance(arg, Objective):
                arg.print_tree(indent + 1, f"arg[{idx}]")
            else:
                arg_type = type(arg).__name__
                arg_repr = repr(arg)[:50]
                print(f"{'  '*(indent+1)}[arg[{idx}]] {arg_type}: {arg_repr}")

def _evaluate_quantum_arg(quantum_arg, variables, initial_state, *args, **kwargs):
    # result = quantum_arg(variables=variables, initial_state=initial_state, *args, **kwargs)

    # if len(result)==2:
    #     real_val = result[0]
    #     imag_val = result[1]
    #     return real_val, imag_val

    # return result
    return quantum_arg(variables=variables, initial_state=initial_state, *args, **kwargs)

def ExpectationValue(U, H, optimize_measurements=False, *args, **kwargs) -> Objective:
    """
    Initialize an Objective which is just a single expectationvalue
    """
    if optimize_measurements:
        if optimize_measurements is True:
            # If optimize_measurements = True, then there are no further options
            # provided. Therefore, we will use the default values.
            options = None
        else:
            options = optimize_measurements
        commuting_parts, suggested_samples = compile_commuting_parts(H=H, options=options)
        result = 0.0
        for i, HandU in enumerate(commuting_parts):
            qwc, Um = HandU
            Etmp = ExpectationValue(H=qwc, U=U + Um, optimize_measurements=False, samples=suggested_samples[i])
            result += Etmp
        return result
    else:
        return Objective.ExpectationValue(U=U, H=H, *args, **kwargs)


class TequilaVariableException(TequilaException):
    def __str__(self):
        return "Error in tequila variable:" + self.message


class Variable:
    """
    Hashable class representing generalized variables.

    E.g, variables are the stand-in parameters of parametrized gates. We implement, here, the ability to manipulate
    this class as if they were floats, in order to allow for symbolic and functional operations on this type.

    Attributes
    ----------
    name:
        the name, identifying the variable.

    Methods
    -------
    extract_variables:
        returns a list containing only self.
    apply:
        generate an objective which applies a callable to self.
    wrap:
        alias for apply.
    """

    @property
    def name(self):
        return self._name

    def __lt__(self, other):
        return hash(self.name) < hash(other.name)

    def __hash__(self):
        return hash(self.name)

    def __init__(self, name: typing.Union[str, typing.Hashable]):
        """
        Parameters
        ----------
        name: hashable:
            a unique identifier for this variable. All Variable instances with the same name are considered the same.

        Raises
        ------
        TequilaVariableException
        """
        if name is None:
            raise TequilaVariableException("Tried to initialize a variable with None")
        if not isinstance(name, typing.Hashable) or not hasattr(name, "__hash__"):
            raise TequilaVariableException("Name of variable has to ba a hashable type")
        self._name = name

    def __call__(self, variables, *args, **kwargs):
        """
        Convenience function for easy usage
        :param variables: dictionary which carries all variable values
        :return: evaluate variable
        """
        return variables[self]

    def extract_variables(self):
        """
        Convenience function for easy usage
        :return: self wrapped in list
        """
        return [self]

    def map_variables(self, variables, *args, **kwargs):
        """
        see same function in Objective
        """
        if self in variables:
            return variables[self]
        else:
            return self

    def __eq__(self, other):
        if hasattr(other, "name"):
            return self.name == other.name
        else:
            return self.name == other

    def _left_helper(self, op, other):
        """
        function for use by magic methods, which all have an identical structure, differing only by the
        external operator they call.

        left helper is responsible for all 'self # other' operations. Note similarity

        Parameters
        ----------
        op: callable:
            the operation to be performed
        other:
            the right-hand argument of the operation to be performed

        Returns
        -------
        Objective:
            an Objective, who transform is op, acting on self and other
        """
        if isinstance(other, numbers.Number):
            t = lambda v: op(v, other)
            new = Objective(args=[self], transformation=t)
        elif isinstance(other, Variable):
            t = op
            new = Objective(args=[self, other], transformation=t)
        elif isinstance(other, Objective):
            new = Objective(args=[self])
            new = new.binary_operator(left=new, right=other, op=op)
        elif is_quantum_arg(other):
            new = Objective(args=[self, other], transformation=op)
        # elif isinstance(other, ExpectationValueImpl):
        #     new = Objective(args=[self, other], transformation=op)
        else:
            raise TequilaException(
                "unknown type in left_helper of objective arithmetics with operation {}: {}".format(
                    type(op), type(other)
                )
            )
        return new

    def _right_helper(self, op, other):
        """
        see _left_helper for details.
        """
        if isinstance(other, numbers.Number):
            t = lambda v: op(other, v)
            new = Objective(args=[self], transformation=t)
        elif isinstance(other, Variable):
            t = op
            new = Objective(args=[other, self], transformation=t)
        elif isinstance(other, Objective):
            new = Objective(args=[self])
            new = new.binary_operator(right=new, left=other, op=op)
        elif is_quantum_arg(other):
            new = Objective(args=[self, other], transformation=op)
        # elif isinstance(other, ExpectationValueImpl):
        #     new = Objective(args=[other, self], transformation=op)
        else:
            raise TequilaException(
                "unknown type in left_helper of objective arithmetics with operation {}: {}".format(
                    type(op), type(other)
                )
            )
        return new

    def __mul__(self, other):
        return self._left_helper(numpy.multiply, other)

    def __add__(self, other):
        return self._left_helper(numpy.add, other)

    def __sub__(self, other):
        return self._left_helper(numpy.subtract, other)

    def __truediv__(self, other):
        return self._left_helper(numpy.true_divide, other)

    def __neg__(self):
        return Objective(args=[self], transformation=lambda v: numpy.multiply(v, -1.0))

    def __pow__(self, other):
        return self._left_helper(numpy.power, other)

    def __rpow__(self, other):
        return self._right_helper(numpy.power, other)

    def __rmul__(self, other):
        return self._right_helper(numpy.multiply, other)

    def __radd__(self, other):
        return self._right_helper(numpy.add, other)

    def __rtruediv__(self, other):
        return self._right_helper(numpy.true_divide, other)

    def __invert__(self):
        new = Objective(args=[self])
        return new**-1.0

    def __len__(self):
        return 1

    def __ne__(self, other):
        if self.__eq__(other):
            return False
        else:
            return True

    def apply(self, other):
        assert callable(other)
        return Objective(args=[self], transformation=other)

    def wrap(self, other):
        return self.apply(other)

    def __repr__(self):
        return str(self.name)

    def toJson(self):
        import json

        return json.dumps(self, default=lambda o: o.__dict__)


class FixedVariable(float):
    """
    Wrapper over floats, to allow them to mimic Variable objects, for 'duck typing' of gate parameters.

    Methods
    -------
    apply:
        generate an objective with one argument -- self -- and other as transformation.
    wrap:
        alias for apply.
    """

    def __call__(self, *args, **kwargs):
        return self

    def apply(self, other):
        assert callable(other)
        return Objective(args=[self], transformation=other)

    def wrap(self, other):
        return self.apply(other)

    def map_variables(self, *args, **kwargs):
        return self


def format_variable_list(variables: typing.List[typing.Hashable]) -> typing.List[Variable]:
    """
    Convenience functions to assign tequila variables.
    Parameters
    ----------
    variables:
        a list with Hashables as elements.

    Returns
    -------
    list:
        a list with tq.Variable types as keys
    """
    if variables is None:
        return variables
    else:
        return [assign_variable(k) for k in variables]


def format_variable_dictionary(
    variables: typing.Dict[typing.Hashable, typing.Any],
) -> typing.Dict[Variable, typing.Any]:
    """
    Convenience function to assign tequila variables.
    Parameters
    ----------
    variables:
        a dictionary with Hashables as keys

    Returns
    -------
    dict:
        a dictionary with tq.Variable types as keys
    """
    if variables is None:
        return variables
    else:
        return Variables(variables)


def assign_variable(
    variable: typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable],
) -> typing.Union[Variable, FixedVariable]:
    """
    Convenience function; maps various objects into Variable, FixedVariable, or Variables, for easy duck-typing.

    Parameters
    ----------
    variable:
        a string, a number or a variable.

    Raises
    ------

    TequilaVariableException


    Returns
    -------
    Variable or FixedVariable:
        A duck-typing adjusted version of the input. If hashable, a Variable. If a number, a FixedVariable.
    """
    if isinstance(variable, str):
        return Variable(name=variable)
    elif isinstance(variable, Variable):
        return variable
    elif isinstance(variable, Objective):
        return variable
    elif isinstance(variable, FixedVariable):
        return variable
    elif isinstance(variable, numbers.Number):
        if not isinstance(variable, numbers.Real):
            raise TequilaVariableException("You tried to assign a complex number to a FixedVariable")
        return FixedVariable(variable)
    elif isinstance(variable, typing.Hashable):
        return Variable(name=variable)
    else:
        raise TequilaVariableException(
            "Only hashable types can be assigned to Variables. You passed down "
            + str(variable)
            + " type="
            + str(type(variable))
        )


class Variables(collections.abc.MutableMapping):
    """
    Dictionary-like object for tequila variables.

    Allows hashable types and variable types as keys

    Attributes
    ----------
    store: dict:
        the internal dictionary around which this structure is built.

    """

    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        return self.store[assign_variable(key)]

    def __setitem__(self, key, value):
        self.store[assign_variable(key)] = value

    def __delitem__(self, key):
        del self.store[assign_variable(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __str__(self):
        result = ""
        for k, v in self.items():
            result += "{} : {}\n".format(str(k), str(v))
        return result

    def __repr__(self):
        xdict = {k: v for k, v in self.items()}
        return xdict.__repr__()
