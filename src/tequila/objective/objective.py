import typing, copy, numbers

from tequila import TequilaException
from tequila.utils import JoinedTransformation, to_float
from tequila.hamiltonian import paulis
from tequila.grouping.binary_rep import BinaryHamiltonian
from tequila.autograd_imports import numpy

import collections


class ExpectationValueImpl:
    """
    Implements the (uncompiled) Expectation Value as a class. Should not be called directly.

    common arithmetical operations like addition, multiplication, etc. are defined, to return Objective objects.

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
    def H(self):
        if self._hamiltonian is None:
            return paulis.QubitHamiltonian.unit()
        else:
            return self._hamiltonian

    def extract_variables(self) -> typing.Dict[str, numbers.Real]:
        result = []
        if self.U is not None:
            result = self.U.extract_variables()
        return result

    def update_variables(self, variables: typing.Dict[str, numbers.Real]):
        if self.U is not None:
            self.U.update_variables(variables)

    def __init__(self, U=None, H=None, contraction=None, shape=None):
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

    def __call__(self, *args, **kwargs):
        raise TequilaException(
            "Tried to call uncompiled ExpectationValueImpl, compile your objective before calling with tq.compile(objective) or evaluate with tq.simulate(objective)")

    def info(self, short=True, *args, **kwargs):
        if short:
            print("Expectation Value with {qubits} active qubits and {paulis} paulistrings".format(
                qubits=len(self.U.qubits), paulis=len(self.H)))
        else:
            print("Hamiltonian:\n", str(self.H))
            print("\n", str(self.U))


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
        else:
            return "free"

    def extract_variables(self):
        """
        Extract all variables on which the objective depends
        :return: List of all Variables
        """
        variables = []
        for arg in self.args:
            if hasattr(arg, 'extract_variables'):
                variables += arg.extract_variables()
            else:
                variables += []

        return list(set(variables))

    def is_expectationvalue(self):
        """
        :return: bool: whether or not this objective is just a wrapped ExpectationValue
        """
        return len(self.args) == 1 and self._transformation is None and type(self.args[0]) is ExpectationValueImpl

    def has_expectationvalues(self):
        """
        :return: bool: wether or not this objective has expectationvalues or is just a function of the variables
        """
        # testing if all arguments are only variables and give back the negative
        return not all([hasattr(arg, "name") for arg in self.args])

    @classmethod
    def ExpectationValue(cls, U=None, H=None, *args, **kwargs):
        """
        Initialize a wrapped expectationvalue directly as Objective
        """
        E = ExpectationValueImpl(H=H, U=U, *args, **kwargs)
        return Objective(args=[E])

    @property
    def transformation(self) -> typing.Callable:
        if self._transformation is None:
            return lambda x: x
        else:
            return self._transformation

    @property
    def args(self) -> typing.Tuple:

        if self._args is None:
            return tuple()
        else:
            return self._args

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
        elif isinstance(other, ExpectationValueImpl):
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
        elif isinstance(other, ExpectationValueImpl):
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
        return self._left_helper(numpy.float_power, other)

    def __rpow__(self, other):
        return self._right_helper(numpy.float_power, other)

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
        return new ** -1

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
        return Objective(args=left.args,
                         transformation=lambda *args: op(left.transformation(*args)))

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
                    'BinaryOperator method called on types ' + str(type(left)) + ',' + str(type(right)))
        elif isinstance(left, numbers.Number):
            if isinstance(right, Objective):
                return cls.unary_operator(left=right, op=lambda E: op(left, E))
            else:
                raise TequilaException(
                    'BinaryOperator method called on types ' + str(type(left)) + ',' + str(type(right)))
        else:
            split_at = len(left.args)
            return Objective(args=left.args + right.args,
                             transformation=JoinedTransformation(left=left.transformation, right=right.transformation,
                                                                 split=split_at, op=op))

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

    def get_expectationvalues(self):
        """
        Returns
        -------
        list:
            all the expectation values that make up the objective.
        """
        return [arg for arg in self.args if hasattr(arg, "U")]

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

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        variables = self.extract_variables()
        types = [type(E) for E in self.get_expectationvalues()]
        types = list(set(types))

        if ExpectationValueImpl in types:
            if len(types) == 1:
                types = "not compiled"
            else:
                types = "partially compiled to " + str([t for t in types if t is not ExpectationValueImpl])

        unique = self.count_expectationvalues(unique=True)
        return "Objective with {} unique expectation values\n" \
               "variables = {}\n" \
               "types     = {}".format(unique, variables, types)

    def __call__(self, variables=None, *args, **kwargs):
        """
        Return the output of the calculation the objective represents.

        Parameters
        ----------
        variables: dict:
            dictionary instantiating all variables that may appear within the objective.
        args
        kwargs

        Returns
        -------
        float:
            the result of the calculation represented by this objective.
        """
        variables = format_variable_dictionary(variables)
        # avoid multiple evaluations
        evaluated = {}
        ev_array = []
        for E in self.args:
            if E not in evaluated:
                expval_result = E(variables=variables, *args, **kwargs)
                evaluated[E] = expval_result
            else:
                expval_result = evaluated[E]
            ev_array.append(expval_result)
        return self.transformation(*ev_array)


def ExpectationValue(U, H, optimize_measurements: bool=False, *args, **kwargs) -> Objective:
    """
    Initialize an Objective which is just a single expectationvalue
    """
    if optimize_measurements:
        binary_H = BinaryHamiltonian.init_from_qubit_hamiltonian(H)
        commuting_parts = binary_H.commuting_groups()
        result = Objective()
        for cH in commuting_parts:
            qwc, Um = cH.get_qubit_wise()
            Etmp = ExpectationValue(H=qwc, U=U + Um, optimize_measurements=False)
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

    def __hash__(self):
        return hash(self.name)

    def __init__(self, name: typing.Union[str, typing.Hashable]):
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

    def __eq__(self, other):
        return type(self) == type(other) and self.name == other.name

    def _left_helper(self, op, other):
        """
        function for use by magic methods, which all have an identical structure, differing only by the
        external operator they call.

        left helper is responsible for all 'self # other' operations. Note similarity
        to the same function in Objective.

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
        elif isinstance(other, ExpectationValueImpl):
            new = Objective(args=[self, other], transformation=op)
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
        elif isinstance(other, ExpectationValueImpl):
            new = Objective(args=[other, self], transformation=op)
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
        return Objective(args=[self], transformation=lambda v: numpy.multiply(v, -1))

    def __pow__(self, other):
        return self._left_helper(numpy.float_power, other)

    def __rpow__(self, other):
        return self._right_helper(numpy.float_power, other)

    def __rmul__(self, other):
        return self._right_helper(numpy.multiply, other)

    def __radd__(self, other):
        return self._right_helper(numpy.add, other)

    def __rtruediv__(self, other):
        return self._right_helper(numpy.true_divide, other)

    def __invert__(self):
        new = Objective(args=[self])
        return new ** -1

    def __ne__(self, other):
        if self.__eq__(other):
            return False
        else:
            return True

    def apply(self, other):
        assert (callable(other))
        return Objective(args=[self], transformation=other)

    def wrap(self, other):
        return self.apply(other)

    def __repr__(self):
        return str(self.name)


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
        assert (callable(other))
        return Objective(args=[self], transformation=other)

    def wrap(self, other):
        return self.apply(other)


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


def format_variable_dictionary(variables: typing.Dict[typing.Hashable, typing.Any]) -> typing.Dict[
    Variable, typing.Any]:
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


def assign_variable(variable: typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable]) -> typing.Union[
    Variable, FixedVariable]:
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
            "Only hashable types can be assigned to Variables. You passed down " + str(variable) + " type=" + str(
                type(variable)))


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
        return self.__str__()
