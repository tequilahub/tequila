import typing, copy, numbers
from jax import numpy as numpy

from tequila import TequilaException
from tequila.utils import JoinedTransformation, to_float
from tequila.hamiltonian import paulis


class ExpectationValueImpl:
    """
    Internal Object, do not use from the outside
    the implementation of Expectation Values as a class. Capable of being simulated, and differentiated.
    common arithmetical operations like addition, multiplication, etc. are defined, to return Objective objects.
    :param U: a QCircuit, for preparing a state
    :param H: a Hamiltonian, whose expectation value with the state prepared by U is to be determined.
    '''
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
            return paulis.QubitHamiltonian.init_unit()
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

    def __init__(self, U=None, H=None):
        self._unitary = copy.deepcopy(U)
        self._hamiltonian = copy.deepcopy(H)


class Objective:
    """
    the class which represents mathematical manipulation of ExpectationValue and Variable objects. Capable of being simulated,
    and differentiated with respect to the Variables of its Expectationvalues or the Variables themselves
    :param args: an iterable of ExpectationValue's.
    :param transformation: a callable whose positional arguments (potentially, by nesting in a JoinedTransformation)
        are args
    :param simulator: a tequila simulator object. If provided, Objective is callable.

    """

    def __init__(self, args: typing.Iterable = None, transformation: typing.Callable = None, simulator=None):
        if args is None:
            self._args = tuple()
        else:
            self._args = tuple(args)
        self._transformation = transformation
        self.simulator = simulator
        self.last = None

    def extract_variables(self):
        """
        Extract all variables on which the objective depends
        :return: List of all Variables
        """
        variables = []
        for arg in self.args:
            variables += arg.extract_variables()

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
    def ExpectationValue(cls, U=None, H=None):
        """
        Initialize a wrapped expectationvalue directly as Objective
        """
        E = ExpectationValueImpl(H=H, U=U)
        return Objective(args=[E])

    @property
    def transformation(self) -> typing.Callable:
        if self._transformation is None:
            return numpy.sum
        else:
            return self._transformation

    @property
    def args(self) -> typing.Tuple:
        '''
        :return: self.args
        '''
        if self._args is None:
            return tuple()
        else:
            return self._args

    def left_helper(self, op, other):
        '''
        function for use by magic methods, which all have an identical structure, differing only by the
        external operator they call. left helper is responsible for all 'self # other' operations
        :param op: the operation to be performed
        :param other: the right-hand argument of the operation to be performed
        :return: an Objective, who transform is the joined_transform of self with op, acting on self and other
        '''
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

    def right_helper(self, op, other):
        '''
        see the doc of left_helper above for explanation
        '''
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
        return self.left_helper(numpy.multiply, other)

    def __add__(self, other):
        return self.left_helper(numpy.add, other)

    def __sub__(self, other):
        return self.left_helper(numpy.subtract, other)

    def __truediv__(self, other):
        return self.left_helper(numpy.true_divide, other)

    def __neg__(self):
        return self.unary_operator(left=self, op=lambda v: numpy.multiply(v, -1))

    def __pow__(self, other):
        return self.left_helper(numpy.float_power, other)
        # return self.binary_operator(left=self, op=lambda E: numpy.float_power(E, power))

    def __rpow__(self, other):
        return self.right_helper(numpy.float_power, other)
        # return new.binary_operator(left=new,right=other, op=lambda l, r: numpy.float_power(r, l))

    def __rmul__(self, other):
        return self.right_helper(numpy.multiply, other)

    def __radd__(self, other):
        return self.right_helper(numpy.add, other)

    def __rtruediv__(self, other):
        return self.right_helper(numpy.true_divide, other)

    def __invert__(self):
        new = Objective(args=[self])
        return new ** -1

    @classmethod
    def unary_operator(cls, left, op):
        return Objective(args=left.args,
                         transformation=lambda *args: op(left.transformation(*args)))

    @classmethod
    def binary_operator(cls, left, right, op):
        '''
        this function, usually called by the convenience magic-methods of Observable objects, constructs a new Objective
        whose Transformation  is the JoinedTransformation of the lower arguments and transformations
        of the left and right objects, alongside op (if they are or can be rendered as objectives). In case one of left or right
        is a number, calls unary_operator instead.
        :param left: the left hand argument to op
        :param right: the right hand argument to op.
        :param op: an operation; a function object.
        :return: an objective whose Transformation  is the JoinedTransformation of the lower arguments and transformations
        of the left and right objects, alongside op (if they are or can be rendered as objectives). In case one of left or right
        is a number, calls unary_operator instead.
        '''

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
        '''
        convenience function for doing unary_operator with non-arithmetical operations like sin, cosine, etc.
        :param op: an operation to perform on the output of self
        :return: an objective which is evaluated as op(self)
        '''
        return self.unary_operator(self, op)

    def __repr__(self):
        string = "Objective with " + str(len(self.args)) + " arguments"
        if self.last is not None:
            string += " , last call value = " + str(self.last)
        return string

    def __call__(self, variables: typing.Dict[typing.Hashable, numbers.Real], simulator=None, samples=None, *args,
                 **kwargs):
        '''
        Evaluates the expression which Objective represents, if possible.
        :param samples:
        :return:
        '''

        if self.has_expectationvalues():
            if simulator is None:
                simulator = self.simulator
            if simulator is None:
                raise TequilaException("No simulator was specified")
            return to_float(simulator(self, variables=variables, samples=samples, *args, **kwargs))
        else:
            # in case that no simulator is actually needed
            evaluated_args = [variables[arg] for arg in self.args]
            return to_float(self.transformation(*evaluated_args))


def ExpectationValue(U, H) -> Objective:
    """
    Initialize an Objective which is just a single expectationvalue
    """
    return Objective.ExpectationValue(U=U, H=H)


class TequilaVariableException(TequilaException):
    def __str__(self):
        return "Error in tequila variable:" + self.message


class SympyVariable:
    '''
    TODO: can we pleaseeeeee get rid of this thing, Jakob? pretty please?
    '''

    def __init__(self, name=None, value=None):
        self._value = value
        self._name = name

    def __call__(self, *args, **kwargs):
        return self._value

    def __sub__(self, other):
        return SympyVariable(name=self._name, value=self._value - other)

    def __add__(self, other):
        return SympyVariable(name=self._name, value=self._value + other)

    def __mul__(self, other):
        return SympyVariable(name=self._name, value=self._value * other)

    def __neg__(self):
        return SympyVariable(name=self._name, value=-self._value)


class Variable:

    @property
    def name(self):
        return self._name

    def __hash__(self):
        return hash(self.name)

    def __init__(self, name: typing.Union[str, typing.Hashable]):
        if not isinstance(name, typing.Hashable) or not hasattr(name, "__hash__"):
            raise TequilaVariableException("Name of variable has to ba a hashable type")
        self._name = name

    def __call__(self, variables):
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

    def left_helper(self, op, other):
        '''
        function for use by magic methods, which all have an identical structure, differing only by the
        external operator they call. left helper is responsible for all 'self # other' operations. Note similarity
        to the same function in Objective.
        :param op: the operation to be performed
        :param other: the right-hand argument of the operation to be performed
        :return: an Objective, who transform is op, acting on self and other
        '''
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

    def right_helper(self, op, other):
        '''
        see left helper above
        '''
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
        return self.left_helper(numpy.multiply, other)

    def __add__(self, other):
        return self.left_helper(numpy.add, other)

    def __sub__(self, other):
        return self.left_helper(numpy.subtract, other)

    def __truediv__(self, other):
        return self.left_helper(numpy.true_divide, other)

    def __neg__(self):
        return Objective(args=[self], transformation=lambda v: numpy.multiply(v, -1))

    def __pow__(self, other):
        return self.left_helper(numpy.float_power, other)

    def __rpow__(self, other):
        return self.right_helper(numpy.float_power, other)

    def __rmul__(self, other):
        return self.right_helper(numpy.multiply, other)

    def __radd__(self, other):
        return self.right_helper(numpy.add, other)

    def __rtruediv__(self, other):
        return self.right_helper(numpy.true_divide, other)

    def __invert__(self):
        new = Objective(args=[self])
        return new ** -1

    def __iadd__(self, other):
        self._value += other
        return self

    def __isub__(self, other):
        self._value -= other
        return self

    def __imul__(self, other):
        self._value *= other
        return self

    def __idiv__(self, other):
        self._value /= other
        return self

    def __ipow__(self, other):
        self._value **= other
        return self

    def __lt__(self, other):
        return self.value < other

    def __gt__(self, other):
        return self.value > other

    def __ge__(self, other):
        return self.value >= other

    def __le__(self, other):
        return self.value <= other

    def __ne__(self, other):
        if self.__eq__(other):
            return False
        else:
            return True

    def __repr__(self):
        return str(self.name)


class FixedVariable(float):

    def __call__(self, *args, **kwargs):
        return self


def assign_variable(variable: typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable]) -> typing.Union[Variable, FixedVariable]:
    """
    :param variable: a string, a number or a variable
    :return: Variable or FixedVariable depending on the input
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
    elif  hasattr(variable, "evalf"): # evalf detects sympy types ... not differentiable, hidden in the type hinting since it should not really be used
        return SympyVariable(value=variable)
    elif isinstance(variable, typing.Hashable):
        return Variable(name=variable)
    else:
        raise TequilaVariableException("Only hashable types can be assigned to Variables. You passed down " + str(variable) + " type=" + str(type(variable)))