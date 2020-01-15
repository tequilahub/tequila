import numbers, typing
from tequila import TequilaException
from tequila.tools import number_to_string
from jax import numpy as numpy
from jax import numpy as np
from tequila.objective.objective import Objective, ExpectationValueImpl
import copy


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

def assign_variable(variable: typing.Union[typing.List[typing.Hashable], typing.List[numbers.Real], typing.List[Variable]]) -> typing.Union[Variable, FixedVariable]:
    """
    :param variable: a string, a number or a variable
    :return: Variable or FixedVariable depending on the input
    """
    if isinstance(variable, str):
        return Variable(name=variable)
    elif isinstance(variable, Variable):
        return variable
    elif hasattr(variable, 'args'):
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
        raise TequilaVariableException("Only hashable types can be assigned to Variables")
