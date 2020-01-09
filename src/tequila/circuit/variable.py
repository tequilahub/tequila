import numbers
from tequila import TequilaException
from tequila.tools import number_to_string
from jax import numpy as numpy
from jax import numpy as np
from tequila.objective.objective import Objective,ExpectationValueImpl
import copy


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


def enforce_number(number, numeric_type=complex) -> complex:
    """
    Try to convert number into a numeric_type
    No converion is tried when number is already a numeric type
    If numeric_type is set to None, then no conversion is tried
    :param number: the number to convert
    :param numeric_type: the numeric type into which conversion shall be tried when number is not identified as a number
    :return: converted number

    TODO REMOVE; this thing seems depreated
    """
    if isinstance(number, numbers.Number):
        return number
    elif numeric_type is None:
        return number
    else:
        numeric_type(number)


def enforce_number_decorator(*numeric_types):
    """
    :param numeric_types: type for argument 0, 1, 3. Set to none if an argument shall not be converted
    :return: If the arguments are not numbers this decorator will try to convert them to the given numeric_types
    """

    def decorator(function):
        def wrapper(self, *args):
            assert (len(numeric_types) == len(args))
            converted = [enforce_number(number=x, numeric_type=numeric_types[i]) for i, x in enumerate(args)]
            return function(self, *converted)

        return wrapper

    return decorator

class Variable:
    @property
    def variables(self):
        if self.name is not None:
            return {self.name: self.value}
        else:
            return dict()

    @property
    def parameter_list(self):
        return [self]

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = float(value)

    @property
    def name(self) -> str:
        return self._name

    def __init__(self, name=None, value=None):
        if value is not None:
            if type(value) in [float, int]:
                self._value = float(value)
            else:
                self._value = np.real(value)
        else:
            self._value = None

        if type(name) is str:
            self._name = name
        elif name is not None:
            self._name = str(name)
        else:
            self._name = name

        self.is_default = name is None

        if self._value is None:
            self.needs_init = True
        else:
            self.needs_init = False

    def has_var(self, x):
        '''
        convenience function for checking variable identity.
        :param x: a Variable, str, or Dict
        :return: bool: whether or not self is, is named, or has a name which appears in the keys of, x
        '''
        if type(x) is Variable:
            return self == x
        elif type(x) is str:
            return self._name == x
        elif type(x) is dict:
            return self._name in x.keys()
        else:
            raise TypeError('Unsupported type')

    def update(self, x):
        '''
        :param x: dict, Variable, or number.
        convenience function for updating the value of a variable; allows the arguments higher order updates (QCircuit, Transform)
        to be passed down so that the Variable value updates.
        '''

        if type(x) is dict:
            for k in x.keys():
                if self.name == k:
                    self._value = x[k]
        elif type(x) is Variable:
            if x.name == self.name:
                self._value = x.value
        else:
            self._value = float(x)

    def update_variables(self, variables):
        self.update(variables)

    def extract_variables(self):
        return {self.name:self.value}

    def __eq__(self, other):
        if type(self) == type(other):
            if self.name == other.name and self.value == other.value:
                return True
        return False

    def left_helper(self,op,other):
        '''
        function for use by magic methods, which all have an identical structure, differing only by the
        external operator they call. left helper is responsible for all 'self # other' operations. Note similarity
        to the same function in Objective.
        :param op: the operation to be performed
        :param other: the right-hand argument of the operation to be performed
        :return: an Objective, who transform is op, acting on self and other
        '''
        if isinstance(other, numbers.Number):
            t = lambda v: op(v,other)
            new = Objective(args=[self], transformation=t)
        elif isinstance(other, Variable):
            t = op
            new = Objective(args=[self, other], transformation=t)
        elif isinstance(other, Objective):
            new=Objective(args=[self])
            new=new.binary_operator(left=new,right=other,op=op)
        elif isinstance(other,ExpectationValueImpl):
            new=Objective(args=[self,other],transformation=op)
        return new

    def right_helper(self,op,other):
        '''
        see left helper above
        '''
        if isinstance(other, numbers.Number):
            t = lambda v: op(other,v)
            new = Objective(args=[self], transformation=t)
        elif isinstance(other, Variable):
            t = op
            new = Objective(args=[other,self], transformation=t)
        elif isinstance(other, Objective):
            new=Objective(args=[self])
            new=new.binary_operator(right=new,left=other,op=op)
        elif isinstance(other,ExpectationValueImpl):
            new=Objective(args=[other,self],transformation=op)
        return new

    def __mul__(self, other):
        return self.left_helper(numpy.multiply,other)

    def __add__(self, other):
        return self.left_helper(numpy.add,other)

    def __sub__(self, other):
        return self.left_helper(numpy.subtract,other)

    def __truediv__(self, other):
        return self.left_helper(numpy.true_divide,other)


    def __neg__(self):
        return Objective(args=[self], transformation=lambda v: numpy.multiply(v, -1))

    def __pow__(self, other):
        return self.left_helper(numpy.float_power,other)
        # return self.binary_operator(left=self, op=lambda E: numpy.float_power(E, power))
    def __rpow__(self, other):
        return self.right_helper(numpy.float_power,other)
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

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def __call__(self):
        return self.value

    def __repr__(self):
        return str(self.name) + ', ' + str(self._value)

    def __str__(self):
        if self.name is None:
            return number_to_string(self.__call__())
        else:
            return self.name

    def __float__(self):
        # TODO remove
        return float(self.value)
