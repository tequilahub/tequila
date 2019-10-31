from openvqe import OpenVQEException
from openvqe.circuit import transform
from functools import total_ordering
from openvqe import copy
from openvqe import numbers


class SympyVariable:

    def __init__(self, name=None, value=None):
        self._name = name
        self._value = value

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

@total_ordering
class Variable():
    _instances = []

    @property
    def eval(self):
        if hasattr(self, 'value'):
            if hasattr(self, 'transform') and self.transform is not None:
                val = self.value
                for t in self.transform:
                    val = t(val)
                return val
            else:
                return self.value
        else:
            return None

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    @property
    def name(self):
        if self._name is None:
            return "none"
        else:
            return self._name

    def __init__(self, value=None, name: str = None, transform=None):
        if isinstance(value, numbers.Number):
            self._value = value
        else:
            print("VALUE IS ", type(value))
            raise Exception("value needs to be a number")
        self._name = name

        if hasattr(transform, '__iter__'):
            assert all([callable(f) for f in transform])
            self.transform = transform
        elif transform is None:
            self.transform = []
        elif callable(transform):
            self.transform = [transform]

    def __neg__(self):
        return self.with_transform(transform.Multiply(-1))

    @enforce_number_decorator(complex)
    def __sub__(self, other):
        return self.with_transform(transform.Add(other * -1))

    @enforce_number_decorator(complex)
    def __add__(self, other: float):
        return self.with_transform(transform.Add(other))

    @enforce_number_decorator(complex)
    def __radd__(self, other: float):
        if other == 0:
            return self
        else:
            return self.with_transform(transform.Add(other))

    @enforce_number_decorator(complex)
    def __mul__(self, other):
        return self.with_transform(transform.Multiply(other))

    @enforce_number_decorator(complex)
    def __pow__(self, other):
        return self.with_transform(transform.Power(other))

    @enforce_number_decorator(complex)
    def __div__(self, other):
        return self.with_transform(transform.Divide(other))

    @enforce_number_decorator(complex)
    def __rdiv__(self, other):
        if other == 0:
            return 0
        else:
            return self.with_transform(transform.Multiply(other)).with_transform(self.Divide(self())).with_transform(
                self.Divide(self()))

    def __truediv__(self, other):
        return self.with_transform(transform.Divide(other))

    def __getstate__(self):
        return self

    def __lt__(self, other):
        if isinstance(other, numbers.Number):
            return self.eval() < other
        if self.eval < other.eval:
            return False
        return True

    def __eq__(self, other):
        if isinstance(other, numbers.Number):
            return self.value == other
        if self.name != other.name:
            return False
        if self.eval != other.eval:
            print("eval differs")
            return False
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

    def with_transform(self, transform, replace=False):

        clone = Variable(name=self.name, value=self._value, transform=copy.deepcopy(self.transform))
        if replace == True:
            if hasattr(transform, '__iter__'):
                assert all([callable(f) for f in transform])
                clone.transform = transform
            elif callable(transform):
                clone.transform = [transform]
            elif transform is None:
                pass
            else:
                raise OpenVQEException(
                    'invalid object passed to transform; must be a (sequence of) callable function(s)')

        else:
            if hasattr(transform, '__iter__'):
                assert all([callable(f) for f in transform])
                clone.transform.extend(transform)
            elif callable(transform):
                clone.transform.append(transform)
            elif transform is None:
                pass
            else:
                raise OpenVQEException(
                    'invalid object passed to transform; must be a (sequence of) callable function(s)')

        return clone

    def __call__(self):
        return self.eval

    def __repr__(self):
        return 'Variable ' + self.name + ': Value = ' + str(self._value) + ': Eval = ' + str(self.eval)



