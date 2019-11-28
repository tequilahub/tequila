import copy
import numbers

from tequila import TequilaException
from tequila.tools import number_to_string
from inspect import signature
import numpy as np
import operator
import copy


class SympyVariable:

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

    def __eq__(self, other):
        if type(self) == type(other):
            if self.name == other.name and self.value == other.value:
                return True
        return False

    def __add__(self, other: float):
        return Transform(operator.add, [self, other])

    def __radd__(self, other: float):
        if other == 0:
            return self
        else:
            return Transform(operator.add, [other, self])

    def __sub__(self, other):
        return Transform(operator.sub, [self, other])

    def __rsub__(self, other):
        return Transform(operator.sub, [other, self])

    def __mul__(self, other):
        return Transform(operator.mul, [self, other])

    def __rmul__(self, other):

        return Transform(operator.sub, [other, self])

    def __neg__(self):
        return Transform(operator.mul, [self, -1.])

    def __div__(self, other):
        return Transform((operator.itruediv), [self, other])

    def __rdiv__(self, other):
        return Transform(operator.truediv, [other, self])

    def __truediv__(self, other):
        return Transform(operator.truediv, [self, other])

    def __pow__(self, other):
        return Transform(operator.pow, [self, other])

    def __rpow__(self, other):
        return Transform(operator.pow, [other, self])

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

    def __getstate__(self):
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
        return self.name + ', ' + str(self._value)

    def __str__(self):
        if self.name is None:
            return number_to_string(self.__call__())
        else:
            return self.name

    def __float__(self):
        #TODO remove
        return float(self.value)


class Transform:

    @property
    def parameter_list(self):
        vl = []
        for obj in self.args:
            if type(obj) is Variable:
                vl.append(obj)
            elif type(obj) is Transform:
                vl.extend(obj.parameter_list)
        return vl

    @property
    def variables(self):
        vl = {}
        for obj in self.args:
            if type(obj) is Variable:
                if obj.name is not None and obj.name not in vl:
                    vl[obj.name] = obj.value
                elif obj.name is not None:
                    if not np.isclose(vl[obj.name], obj.value):
                        raise TequilaException(
                            'found two variables with the same name and different values, this is unacceptable')
            elif type(obj) is Transform:
                for k, v in obj.variables.items():
                    if k is not None and k not in vl:
                        vl[k] = v
                    elif k is not None:
                        if not np.isclose(vl[k], v):
                            raise TequilaException(
                                'found two variables with the same name and different values, this is unacceptable')
            else:
                pass
        return vl

    @property
    def eval(self):
        new_a = []
        for arg in self.args:
            if hasattr(arg, '__call__'):
                new_a.append(arg())
            else:
                new_a.append(arg)

        return self.f(*new_a)

    def __init__(self, func, args):
        assert callable(func)
        self.args = args
        self.f = DressedOperator(op=func)

    def update(self, pars):
        for arg in self.args:
            if type(pars) is dict:
                for k, v in pars.items():
                    if hasattr(arg, 'update'):
                        if k in arg.variables.keys():
                            arg.update({k: v})
            elif type(pars) is list:
                if hasattr(arg, 'has_var'):
                    for par in pars:
                        if arg.has_var(par):
                            arg.update(par)

    def has_var(self, x):
        '''
        :param x: dict, Variable, or str
        checks if (any of the ) variable(s) passed are present within the transform. Looks for them by name, NOT value.
        return: bool: true if a match found else false.
        '''
        for k, v in self.variables.items():
            if type(x) is dict:
                if k in x.keys():
                    return True
            if type(x) is Variable:
                if k == x.name:
                    return True
            if type(x) is str:
                if k == x:
                    return True

        return False

    def __call__(self):
        return self.eval

    def __eq__(self, other):
        if hasattr(other, 'eval'):
            if hasattr(other, 'variables'):
                if self.eval == other.eval and self.variables == other.variables:
                    ### is this safe?
                    return True
        return False

    def __add__(self, other: float):
        return Transform(operator.add, [self, other])

    def __radd__(self, other: float):
        if other == 0:
            return self
        else:
            return Transform(operator.add, [other, self])

    def __sub__(self, other):
        return Transform(operator.sub, [self, other])

    def __rsub__(self, other):
        return Transform(operator.sub, [other, self])

    def __mul__(self, other):
        return Transform(operator.mul, [self, other])

    def __rmul__(self, other):

        return Transform(operator.sub, [other, self])

    def __neg__(self):
        return Transform(operator.mul, [self, -1])

    def __div__(self, other):
        return Transform(operator.truediv, [self, other])

    def __rdiv__(self, other):
        return Transform(operator.truediv, [other, self])

    def __truediv__(self, other):
        return Transform(operator.truediv, [self, other])

    def __rtruediv__(self, other):
        return Transform(operator.truediv, [other, self])

    def __pow__(self, other):
        return Transform(operator.pow, [self, other])

    def __rpow__(self, other):
        return Transform(operator.pow, [other, self])

    def __getstate__(self):
        return self

    def __lt__(self, other):
        return self.eval < other

    def __gt__(self, other):
        return self.eval > other

    def __ge__(self, other):
        return self.eval >= other

    def __le__(self, other):
        return self.eval <= other

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

    def __float__(self):
        number = self.eval
        if hasattr(number, "imag"):
            assert (number.imag == 0.0)
            number = number.real
        return float(number)

    def __complex__(self):
        number = self.eval
        if hasattr(number, "imag"):
            assert (number.imag == 0.0)
            number = number.real
        return complex(number)

    def __repr__(self):
        funcpart = str(self.f) + ' acting on: '
        argpart = '('
        for i in range(len(self.args)):
            argpart += str(self.args[i])
            if i < len(self.args) - 1:
                argpart += ', '
        argpart += ')'
        val = str(self())
        return funcpart + argpart + ', val=' + val

    def __str__(self):
        result = ""
        fname = str(self.f)

        if len(self.args) == 2:
            result += "(" + str(self.args[0]) + ")" + fname + "(" + str(self.args[-1]) + ")"
        else:
            result += fname + "(" + str(self.args) + ")"

        return result


def has_variable(obj, var):
    '''
    wrapper over the has_var method of transform and variable; for easy, no-error use by higher order functions.
    :param obj: any, meant to be a variable or Transform
    :param var: any, meant to be a variable, a dict, or a str, such as is suitable for has_var
    '''
    if hasattr(obj, 'has_var'):
        return obj.has_var(var)
    else:
        return False


class DressedOperator:
    """
    Can be a function later
    Currently the gradient needs information about the wrapped operator
    """

    _operator_names = {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "truediv": "/",
        "pow": "**"
    }

    def __str__(self):
        name = self.op.__name__
        if name in self._operator_names:
            return self._operator_names[name]
        else:
            return name

    def __init__(self, op):
        self.op = op

    def wrapper(self, l, r=None, *args, **kwargs):
        if type(l) in [Variable, Transform]:
            lv = l()
        else:
            lv = l
        if type(r) in [Variable, Transform]:
            rv = r()
        else:
            rv = r
        return self.op(lv, rv)

    def __call__(self, *args, **kwargs):
        return self.wrapper(*args, **kwargs)
