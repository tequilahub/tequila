import numbers
from tequila import TequilaException
from tequila.tools import number_to_string
from jax import numpy as numpy
from jax import numpy as np
from tequila.utils import JoinedTransformation
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

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            t=lambda v: numpy.multiply(v,other)
            new=Transform(args=[self],transformation=t)
        elif isinstance(other,Variable):
            t=numpy.multiply
            new=Transform(args=[self,other],transformation=t)
        elif isinstance(other,Transform):
            new=other.__rmul__(self)
        return new

    def __add__(self, other):
        if isinstance(other, numbers.Number):
            t=lambda v: numpy.add(v,other)
            new=Transform(args=[self],transformation=t)
        elif isinstance(other,Variable):
            t=numpy.add
            new=Transform(args=[self,other],transformation=t)
        elif isinstance(other,Transform):
            new=other.__radd__(self)
        return new

    def __sub__(self, other):
        if isinstance(other, numbers.Number):
            t=lambda v: numpy.subtract(v,other)
            new=Transform(args=[self],transformation=t)
        elif isinstance(other,Variable):
            t=lambda v1,v2: numpy.subtract(v1,v2)
            new=Transform(args=[self,other],transformation=t)
        elif isinstance(other,Transform):
            new=other.__rsub__(self)
        return new

    def __truediv__(self, other):
        if isinstance(other, numbers.Number):
            t=lambda v: numpy.true_divide(v,other)
            new=Transform(args=[self],transformation=t)
        elif isinstance(other,Variable):
            t=numpy.true_divide
            new=Transform(args=[self,other],transformation=t)
        elif isinstance(other,Transform):
            new=other.__rtruediv__(self)
        return new

    def __neg__(self):
        return Transform(args=[self],transformation=lambda v: numpy.multiply(v,-1))

    def __pow__(self, other):
        if isinstance(other, numbers.Number):
            t=lambda v: numpy.float_power(v,other)
            new=Transform(args=[self],transformation=t)
        elif isinstance(other,Variable):
            t=numpy.float_power
            new=Transform(args=[self,other],transformation=t)
        elif isinstance(other,Transform):
            new=other.__rpow__(self)
        return new
        # return self.binary_operator(left=self, op=lambda E: numpy.float_power(E, power))

    def __rpow__(self, other):
        if isinstance(other, numbers.Number):
            t=lambda v: numpy.float_power(other,v)
            new=Transform(args=[self],transformation=t)
        elif isinstance(other,Variable):
            t=numpy.float_power
            new=Transform(args=[other,self],transformation=t)
        elif isinstance(other,Transform):
            new=other.__pow__(self)
        return new
        #return new.binary_operator(left=new,right=other, op=lambda l, r: numpy.float_power(r, l))

    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            t=lambda v: numpy.multiply(other,v)
            new=Transform(args=[self],transformation=t)
        elif isinstance(other,Variable):
            t=numpy.multiply
            new=Transform(args=[other,self],transformation=t)
        elif isinstance(other,Transform):
            new=other.__mul__(self)
        return new

    def __radd__(self, other):
        if isinstance(other, numbers.Number):
            t=lambda v: numpy.add(other,v)
            new=Transform(args=[self],transformation=t)
        elif isinstance(other,Variable):
            t=numpy.add
            new=Transform(args=[other,self],transformation=t)
        elif isinstance(other,Transform):
            new=other.__add__(self)
        return new

    def __rtruediv__(self, other):
        if isinstance(other, numbers.Number):
            t=lambda v: numpy.true_divide(other,v)
            new=Transform(args=[self],transformation=t)
        elif isinstance(other,Variable):
            t=numpy.true_divide
            new=Transform(args=[other,self],transformation=t)
        elif isinstance(other,Transform):
            new=other.__truediv__(self)
        return new
        #return new.binary_operator(left=new, right=other, op=lambda l, r: numpy.true_divide(r, l))


    def __invert__(self):
        new=Transform(args=[self])
        return new**-1

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
        #TODO remove
        return float(self.value)

class Transform:
    def __init__(self,args,transformation=None):
        self.args=args
        self.transformation=transformation

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
        newer_a=[i for i in new_a]
        return float(self.transformation(*newer_a).real)

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

    @classmethod
    def unary_operator(cls, left, op):

        return Transform(args=left.args,
                         transformation=lambda *args: op(left.transformation(*args)))

    @classmethod

    def binary_operator(cls, left, right, op):
        '''
        this function, usually called by the convenience magic-methods of Variable and Transform objects, constructs a new Objective
        whose Transformation  is the JoinedTransformation of the lower arguments and transformations
        of the left and right objects, alongside op (if they are or can be rendered as objectives). In case one of left or right
        is a number, calls unary_operator instead.
        :param left: the left hand argument to op
        :param right: the right hand argument to op.
        :param op: an operation; a function object.
        :return: a Transform whose Transformation  is the JoinedTransformation of the lower arguments and transformations
        of the left and right objects, alongside op. In case one of left or right is a number, calls unary_operator instead.

        '''
        r=None
        l=None
        if isinstance(left, Variable):
            l=Transform([left])
        if isinstance(left,Transform):
            l=left
        if isinstance(right, Variable):
            r=Transform([right])
        if isinstance(right, Transform):
            r = right

        if isinstance(right, numbers.Number):
            if isinstance(left, Variable) or isinstance(left,Transform):
                return cls.unary_operator(left=l, op=lambda E: op(E, right))
            else:
                raise TequilaException('BinaryOperator method called on types ' + str(type(left)) + ',' +str(type(right)))
        elif isinstance(left, numbers.Number):
            if isinstance(right, Variable) or isinstance(right,Transform):
                return cls.unary_operator(left=r, op=lambda E: op(left,E))
            else:
                raise TequilaException('BinaryOperator method called on types ' + str(type(left)) + ',' +str(type(right)))
        else:
            split_at = len(l.args)
            return Transform(args=l.args + r.args,
                         transformation=JoinedTransformation(left=l.transformation, right=r.transformation,
                                                             split=split_at, op=op))
    def __call__(self):
        return self.eval

    def __eq__(self, other):
        if hasattr(other, 'eval'):
            if hasattr(other, 'variables'):
                if self.eval == other.eval and self.variables == other.variables:
                    return True
        return False

    def __mul__(self, other):
        return self.binary_operator(left=self, right=other, op=numpy.multiply)

    def __add__(self, other):
        return self.binary_operator(left=self, right=other, op=numpy.add)

    def __sub__(self, other):
        return self.binary_operator(left=self, right=other, op=numpy.subtract)

    def __truediv__(self, other):
        return self.binary_operator(left=self, right=other, op=numpy.true_divide)

    def __neg__(self):
        return self.unary_operator(left=self, op=numpy.negative)

    def __pow__(self, power):
        return self.binary_operator(left=self, right=power, op=numpy.float_power)

    def __rpow__(self, other):
        #return self.binary_operator(left=self, right=other, op=lambda l, r: numpy.float_power(r, l))
        return self.binary_operator(left=other, right=self, op=numpy.float_power)

    def __rmul__(self, other):
        return self.binary_operator(left=other,right=self, op=numpy.multiply)

    def __radd__(self, other):
        return self.binary_operator(left=other, right=self, op=numpy.add)

    def __rtruediv__(self, other):
        return self.binary_operator(left=other, right=self, op=numpy.true_divide)

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
        funcpart = str(self.transformation) + ' acting on: '
        argpart = '('
        for i in range(len(self.args)):
            argpart += str(self.args[i])
            if i < len(self.args) - 1:
                argpart += ', '
        argpart += ')'
        val = str(self())
        return funcpart + argpart + ', val=' + val

    def __str__(self):
        funcpart = str(self.transformation) + ' acting on: '
        argpart = '('
        for i in range(len(self.args)):
            argpart += str(self.args[i])
            if i < len(self.args) - 1:
                argpart += ', '
        argpart += ')'
        val = str(self())
        return funcpart + argpart + ', val=' + val

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
