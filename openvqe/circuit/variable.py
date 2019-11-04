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

class Variable():
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value=value

    @property
    def name(self):
        return self._name
    
    def __init__(self, value=None, name: str = ''):
        self._value = value
        self._name = name

    def __eq__(self,other):
        if id(self) != id(other):
            return False
        return True

    def __add__(self, other: float):
        return Transform(self,Add,other)

    def __radd__(self, other: float):
        if other == 0:
            return self
        else:
            return Transform(self,Add,other)

    def __iadd__(self,other):
        self._value+=other
        return self

    def __sub__(self, other):
        return Transform(self,Sub,other)

    def __rsub__(self,other):
        if other == 0:
            return -self
        else:
            first=-self
            return Transform(first,Add,other)

    def __isub__(self,other):
        self._value -= other
        return self

    def __mul__(self, other):
         return Transform(self,Mul,other)

    def __rmul__(self,other):
        return Transform(self,Mul,other)

    def __imul__(self,other):
        self._value *= other
        return self


    def __neg__(self):
        return Transform(self,Mul,-1)


    def __div__(self, other):
        return Transform(self,Div,other)

    def __rdiv__(self, other):
        first=Transform(self,Inverse,None)
        return Transform(first,Mul,other)

    def __truediv__(self, other):
        return Transform(self,Div,other)

    def __idiv__(self,other):
        self._value /= other
        return self

    def __pow__(self, other):
        return Transform(self,Pow,other)

    def __rpow__(self,other):
        return Transform(other,Pow,self)

    def __ipow__(self,other):
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
        if self.__eq__(self, other):
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

class Transform():

    @property
    def variables(self):
        vl=[]
        if hasattr(self,'l'):
            if type(self.l) is Variable:
                vl.append(self.l)
            elif type(self.l) is Transform:
                vl.extend(self.l.variables)
            else:
                pass
        if hasattr(self,'r'):
            if type(self.r) is Variable:
                vl.append(self.r)
            elif type(self.r) is Transform:
                vl.extend(self.r.variables)
            else:
                pass
        if len(vl) == 0:
            print('warning: found no variables in this Transform')
        return vl

    @property
    def eval(self):
        if self.l is not None:
            try:
                lv=self.l()
            except:
                lv=self.l
        if self.r is not None:
            try:
                rv=self.r()
            except:
                rv=self.r

            return self.f(lv,rv)
        else:
            return self.f(lv)
    
    

    def __init__(self,left,func,right=None):
        self.l=left
        self.f=func
        self.r=right

    def __call__(self):
        return self.eval

     def __eq__(self, other):

        if id(self) != id(other):
            return False

        return True


    def __add__(self, other: float):
        return Transform(self,Add,other)

    def __radd__(self, other: float):
        if other == 0:
            return self
        else:
            return Transform(self,Add,other)

    def __sub__(self, other):
        return Transform(self,Sub,other)

    def __rsub__(self,other):
        if other == 0:
            return -self
        else:
            first=-self
            return Transform(first,Add,other)

    def __mul__(self, other):
        # return self._return*other
         return Transform(self,Mul,other)

    def __rmul__(self,other):
        if other == 0:
            return 0
        else:
            return Transform(self,Mul,other)

    def __neg__(self):
        return Transform(self,Mul,-1)


    def __div__(self, other):
        return Transform(self,Div,other)

    def __rdiv__(self, other):
        if other == 0:
            return 0
        else:
            first=Transform(self,Inverse,None)
            return Transform(first,Mul,other)

    def __truediv__(self, other):
        return Transform(self,Div,other)


    def __pow__(self, other):
        return Transform(self,Pow,other)

    def __rpow__(self,other):
        return Transform(other,Pow,self)

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
        if self.__eq__(self, other):
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

    def __repr__

def Add(l,r):
    if type(l) in [Variable, Transform]:
        lv=l()
    else:
        lv=l
    if type(r) in [Variable, Transform]:
        rv=r()
    else:
        rv=r
    return lv+rv

def Sub(l,r):
    if type(l) in [Variable, Transform]:
        lv=l()
    else:
        lv=l
    if type(r) in [Variable, Transform]:
        rv=r()
    else:
        rv=r
    return lv -rv

def Mul(l,r):
    if type(l) in [Variable, Transform]:
        lv=l()
    else:
        lv=l
    if type(r) in [Variable, Transform]:
        rv=r()
    else:
        rv=r
    return lv*rv

def Div(l,r):
    if type(l) in [Variable, Transform]:
        lv=l()
    else:
        lv=l
    if type(r) in [Variable, Transform]:
        rv=r()
    else:
        rv=r
    return lv-rv

def Inverse(l):
    if type(l) in [Variable, Transform]:
        lv=l()
    else:
        lv=l

    return 1.0/l

def Pow(l,r):
    if type(l) in [Variable, Transform]:
        lv=l()
    else:
        lv=l
    if type(r) in [Variable, Transform]:
        rv=r()
    else:
        rv=r
    return l**r