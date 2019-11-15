from openvqe import OpenVQEException
from functools import total_ordering
from openvqe import copy
from openvqe import numbers
from inspect import signature
from openvqe import numpy as np

class SympyVariable:

    def __init__(self, name=None, value=None):
        self._value=value
        self._name=name

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
    def variables(self):
        return {self.name:self.value}
    
    @property
    def parameter_list(self):
        return [self]
    
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value=float(value)

    @property
    def name(self):
        return self._name
    


    def __init__(self,name=None,value=None):
        if value is not None:
            if type(value) in [float,int]:
                self._value = float(value)
            else:
                self._value = np.real(value)
        else:
            self._value = None

        if type(name) is str:
            self._name = name

        else:
            if type(name) is not None:
                self._name= str(name)
            else:
                self.name='None'

        if self._name is 'None':
            self.is_default=True
        else:
            self.is_default=False
        if self._value is None:
            self.needs_init=True
        else:
            self.needs_init=False

    def has_var(self,x):
        if type(x) is Variable:
            return self == x
        elif type(x) is str:
            return self._name == x
        elif type(x) is dict:
            return self._name in x.keys()
        else:
            raise TypeError('Unsupported type')

    def update(self,x):
        if type(x) is dict:
            for k in x.keys():
                if self.name ==k:
                    self._value=x[k]
        elif type(x) is Variable:
            if x.name == self.name:
                self._value=x.value
        else:
            self._value=float(x)

    def __eq__(self,other):
        if type(self)==type(other):
            if self.name ==other.name and self.value==other.value:
                    return True
        return False

    def __add__(self, other: float):
        return Transform(Add,[self,other])

    def __radd__(self, other: float):
        if other == 0:
            return self
        else:
            return Transform(Add,[other,self])

    def __sub__(self, other):
        return Transform(Sub,[self,other])

    def __rsub__(self,other):
            return Transform(Sub,[other,self])

    def __mul__(self, other):
         return Transform(Mul,[self,other])

    def __rmul__(self,other):

        return Transform(Sub,[other,self])

    def __neg__(self):
        return Transform(Mul,[self,-1])


    def __div__(self, other):
        return Transform(Div,[self,other])

    def __rdiv__(self, other):
        return Transform(Div,[other,self])

    def __truediv__(self, other):
        return Transform(Div,[self,other])


    def __pow__(self, other):
        return Transform(Pow,[self,other])

    def __rpow__(self,other):
        return Transform(Pow,[other,self])

    def __iadd__(self,other):
        self._value+=other
        return self

    def __isub__(self,other):
        self._value -= other
        return self

    def __imul__(self,other):
        self._value *= other
        return self


    def __idiv__(self,other):
        self._value /= other
        return self

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

    def __float__(self):
        return float(self.value)

class Transform():

    @property
    def parameter_list(self):
        vl=[]
        for obj in self.args:
            if type(obj) is Variable:
                if obj not in vl:
                    vl.append(obj)
            elif type(obj) is Transform:
                for v in obj.variables:
                    if v not in vl:
                        vl.append(v)
            else:
                pass
        return vl


    @property
    def variables(self):
        vl={}
        for obj in self.args:
            if type(obj) is Variable:
                if obj.name not in vl.keys():
                    vl[obj.name]=obj.value
                else:
                    if not np.isclose(vl[obj.name],obj.value):
                        raise OpenVQEException('found two variables with the same name and different values, this is unacceptable')
            elif type(obj) is Transform:
                for k,v in obj.variables.items():
                    if k not in vl.keys():
                        vl[k]=v
                    else:
                        if not np.isclose(vl[k],v):
                            raise OpenVQEException('found two variables with the same name and different values, this is unacceptable')
            else:
                pass
        return vl
    
    @property
    def eval(self):
        new_a=[]
        for arg in self.args:
            if hasattr(arg,'__call__'):
                new_a.append(arg())
            else:
                new_a.append(arg)

        return self.f(*new_a)


    
    
    def __init__(self,func,args):
        assert callable(func)
        assert len(args) == len(signature(func).parameters)
        self.args=args
        self.f=func




    def update(self,pars):
        for arg in self.args:
            if type(pars) is dict:
                for k,v in pars.items():
                    if hasattr(arg,'update'):
                        if k in arg.variables.keys():
                            arg.update({k:v})
            elif type(pars) is list:
                if hasattr(arg,'has_var'):
                    for par in pars:
                        if arg.has_var(par):
                            arg.update(par)


    def has_var(self,x):
        for k,v in self.variables.items():
            if type(x) is dict:
                if k in x.keys():
                    return True
            if type(x) is Variable:
                if k == x.name:
                    return True
        return False

    def __call__(self):
        return self.eval

    def __eq__(self, other):
        if hasattr(other,'eval'):
            if hasattr(other,'variables'):
                if self.eval==other.eval and self.variables==other.variables:
                    ### is this safe?
                    return True
        return False


    def __add__(self, other: float):
        return Transform(Add,[self,other])

    def __radd__(self, other: float):
        if other == 0:
            return self
        else:
            return Transform(Add,[other,self])

    def __sub__(self, other):
        return Transform(Sub,[self,other])

    def __rsub__(self,other):
            return Transform(Sub,[other,self])

    def __mul__(self, other):
        # return self._return*other
         return Transform(Mul,[self,other])

    def __rmul__(self,other):

        return Transform(Sub,[other,self])

    def __neg__(self):
        return Transform(Mul,[self,-1])


    def __div__(self, other):
        return Transform(Div,[self,other])

    def __rdiv__(self, other):
        return Transform(Div,[other,self])

    def __truediv__(self, other):
        return Transform(Div,[self,other])

    def __rtruediv__(self,other):
        return Transform(Div,[other,self])



    def __pow__(self, other):
        return Transform(Pow,[self,other])

    def __rpow__(self,other):
        return Transform(Pow,[other,self])

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
        return float(self.eval)


def has_variable(obj,var):
    if hasattr(obj,'has_var'):
        return obj.has_var(var)
    else:
        return False

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
    return lv/rv

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

def Sqr(arg):
    if type(arg) in [Variable,Transform]:
        return np.sqrt(arg())
    else:
        return np.sqrt(arg)