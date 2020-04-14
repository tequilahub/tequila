import typing, copy, numbers

from tequila import TequilaException
from tequila.utils import JoinedTransformation, to_float
from tequila.hamiltonian import paulis
from tequila.autograd_imports import numpy

import collections

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


    def __init__(self, U=None, H=None, contraction=None, shape=None):
        self._unitary = copy.deepcopy(U)
        if hasattr(H, "paulistrings"):
            self._hamiltonian = tuple([copy.deepcopy(H)])
        else:
            self._hamiltonian = tuple(H)
        self._contraction = contraction
        self._shape = shape

    def __call__(self, *args, **kwargs):
        raise TequilaException("Tried to call uncompiled ExpectationValueImpl, compile your objective before calling with tq.compile(objective) or evaluate with tq.simulate(objective)")

    def info(self, short=True, *args, **kwargs):
        if short:
            print("Expectation Value with {qubits} active qubits and {paulis} paulistrings".format(
                qubits=len(self.U.qubits), paulis=len(self.H)))
        else:
            print("Hamiltonian:\n", str(self.H))
            print("\n", str(self.U))


class Objective:
    """
    the class which represents mathematical manipulation of ExpectationValue and Variable objects. Capable of being simulated,
    and differentiated with respect to the Variables of its Expectationvalues or the Variables themselves
    :param args: an iterable of ExpectationValue's.
    :param transformation: a callable whose positional arguments (potentially, by nesting in a JoinedTransformation)
        are args
    :param simulator: a tequila simulator object. If provided, Objective is callable.

    """

    def __init__(self, args: typing.Iterable, transformation: typing.Callable = None):
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
            if hasattr(arg,'extract_variables'):
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

    def __rpow__(self, other):
        return self.right_helper(numpy.float_power, other)

    def __rmul__(self, other):
        return self.right_helper(numpy.multiply, other)

    def __radd__(self, other):
        return self.right_helper(numpy.add, other)

    def __rsub__(self, other):
        return self.right_helper(numpy.subtract, other)

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

    def apply(self, op):
        # same as wrap, might be more intuitive for some
        return self.wrap(op=op)

    def count_expectationvalues(self):
        i = 0
        for arg in self.args:
            if hasattr(arg, "U"):
                i += 1
        return i

    def __repr__(self):
        variables = self.extract_variables()
        ev = []
        argstring = ""
        i = 0
        for arg in self.args:
            if hasattr(arg, "U"):
                ev.append(i)
                argstring += "E_" + str(i) + ", "
                i += 1
            elif hasattr(arg, "name"):
                argstring += str(arg) + ", "
            else:
                assert not arg.has_expectationvalues()
                argstring += "g({}), ".format(arg.extract_variables())
        return "Objective with {} expectation values\n" \
               "Objective = f({})\n" \
               "variables = {}".format(len(ev), argstring.strip().rstrip(','), variables)

    def __call__(self, variables = None, *args, **kwargs):
        return self.transformation(*[Ei(variables=variables, *args, **kwargs) for Ei in self.args])


def ExpectationValue(U, H, *args, **kwargs) -> Objective:
    """
    Initialize an Objective which is just a single expectationvalue
    """
    return Objective.ExpectationValue(U=U, H=H, *args, **kwargs)


class TequilaVariableException(TequilaException):
    def __str__(self):
        return "Error in tequila variable:" + self.message


class Variable:

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

    def __ne__(self, other):
        if self.__eq__(other):
            return False
        else:
            return True

    def apply(self, other):
        assert (callable(other))
        return Objective(args=[self], transformation=other)

    def wrap(self,other):
        return self.apply(other)

    def __repr__(self):
        return str(self.name)


class FixedVariable(float):

    def __call__(self, *args, **kwargs):
        return self


def format_variable_list(variables: typing.List[typing.Hashable]) -> typing.List[Variable]:
    """
    Convenience functions to assign tequila variables
    :param variables: a list with Hashables as keys
    :return: a list with tq.Variable types as keys
    """
    if variables is None:
        return variables
    else:
        return [assign_variable(k) for k in variables]


def format_variable_dictionary(variables: typing.Dict[typing.Hashable, typing.Any]) -> typing.Dict[
    Variable, typing.Any]:
    """
    Convenience functions to assign tequila variables
    :param variables: a dictionary with Hashables as keys
    :return: a dictionary with tq.Variable types as keys
    """
    if variables is None:
        return variables
    else:
        return Variables(variables)


def assign_variable(variable: typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable]) -> typing.Union[
    Variable, FixedVariable]:
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
    elif isinstance(variable, typing.Hashable):
        return Variable(name=variable)
    else:
        raise TequilaVariableException(
            "Only hashable types can be assigned to Variables. You passed down " + str(variable) + " type=" + str(
                type(variable)))


class Variables(collections.abc.MutableMapping):
    """
    Dictionary for tequila variables
    Allows hashable types and variable types as keys
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
        for k,v in self.items():
            result += "{} : {}\n".format(str(k), str(v))
        return result

    def __repr__(self):
        return self.__str__()





