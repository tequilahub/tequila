import typing, copy, numbers
from jax import numpy as numpy
import numpy as np
from tequila import TequilaException
from tequila.utils import JoinedTransformation
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
        result = dict()
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

    def has_var(self, x):
        '''
        :param x: dict, Variable, or str
        checks if (any of the ) variable(s) passed are present within Objective. Looks for them by name, NOT value.
        return: bool: true if a match found else false.
        '''
        for k, v in self.extract_variables().items():
            if type(x) is dict:
                if k in x.keys():
                    return True
            if hasattr(x,'name') and hasattr(x,'value'):
                if k == x.name:
                    return True
            if type(x) is str:
                if k == x:
                    return True

        return False
    def extract_variables(self):
        """
        :return: a dictionary, containing every variable from every ExpectationValue in the objective and every Variable.
        """
        variables = dict()
        for E in self.args:
            variables = {**variables, **E.extract_variables()}
        return variables

    def update_variables(self, variables):
        '''
        :param variables: a list of Variables or dictionary of str, number pairs with which ALL expectationvalues and variables of the
        Objective are to be updated. Calls the update_variables method of ExpectationValue (and Variable),
        The former of which in turn calls that of QCircuit, which ultimately accesses the update methods of
         Variable's themselves.
        :return: self, for ease of use
        '''
        for E in self.args:
            E.update_variables(variables=variables)
        return self

    def __init__(self, args: typing.Iterable, transformation: typing.Callable = None,loaded=None):
        self._args = tuple(args)
        self._transformation = transformation
        self.loaded=loaded
        self.last=None

    def load(self,simulator):
        '''
        attach a simulator to the Objective to render it callable
        :param simulator: a Tequila simulator object
        :return: self, for ease of use
        '''
        self.loaded=simulator
        return self
    def is_expectationvalue(self):
        """
        :return: bool: whether or not this objective is just a wrapped ExpectationValue
        """
        return len(self.args) == 1 and self._transformation is None and type(self.args[0]) is ExpectationValueImpl

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

    def left_helper(self,op,other):
        '''
        function for use by magic methods, which all have an identical structure, differing only by the
        external operator they call. left helper is responsible for all 'self # other' operations
        :param op: the operation to be performed
        :param other: the right-hand argument of the operation to be performed
        :return: an Objective, who transform is the joined_transform of self with op, acting on self and other
        '''
        if isinstance(other, numbers.Number):
            t = lambda v: op(v,other)
            new = self.unary_operator(left=self,op=t)
        elif hasattr(other,'name'):
            t = op
            nother = Objective(args=[other])
            new = self.binary_operator(left=self,right=nother,op=t)
        elif isinstance(other, Objective):
            new=self.binary_operator(left=self,right=other,op=op)
        elif isinstance(other,ExpectationValueImpl):
            new=self.binary_operator(left=self,right=Objective(args=[other]),op=op)
        return new

    def right_helper(self,op,other):
        '''
        see the doc of left_helper above for explanation
        '''
        if isinstance(other, numbers.Number):
            t = lambda v: op(other,v)
            new = self.unary_operator(left=self,op=t)
        elif hasattr(other,'name'):
            t = op
            nother = Objective(args=[other])
            new = self.binary_operator(left=nother,right=self,op=t)
        elif isinstance(other, Objective):
            new=self.binary_operator(left=other,right=self,op=op)
        elif isinstance(other,ExpectationValueImpl):
            new=self.binary_operator(left=Objective(args=[other]),right=self,op=op)
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
        return self.unary_operator(left=self, op=lambda v: numpy.multiply(v, -1))

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

    def wrap(self,op):
        '''
        convenience function for doing unary_operator with non-arithmetical operations like sin, cosine, etc.
        :param op: an operation to perform on the output of self
        :return: an objective
        '''
        return self.unary_operator(self,op)
    def __repr__(self):
        string="Objective with " + str(len(self.args)) + " arguments"
        if self.last is not None:
            string+=" , last call value = "+str(self.last)
        return string

    def __call__(self,samples=None):
        '''
        Evaluates the expression which Objective represents, if possible.
        :param samples:
        :return:
        '''
        if all([hasattr(arg,'name')==True for arg in self.args]):
            back=self.transformation(*[arg() for arg in self.args])
            return float(back)
        if self.loaded is None:
            raise TequilaException('Objective cannot be called when Expectation Values are present if no simulator is attached!')
        else:
            if samples is None:
                back=self.loaded.simulate_objective(objective=self)
            else:
                back= self.loaded.measure_objective(objective=self,samples=samples)
        try:
            self.last=float(back)
            return float(back)
        except:
            self.last=back
            return back
def ExpectationValue(U, H) -> Objective:
    """
    Initialize an Objective which is just a single expectationvalue
    """
    return Objective.ExpectationValue(U=U, H=H)