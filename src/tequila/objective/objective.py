import typing, copy, numbers
from jax import numpy as numpy
from tequila import paulis,TequilaException
from tequila.utils import JoinedTransformation

"""
Preliminary structure to carry information over to backends
Needs to be restructured and clarified but currently does the job
"""


class ExpectationValue:
    '''
    the implementation of Expectation Values as a class. Capable of being simulated, and differentiated.
    common arithmetical operations like addition, multiplication, etc. are defined, to return Objective objects.
    :param U: a QCircuit, for preparing a state
    :param H: a Hamiltonian, whose expectation value with the state prepared by U is to be determined.
    '''
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

    def __mul__(self, other):
        new=Objective([self])
        return new.binary_operator(left=new, right=other, op=numpy.multiply)

    def __add__(self, other):
        new=Objective([self])
        return new.binary_operator(left=new, right=other, op=numpy.add)

    def __sub__(self, other):
        new=Objective([self])
        return new.binary_operator(left=new, right=other, op=numpy.subtract)

    def __truediv__(self, other):
        new=Objective([self])
        return new.binary_operator(left=new, right=other, op=numpy.true_divide)

    def __neg__(self):
        new=Objective([self])
        return new.unary_operator(left=new, op=numpy.negative)

    def __pow__(self, power):
        new=Objective([self])
        #return new.unary_operator(left=new, op=lambda E: numpy.float_power(E, power))
        return new.binary_operator(left=new, right=power, op=lambda l, r: numpy.float_power(l, r))

    def __rpow__(self, other):
        new=Objective([self])
        #return new.unary_operator(left=new, op=lambda E: other ** E)
        return new.binary_operator(left=new, right=other, op=lambda l, r: numpy.float_power(r,l))

    def __rmul__(self, other):
        new=Objective([self])
        return new.unary_operator(left=new, op=lambda E: numpy.multiply(other, E))

    def __radd__(self, other):
        new=Objective([self])
        return new.unary_operator(left=new, op=lambda E: numpy.add(other, E))

    def __rtruediv__(self, other):
        new=Objective([self])
        return new.binary_operator(left=new, right=other,op=lambda l,r: numpy.true_divide(r, l))

    def __invert__(self):
        new=Objective([self])
        return new.unary_operator(left=new, op=lambda E: numpy.power(E, -1))


class Objective:
    '''
    the class which represents mathematical manipulation of ExpectationValue objects. Capable of being simulated,
    and differentiated with respect to the Variables of its Expectationvalues.
    :param expectationvalues: an iterable of ExpectationValue's.
    :param transformation: a callable whose positional arguments (potentially, by nesting in a JoinedTransformation)
    are the expectationvalues, in order.
    '''
    def extract_variables(self):
        '''
        :return: a dictionary, containing every Variable from every ExpectationValue in the objective.
        '''
        variables = dict()
        for E in self._expectationvalues:
            variables = {**variables, **E.extract_variables()}
        return variables

    def update_variables(self, variables):
        '''
        :param variables: a list of Variables or dictionary of str, number pairs with which ALL expectationvalues of the
        Objective are to be updated. Calls the update_variables method of ExpectationValue,
        which in turn calls that of QCircuit, which ultimately accesses the update methods of
        Transform and Variable's themselves.
        :return: self, for ease of use
        '''
        for E in self._expectationvalues:
            E.update_variables(variables=variables)
        return self

    def __init__(self, expectationvalues: typing.Iterable[ExpectationValue], transformation: typing.Callable = None):
        self._expectationvalues = tuple(expectationvalues)
        self._transformation = transformation

    def is_expectationvalue(self):
        '''
        :return: bool: whether or not this objective is just a wrapped ExpectationValue
        '''
        return len(self.expectationvalues) == 1 and self._transformation is None

    @classmethod
    def ExpectationValue(cls, U=None, H=None):
        """
        Initialize a wrapped expectationvalue directly as Objective
        """
        E = ExpectationValue(H=H, U=U)
        return Objective(expectationvalues=[E])

    @property
    def transformation(self) -> typing.Callable:
        if self._transformation is None:
            return numpy.sum
        else:
            return self._transformation

    @property
    def expectationvalues(self) -> typing.Tuple:
        '''
        :return: self._expectationvalues
        '''
        if self._expectationvalues is None:
            return tuple()
        else:
            return self._expectationvalues

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
        return self.binary_operator(left=self, right=power, op=lambda l, r: numpy.float_power(l, r))
        #return self.binary_operator(left=self, op=lambda E: numpy.float_power(E, power))

    def __rpow__(self, other):
        return self.binary_operator(left=self, right=other,op=lambda l,r: numpy.float_power(r,l))

    def __rmul__(self, other):
        return self.unary_operator(left=self, op=lambda E: numpy.multiply(other, E))

    def __radd__(self, other):
        return self.unary_operator(left=self, op=lambda E: numpy.add(other, E))

    def __rtruediv__(self, other):
        return self.binary_operator(left=self, right=other,op=lambda l,r: numpy.true_divide(r, l))

    def __invert__(self):
        return self.unary_operator(left=self, op=lambda E: numpy.power(E, -1))

    @classmethod
    def unary_operator(cls, left, op):

        return Objective(expectationvalues=left.expectationvalues,
                         transformation=lambda *args: op(left.transformation(*args)))

    @classmethod
    def binary_operator(cls, left, right, op):
        '''
        this function, usually called by the convenience magic-methods of Observable and ExpectationValue objects, constructs a new Objective
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
        r=None
        l=None
        if isinstance(left, ExpectationValue):
            l = Objective([left])
        if isinstance(left,Objective):
            l=left
        if isinstance(right, ExpectationValue):
            r = Objective(expectationvalues=[right])
        if isinstance(right, Objective):
            r = right

        if isinstance(right, numbers.Number):
            if isinstance(l,Objective) or isinstance(left,Objective):
                return cls.unary_operator(left=left, op=lambda E: op(E, right))
            else:
                raise TequilaException('BinaryOperator method called on types ' + str(type(left)) + ',' +str(type(right)))
        elif isinstance(left, numbers.Number):
            if isinstance(r,Objective):
                return cls.unary_operator(left=r, op=lambda E: op(left,E))
            else:
                raise TequilaException('BinaryOperator method called on types ' + str(type(left)) + ',' +str(type(right)))
        else:
            split_at = len(l.expectationvalues)
            return Objective(expectationvalues=l.expectationvalues + r.expectationvalues,
                         transformation=JoinedTransformation(left=l.transformation, right=r.transformation,
                                                              split=split_at, op=op))
    def __repr__(self):
        return "Objective with " + str(len(self.expectationvalues)) + " expectationvalues"
