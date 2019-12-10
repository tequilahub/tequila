import typing, copy, numbers
from jax import numpy as numpy
from tequila import paulis

"""
Preliminary structure to carry information over to backends
Needs to be restructured and clarified but currently does the job
"""


class ExpectationValue:

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


class JoinedTransformation:

    def __init__(self, left, right, split, op):
        self.split = split
        self.left = left
        self.right = right
        self.op = op

    def __call__(self, *args, **kwargs):
        E_left = args[:self.split]
        E_right = args[self.split:]
        return self.op(self.left(*E_left, **kwargs), self.right(*E_right, **kwargs))


class Objective:

    def extract_variables(self):
        variables = dict()
        for E in self._expectationvalues:
            variables = {**variables, **E.extract_variables()}
        return variables

    def update_variables(self, variables=dict):
        for E in self._expectationvalues:
            E.update_variables(variables=variables)
        return self

    def __init__(self, expectationvalues: typing.Iterable[ExpectationValue], transformation: typing.Callable = None):
        self._expectationvalues = tuple(expectationvalues)
        self._transformation = transformation

    @classmethod
    def ExpectationValue(cls, H=None, U=None):
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
        return self.unary_operator(left=self, op=lambda E: numpy.float_power(E, power))

    def __rpow__(self, other):
        return self.unary_operator(left=self, op=lambda E: other ** E)

    def __rmul__(self, other):
        return self.unary_operator(left=self, op=lambda E: numpy.multiply(other, E))

    def __radd__(self, other):
        return self.unary_operator(left=self, op=lambda E: numpy.add(other, E))

    def __rtruediv__(self, other):
        return self.unary_operator(left=self, op=lambda E: numpy.true_divide(other, E))

    def __invert__(self):
        return self.unary_operator(left=self, op=lambda E: numpy.power(E, -1))

    @classmethod
    def unary_operator(cls, left, op):
        return Objective(expectationvalues=left.expectationvalues,
                         transformation=lambda *args: op(left.transformation(*args)))

    @classmethod
    def binary_operator(cls, left, right, op):
        if isinstance(right, numbers.Number):
            return cls.unary_operator(left=left, op=lambda E: op(E, right))
        else:
            split_at = len(left.expectationvalues)
            return Objective(expectationvalues=left.expectationvalues + right.expectationvalues,
                             transformation=JoinedTransformation(left=left.transformation, right=right.transformation,
                                                                 split=split_at, op=op))

    def __repr__(self):
        return "Objective with " + str(len(self.expectationvalues)) + " expectationvalues"
