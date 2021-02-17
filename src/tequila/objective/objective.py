import typing, copy, numbers

from tequila import TequilaException
from tequila.utils import JoinedTransformation, to_float
from tequila.tools.convenience import list_assignment
from tequila.hamiltonian import paulis
from tequila.grouping.binary_rep import BinaryHamiltonian
import numpy as onp
from tequila.autograd_imports import numpy as numpy


import collections


class ExpectationValueImpl:
    """
    Implements the (uncompiled) Expectation Value as a class. Should not be called directly.

    Notes
    -----
    Though not obscured from the user, users should not initialize this class directly, since it lacks convenience
    functions for arithmetics. Instead, these are handled by the VectorObjective class; initializing an VectorObjective
    of a single Expectation Value is done with VectorObjective.ExpectationValue or with tq.ExpectationValue.

    See Also
    --------
    VectorObjective
    ExpectationValue

    Attributes
    ----------
    U:
        a QCircuit, for preparing a state
    H:
        a Hamiltonian, whose expectation value with the state prepared by U is to be determined.

    Methods
    -------
    extract_variables:
        wrapper over extract_variables for QCircuit.
    update_variables:
        wrapper over update_variables for QCircuit.
    info:
        return information about the ExpectationValue.
    """

    @property
    def U(self):
        if self._unitary is None:
            return None
        else:
            return self._unitary

    @property
    def H(self) -> list:
        if self._hamiltonian is None:
            return paulis.QubitHamiltonian.unit()
        else:
            return self._hamiltonian

    def count_measurements(self):
        return sum([H.count_measurements() for H in self.H])

    def extract_variables(self) -> typing.Dict[str, numbers.Real]:
        """
        Wrapper over identically named function in QCircuit. Returns all variables from the underlying unitary.
        Returns
        -------

        """
        result = []
        if self.U is not None:
            result = self.U.extract_variables()
        return result

    def replace_variables(self, replacement):
        if self.U is not None:
            self.U.replace_variables(replacement)

    def __init__(self, U=None, H=None, contraction=None, shape=None, *args, **kwargs):
        """

        Parameters
        ----------
        U: QCircuit, Optional:
            the unitary for state preparation in the expectation value.
        H: optional:
            the hamiltonian
        contraction:
            callable that should transform the output of calling the expectation value.
        shape: optional:
            the shape of the return value for the object
        """
        self._unitary = copy.deepcopy(U)
        if hasattr(H, "paulistrings"):
            self._hamiltonian = tuple([copy.deepcopy(H)])
        else:
            self._hamiltonian = tuple(H)
        self._contraction = contraction
        self._shape = shape

    def map_qubits(self, qubit_map: dict):
        """

        Maps the qubit within the underlying Hamiltonians and Unitaries

        Parameters
        ----------
        qubit_map
            a dictionary which maps old to new qubits

        Returns
        -------
        the ExpectationValueImpl structure with mapped qubits

        """
        return ExpectationValueImpl(H=tuple([H.map_qubits(qubit_map=qubit_map) for H in self.H]), U=self.U.map_qubits(qubit_map=qubit_map), contraction=self._contraction, shape=self._shape)


    def __call__(self, *args, **kwargs):
        raise TequilaException(
            "Tried to call uncompiled ExpectationValueImpl, compile your objective before calling with tq.compile(objective) or evaluate with tq.simulate(objective)")

    def info(self, short=True, *args, **kwargs):
        if short:
            print("Expectation Value with {qubits} active qubits and {paulis} paulistrings".format(
                qubits=len(self.U.qubits), paulis=len(self.H)))
        else:
            print("Hamiltonian:\n", str(self.H))
            print("\n", str(self.U))

def identity(x):
    """
    Returns input unchanged.

    Use: in place of lambda x: x, this function is used for smarter auto-diff in grad.
    Parameters
    ----------
    x:
        anything.

    Returns
    -------
    object
        input, returned unchanged.
    """
    return x


class Objective:
    """
    the class which represents mathematical manipulation of ExpectationValue and Variable objects. The core of tequila.

    Todo: Jakob, wanna write some nice examples here?

    Attributes:

    backend: str:
        a string; the backend to which the objective has been compiled, if any. If no expectationvalues, returns 'free'.

    """

    def __init__(self, args: typing.Iterable = None, transformation: typing.Callable = None):
        if args is None:
            self._args = tuple()
            self._transformation = lambda *x: 0.0
        else:
            self._args = tuple(args)
            self._transformation = transformation

    def map_qubits(self, qubit_map: dict):
        """

        Maps qubits for all quantum circuits and hamiltonians in the objective

        Parameters
        ----------
        qubit_map
            a dictionary which maps old to new qubits
            keys and values should be integers

        Returns
        -------
        the Objective with mapped qubits

        """
        mapped_args = []
        for arg in self.args:
            if hasattr(arg, "map_qubits"):
                mapped_args.append(arg.map_qubits(qubit_map=qubit_map))
            else:
                assert not hasattr(arg, "U") # failsave
                assert not hasattr(arg, "H") # failsave
                mapped_args.append(arg) # for purely variable dependend arguments

        return Objective(args=mapped_args, transformation=self.transformation)

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

    def extract_variables(self) -> list:
        """
        Extract all variables on which the objective depends
        :return: List of all Variables
        """
        variables = []
        for arg in self.args:
            if hasattr(arg, 'extract_variables'):
                variables += arg.extract_variables()
            else:
                variables += []
        # remove duplicates without affecting ordering
        # allows better reproducibility for random initialization
        # won't work with set
        unique = []
        for i in variables:
            if i not in unique:
                unique.append(i)
        return unique

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
    def transformations(self) -> typing.Callable:
        return [self.transformation]

    @property
    def args(self) -> typing.Tuple:

        if self._args is None:
            return tuple()
        else:
            return self._args
    @property
    def argsets(self):
        return [self.args]

    def _left_helper(self, op, other):
        """
        function for use by magic methods, which all have an identical structure, differing only by the
        external operator they call.
        left helper is responsible for all 'self # other' operations; right helper, for "other # self".

        Parameters
        ----------
        op: callable:
            the operation to be performed
        other:
            the right-hand argument of the operation to be performed

        Returns
        -------
        Objective:
            an Objective, who transform is the joined_transform of self with op, acting on self and other.
        """
        if isinstance(other, numbers.Number):
            t = lambda v: op(v, other)
            new = self.unary_operator(left=self, op=t)
        elif isinstance(other, Objective):
            new = self.binary_operator(left=self, right=other, op=op)
        elif isinstance(other, VectorObjective):
            new = other.binary_operator(left=self, right=other, op=op)
        elif isinstance(other, ExpectationValueImpl):
            new = self.binary_operator(left=self, right=Objective(args=[other]), op=op)
        else:
            t = op
            nother = Objective(args=[assign_variable(other)])
            new = self.binary_operator(left=self, right=nother, op=t)
        return new

    def _right_helper(self, op, other):
        """
        see the doc of _left_helper above for explanation
        """
        if isinstance(other, numbers.Number):
            t = lambda v: op(other, v)
            new = self.unary_operator(left=self, op=t)
        elif isinstance(other, Objective):
            new = self.binary_operator(left=other, right=self, op=op)
        elif isinstance(other, VectorObjective):
            new = other.binary_operator(left=other, right=self, op=op)
        elif isinstance(other, ExpectationValueImpl):
            new = self.binary_operator(left=Objective(args=[other]), right=self, op=op)
        else:
            t = op
            nother = Objective(args=[assign_variable(other)])
            new = self.binary_operator(left=nother, right=self, op=t)
        return new

    def __mul__(self, other):
        return self._left_helper(numpy.multiply, other)

    def __add__(self, other):
        return self._left_helper(numpy.add, other)

    def __sub__(self, other):
        return self._left_helper(numpy.subtract, other)

    def __truediv__(self, other):
        return self._left_helper(numpy.true_divide, other)

    def __neg__(self):
        return self.unary_operator(left=self, op=lambda v: numpy.multiply(v, -1))

    def __pow__(self, other):
        return self._left_helper(numpy.float_power, other)

    def __rpow__(self, other):
        return self._right_helper(numpy.float_power, other)

    def __rmul__(self, other):
        return self._right_helper(numpy.multiply, other)

    def __radd__(self, other):
        return self._right_helper(numpy.add, other)

    def __rsub__(self, other):
        return self._right_helper(numpy.subtract, other)

    def __rtruediv__(self, other):
        return self._right_helper(numpy.true_divide, other)

    def __invert__(self):
        new = Objective(args=[self])
        return new ** -1

    @classmethod
    def unary_operator(cls, left, op):
        """
        Arithmetical function for unary operations.
        Generally, called by the magic methods of Objective itself.
        Parameters
        ----------
        left: Objective:
            the objective to which op will be applied

        op: Callable:
            an operation to apply to left

        Returns
        -------
        Objective:
            Objective representing op applied to objective left.

        """
        return Objective(args=left.args,
                         transformation=lambda *args: op(left.transformation(*args)))

    @classmethod
    def binary_operator(cls, left, right, op):
        """
        Core arithmetical method for creating differentiable callables of two Tequila Objectives and or Variables.

        this function, usually called by the convenience magic-methods of Observable objects, constructs a new Objective
        whose Transformation  is the JoinedTransformation of the lower arguments and transformations
        of the left and right objects, alongside op (if they are or can be rendered as objectives).
        In case one of left or right is a number, calls unary_operator instead.

        Parameters
        ----------
        left:
            the left hand argument to op
        right:
            the right hand argument to op.
        op: callable:
            an operation; a function object.

        Returns
        -------
        Objective:
            an objective whose Transformation is op acting on left and right.
        """

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
        """
        convenience function for doing unary_operator with non-arithmetical operations like sin, cosine, etc.
        Parameters
        ----------
        op: callable:
            an operation to perform on the output of self

        Returns
        -------
        Objective:
            an objective which is evaluated as op(self)
        """
        return self.unary_operator(self, op)

    def apply(self, op):
        """alias for wrap"""
        return self.wrap(op=op)

    def get_expectationvalues(self):
        """
        Returns
        -------
        list:
            all the expectation values that make up the objective.
        """
        return [arg for arg in self.args if hasattr(arg, "U")]

    def count_measurements(self):
        """
        Count all measurements necessary for this objective:
        Function will iterate to all unique expectation values and count the
        number of Pauli strings in the corresponding Hamiltonians
        Returns
        -------
        Number of measurements required for this objective
        Measurements can be on different circuits (with regards to gates, depth, size, qubits)
        """
        return sum(E.count_measurements() for E in list(set(self.get_expectationvalues())))

    def count_expectationvalues(self, unique=True):
        """
        Parameters
        ----------
        unique: bool:
            whether or not to count identical expectationvalues as distinct.

        Returns
        -------
        int:
            how many (possibly, how many unique) expectationvalues are contained within the objective.

        """
        if unique:
            return len(set(self.get_expectationvalues()))
        else:
            return len(self.get_expectationvalues())

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        variables = self.extract_variables()
        if len(variables) > 5:
            variables = len(variables)

        types = [type(E) for E in self.get_expectationvalues()]
        types = list(set(types))

        if ExpectationValueImpl in types:
            if len(types) == 1:
                types = "not compiled"
            else:
                types = "partially compiled to " + str([t for t in types if t is not ExpectationValueImpl])

        unique = self.count_expectationvalues(unique=True)
        measurements = self.count_measurements()
        return "Objective with {} unique expectation values\n" \
               "total measurements = {}\n" \
               "variables          = {}\n" \
               "types              = {}".format(unique, measurements, variables, types)

    def __call__(self, variables=None, *args, **kwargs):
        """
        Return the output of the calculation the objective represents.

        Parameters
        ----------
        variables: dict:
            dictionary instantiating all variables that may appear within the objective.
        args
        kwargs

        Returns
        -------
        float:
            the result of the calculation represented by this objective.
        """
        variables = format_variable_dictionary(variables)
        # failsafe
        check_variables = {k: k in variables for k in self.extract_variables()}
        if not all(list(check_variables.values())):
            raise TequilaException("Objective did not receive all variables:\n"
                                   "You gave\n"
                                   " {}\n"
                                   " but the objective depends on\n"
                                   " {}\n"
                                   " missing values for\n"
                                   " {}".format(variables, self.extract_variables(), [k for k,v in check_variables.items() if not v]))

        # avoid multiple evaluations
        evaluated = {}
        ev_array = []
        for E in self.args:
            if E not in evaluated:
                expval_result = E(variables=variables, *args, **kwargs)
                evaluated[E] = expval_result
            else:
                expval_result = evaluated[E]
            ev_array.append(expval_result)
        result = onp.asarray(self.transformation(*ev_array),dtype=float)
        if result.shape == ():
            return float(result)
        else:
            return result


    def contract(self):
        """
        Exists only to be convient in optimizers, which all contract over VectrObjectives.
        Returns
        -------
        Objective:
            itself.
        """
        return self

    def __len__(self):
        return  1

class VectorObjective:
    """
    fundamental class for arithmetic and transformations on quantum data; the core of tequila.

    A callable class which returns either a scalar or a vector. Implements all tools needed for arithmetic,
    as well as automatic differentiation, of user defined objectives, which may be either quantum or classical
    in nature.

    Attributes
    ----------
    args:
        list, contracting over argsets, and removing all duplicates.
    argsets:
        a list of lists of arguments to each of the functions represented by the objective.
        Elements of said lists of arguments should be ExpectationValueImpl, Variable, or FixedVariable (i.e, a number)
    backend:
        string; what simulation backend, if any, the VectorObjective has been compiled for. If VectorObjective contains NO
        quantum simulables, then the string 'free' will be returned.
    transformations:
        a list of callables; the functions represented by the objective. In the case that VectorObjective has only one
        callable -- i.e, returns a scalar when called -- then the objective represents a minimizable objective function,
        such as is appropriate to be optimized over.

    Methods
    -------
    apply:
        see 'wrap'; just an alias.
    apply_op_list:
        compose a list of operations, elementwise, with the transforms of the objective.
    apply_to:
        apply a composition  of list of operations to a set of the objectives transformations determined by a list
        of positions, elementwise. I.E: apply 'multiply by 1' and 'divide by 2' to the transformations at positions
        3 and 5.
    binary_operator:
        class method. used by magic methods to construct new objectives arithmetically.
        generally used for operations between two tequila types
    contract:
        return a 1-d objective, the sum over the vector represented by objective which calls this method.
    count_expectationvalues:
        return the number of expectationvalues in args
    count_expectationvalues_at:
        return the number of expectationvvalues in a specific argset
    empty:
        static method. return an empty objective of a given length.
    ExpectationValue:
        static method. return a 1-d objective of a single, ntrasformed expectation value.
    extract_variables:
        extract all the variables on which any position of th eobjective depends
    extract_variables_at:
        exctract all the variables on which a specific position of the objective depends
    from_list:
        static method. Initialize a new objective by 'stacking' a list thereof end to end in order.
    get_expectationvalues:
        get every expectationvalue on which the objective depends
    get_expectationvalues_at:
        get every expectationvalue on which a specific position of the objective depends
    has_expectationvalues:
        return whether or not there are any expectationvalues in args
    has_expectationvalues_at:
        return whether or not there are expectationvalues in a specific argset
    is_expectationvalue:
        return whether or not the whole objective is just a wrapper around a single expectation value.
    wrap:
        compose a callable with each and every transformation of the objective.
    unary_operator:
        class method. used by magic methods to construct new objectives arithmetically.
        generally used for operations between tequila types and other python types.

    """
    def __init__(self, argsets: typing.Iterable = None, transformations: typing.Iterable[callable] = None):
        """

        Parameters
        ----------
        argsets: iterable:
            a (list, tuple) of (list,tuple) of arguments; each set the argument of a transformation.
            In geneneral, the elements of each argset are ExpectationValueImpl, Variable, or  float.
        transformations: iterable:
            a (list, tuple) of Callable; these determine the output of call.
        """

        if argsets is None:
            self._argsets = ((),)
            self._transformations = tuple([lambda *x: 0.0])
        else:
            # make sure all things initialize rightly
            groups=[]
            for argset in argsets:
                la = list_assignment(argset)
                tup=tuple(la)
                groups.append(tup)
            tups=tuple(groups)
            self._argsets = tups
            if transformations is None:
                self._transformations = tuple([identity for i in range(len(self.argsets))])
            else:
                self._transformations = tuple(list_assignment(transformations))

            assert len(self.argsets) == len(self.transformations)

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
        Extract all variables on which the objective depends.
        Returns
        -------
        list:
            all the variables.
        """

        variables = []
        for arg in self.args:
            if hasattr(arg, 'extract_variables'):
                variables += arg.extract_variables()
            else:
                variables += []

        return list(set(variables))

    def extract_variables_at(self,pos):
        """
        Return all the variables from a given argset.
        Parameters
        ----------
        pos: int:
            which position in the list of argsets to return from.

        Returns
        -------
        list:
            a list of all the variables from a certain position in the list of argsets.
        """
        assert isinstance(pos,int)
        assert pos <= len(self) -1
        variables = []
        for arg in self.argsets[pos]:
            if hasattr(arg, 'extract_variables'):
                variables += arg.extract_variables()
            else:
                variables += []

        return list(set(variables))
    def is_expectationvalue(self):
        """
        Returns
        -------
        bool:
            whether or not this objective is just a wrapped ExpectationValue
        """
        return len(self.args) == 1 and self._transformations is None and type(self.args[0]) is ExpectationValueImpl


    def has_expectationvalues(self):
        """
        Returns
        -------
        bool:
            whether or not any element from any argset is an Expectation Value.

        """
        return any([hasattr(arg, "U") for arg in self.args])

    def has_expectationvalues_at(self,pos):
        """
        whether or not the nth argset has any expectation values in it.
        Uses (O, N-1) notation (i.e, the first position is zero, the last is len(self) -1)
        Parameters
        ----------
        pos: int:
            which position in the list of argsets to check.

        Returns
        -------
        bool:
            whether or not  the 'pos'th argset contains any expectationvalues
        """
        assert isinstance(pos,int)
        assert pos <= len(self) -1
        return any([hasattr(arg,'U') for arg in self.argsets[pos] ])

    @property
    def transformations(self) -> typing.Tuple:
        back=[]
        for t in self._transformations:
            if t is None:
                back.append(identity)
            else:
                back.append(t)
        return tuple(back)

    @property
    def args(self) -> typing.Tuple:
        all_args= []
        for argset in self.argsets:
            for arg in argset:
                all_args.append(arg)

        return tuple(list(set(all_args)))

    @property
    def argsets(self) -> typing.Tuple:

        if self._argsets is None:
            return ((),)
        else:
            return self._argsets

    def _left_helper(self, op, other):
        """
        function for use by magic methods, which all have an identical structure, differing only by the
        external operator they call.
        left helper is responsible for all 'self # other' operations; right helper, for "other # self".

        Parameters
        ----------
        op: callable:
            the operation to be performed
        other:
            the right-hand argument of the operation to be performed

        Returns
        -------
        VectorObjective:
            an VectorObjective, who transform is the joined_transform of self with op, acting on self and other.
        """
        sized=self._size_helper(len(self),other)
        if isinstance(sized, numpy.ndarray):
            ops=[lambda v: op(v, s) for s in sized]
            new = self.unary_operator(left=self, ops=ops)
        elif isinstance(sized, VectorObjective):
            new = self.binary_operator(left=self, right=other, op=op)
        else:
            t = op
            nother = VectorObjective(argsets=[[assign_variable(other)]])
            new = self.binary_operator(left=self, right=nother, op=t)
        return new

    def _right_helper(self, op, other):
        """
        see the doc of _left_helper above for explanation
        """
        sized = self._size_helper(len(self), other)
        if isinstance(sized, numpy.ndarray):
            ops = [lambda v: op(s, v) for s in sized]
            new = self.unary_operator(left=self, ops=ops)
        elif isinstance(other, VectorObjective):
            new = self.binary_operator(left=other, right=self, op=op)
        else:
            t = op
            nother = VectorObjective(argsets=[[assign_variable(other)]])
            new = self.binary_operator(left=nother, right=self, op=t)
        return new

    def __len__(self):
        return len(self.transformations)

    def __mul__(self, other):
        return self._left_helper(numpy.multiply, other)

    def __add__(self, other):
        return self._left_helper(numpy.add, other)

    def __sub__(self, other):
        return self._left_helper(numpy.subtract, other)

    def __truediv__(self, other):
        return self._left_helper(numpy.true_divide, other)

    def __neg__(self):
        return self._left_helper(numpy.multiply,-1.0)

    def __pow__(self, other):
        return self._left_helper(numpy.float_power, other)

    def __rpow__(self, other):
        return self._right_helper(numpy.float_power, other)

    def __rmul__(self, other):
        return self._right_helper(numpy.multiply, other)

    def __radd__(self, other):
        return self._right_helper(numpy.add, other)

    def __rsub__(self, other):
        return self._right_helper(numpy.subtract, other)

    def __rtruediv__(self, other):
        return self._right_helper(numpy.true_divide, other)

    def __invert__(self):
        new = VectorObjective(argsets=self.argsets, transformations=self.transformations)
        return new ** -1.0

    @staticmethod
    def _size_helper(size,other):
        ls=size
        if isinstance(other, numbers.Number):
            vectorized=[other for i in range(ls)]
            return numpy.asarray(vectorized)
        elif isinstance(other,numpy.ndarray):
            if ls == 1:
                return other.flatten()
            flat=other.flatten()
            if flat.shape[0] != ls:
                raise TequilaException('cannot combine objective of len {} with {} element array!'.format(ls,flat.shape[0]))
            else:
                return flat
        elif isinstance(other, Objective):
            # same as 1d VectorObjective
            argset = other.argsets[0]
            argsets = [argset for i in range(ls)]
            transform = other.transformations[0]
            transforms = [transform for i in range(ls)]
            return VectorObjective(argsets=argsets, transformations=transforms)
        elif isinstance(other, VectorObjective):
            if ls == 1:
                return other
            lo = len(other)
            if lo == 1:
                argset = other.argsets[0]
                argsets = [argset for i in range(ls)]
                transform = other.transformations[0]
                transforms = [transform for i in range(ls)]
                return VectorObjective(argsets=argsets, transformations=transforms)
            if ls != lo:
                raise TequilaException('cannot combine objectives of len  {} and {}!'.format(ls,len(other)))
            else:
                return other
        elif isinstance(other, ExpectationValueImpl):
            return VectorObjective(argsets=[[other] for i in range(ls)])
        elif isinstance(other, Variable):
            return VectorObjective(argsets=[[other] for i in range(ls)])
        else:
            raise TequilaException('Received unknown object of type {}'.format(type(other)))

    @classmethod
    def unary_operator(cls, left, ops):
        """
        Arithmetical function for unary operations.
        Generally, called by the magic methods of VectorObjective itself.
        Parameters
        ----------
        left: VectorObjective:
            the objective to which op will be applied

        ops: list of Callable:
            operations to apply to left

        Returns
        -------
        VectorObjective:
            VectorObjective representing ops applied to objective left.

        """
        transformations=[]
        for i,op in enumerate(ops):
            transformations.append(lambda *args: op(left.transformations[i](*args)))
        return VectorObjective(argsets=left.argsets,
                               transformations=transformations)

    @classmethod
    def binary_operator(cls, left, right, op):
        """
        Core arithmetical method for creating differentiable callables of two Tequila Objectives and or Variables.

        this function, usually called by the convenience magic-methods of Observable objects, constructs a new VectorObjective
        whose Transformation  is the JoinedTransformation of the lower arguments and transformations
        of the left and right objects, alongside op (if they are or can be rendered as objectives).
        In case one of left or right is a number, calls unary_operator instead.

        Parameters
        ----------
        left:
            the left hand argument to op
        right:
            the right hand argument to op.
        op: callable:
            an operation; a function object.

        Returns
        -------
        VectorObjective:
            op acting on left, right. Will arguments appropriately if one is scalar.
        """
        if isinstance(right, numbers.Number) or isinstance(right, numpy.ndarray):
            if isinstance(left, VectorObjective):
                sized_r=left._size_helper(len(left),right)
                sized_l=left._size_helper(len(right),left)
                ops=[lambda E: op(E, s) for s in sized_r]
                return cls.unary_operator(left=sized_l, ops=ops)
            else:
                raise TequilaException(
                    'BinaryOperator method called on types ' + str(type(left)) + ',' + str(type(right)))
        elif isinstance(left, numbers.Number) or isinstance(left, numpy.ndarray):
            if isinstance(right, VectorObjective):
                sized_r = right._size_helper(len(left), right)
                sized_l = right._size_helper(len(right), left)
                ops = [lambda E: op(s, E) for s in sized_l]
                return cls.unary_operator(left=sized_r, ops=ops)
            else:
                raise TequilaException(
                    'BinaryOperator method called on types ' + str(type(left)) + ',' + str(type(right)))
        else:
            sized_r=VectorObjective._size_helper(len(left), right)
            sized_l=VectorObjective._size_helper(len(right), left)
            sets = []
            trans = []
            for i in range(len(sized_r)):
                left_args=sized_l.argsets[i]
                left_f=sized_l.transformations[i]
                right_args=sized_r.argsets[i]
                right_f=sized_r.transformations[i]
                split_at=len(left_args)
                sets.append(left_args+right_args)
                new_tran=JoinedTransformation(left=left_f,right=right_f,split=split_at,op=op)
                trans.append(new_tran)
            return VectorObjective(argsets=sets,
                                   transformations=trans)

    def wrap(self, op):
        """
        convenience function for doing unary_operator with non-arithmetical operations like sin, cosine, etc.
        Parameters
        ----------
        op: callable:
            an operation to perform on the output of self

        Returns
        -------
        VectorObjective:
            an objective which is evaluated as op(self)
        """
        ops = [op for i in range(len(self))]
        return self.unary_operator(self, ops)

    def apply(self, op):
        """alias for wrap"""
        return self.wrap(op=op)

    def apply_to(self,op,positions):
        """
        Apply a single operation to all transformations in a list
        Parameters
        ----------
        op: callable:
            the operation to apply to transformations in specified positions
        positions: list:
            which of the transformations in self.transformation to apply op to.

        Returns
        -------
        VectorObjective:
            self, with op applied at chosen positions.
        """
        positions=list(set(list_assignment(positions)))
        assert max(positions) <= len(self)-1
        argsets=self.argsets
        transformations=self.transformations
        new_transformations=[]
        for i, t in enumerate(transformations):
            if i in positions:
                new = lambda *args: op(t(*args))
            else:
                new= t
            new_transformations.append(new)

        return VectorObjective(argsets=argsets, transformations=new_transformations)

    def apply_op_list(self,oplist):
        """
        Apply a list of operations to each output of self.
        Parameters
        ----------
        oplist:  list of Callabe:
            the list of operations to compose with the current transformations. Must have equal length.

        Returns
        -------
        VectorObjective:
            an VectorObjective, corresponding to oplist[i] composed with self.transformations[i] for all i.
        """
        assert len(oplist) == len(self)
        argsets = self.argsets
        transformations = self.transformations
        new_transformations = []
        for i, t in enumerate(transformations):
            new = lambda *args: oplist[i](t(*args))
            new_transformations.append(new)

        return VectorObjective(argsets=argsets, transformations=new_transformations)

    def get_expectationvalues(self):
        """
        Returns
        -------
        list:
            all the expectation values that make up the objective.
        """
        return [arg for arg in self.args if hasattr(arg, "U")]

    def get_expectationvalues_at(self,pos):
        """
        Return all the expectationvalues from a certain set of arguments.
        Parameters
        ----------
        pos: int:
            which position in the set of argsets to return all expectationvalues from.

        Returns
        -------
        list:
            a list of expectation values.
        """
        assert isinstance(pos,int)
        assert pos <= len(self) - 1
        return [arg for arg in self.argsets[pos] if hasattr(arg,'U')]

    def count_expectationvalues(self, unique=True):
        """
        Parameters
        ----------
        unique: bool:
            whether or not to count identical expectationvalues as distinct.

        Returns
        -------
        int:
            how many (possibly, how many unique) expectationvalues are contained within the objective.

        """
        if unique:
            return len(set(self.get_expectationvalues()))
        else:
            return len(self.get_expectationvalues())

    def count_expectationvalues_at(self,pos, unique=True):
        """
        Count all the expectationvalues in a certain argset.
        Parameters
        ----------
        pos: int:
            which position in the list of argsets to check.
        unique: bool, Default = True:
            whether or not to only count unique instances.


        Returns
        -------
        int:
            how many (possibly, how many unique) expectationvalues are contained within the objective.

        """
        assert isinstance(pos,int)
        assert pos <= len(self) - 1
        if unique:
            return len(set(self.get_expectationvalues_at(pos)))
        else:
            return len(self.get_expectationvalues_at(pos))

    def contract(self):
        """
        return 1-d objective, summing over all transformations.

        Returns
        -------
        VectorObjective:
            an VectorObjective whose output, when called, would be the sum over the output of self.
        """
        argsets = self.argsets
        trans = self.transformations
        group = []
        for i, a in enumerate(argsets):
            o = Objective(args=a, transformation=trans[i])
            group.append(o)
        back = group[0]
        for i in range(1,len(group)):
            back += group[i]
        return back

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        variables = self.extract_variables()
        types = [type(E) for E in self.get_expectationvalues()]
        types = list(set(types))

        if ExpectationValueImpl in types:
            if len(types) == 1:
                types = "not compiled"
            else:
                types = "partially compiled to " + str([t for t in types if t is not ExpectationValueImpl])

        unique = self.count_expectationvalues(unique=True)
        return "VectorObjective with {} unique expectation values\n" \
               "variables = {}\n" \
               "types     = {}".format(unique, variables, types)

    def __call__(self, variables=None, *args, **kwargs):
        """
        Return the output of the calculation the objective represents.

        Parameters
        ----------
        variables: dict:
            dictionary instantiating all variables that may appear within the objective.
        args
        kwargs

        Returns
        -------
        float or numpy.ndarray:
            the result of the calculation represented by this objective.
        """
        variables = format_variable_dictionary(variables)
        # avoid multiple evaluations

        eved = []
        for argset in self.argsets:
            evaluated = {}
            ev_array = []
            for E in argset:
                if E not in evaluated:
                    expval_result = E(variables=variables, *args, **kwargs)
                    evaluated[E] = expval_result
                else:
                    expval_result = evaluated[E]
                ev_array.append(expval_result)
            eved.append(ev_array)
        called = []
        for i, f in enumerate(self.transformations):
            called.append(f(*eved[i]))
        if len(called) == 1:
            return called[0]
        else:
            return onp.asarray(called)

    @staticmethod
    def empty(length: int = 1):
        """
        Initialize an empty objective whose return on call would be of length 'length'.

        Notes
        -----
        Named in this way to mirror numpy syntax for initialization of empty arrays.

        Parameters
        ----------
        length: int (Default = 1):
            how long the empty objective should be (i.e, length of its call return)

        Returns
        -------
        VectorObjective:
            an empty objective of length 'length'.

        """
        assert isinstance(length,int)
        f = lambda *x: 0.0
        trans = [f for i in range(length)]
        return VectorObjective(argsets=None, transformations=trans)

    @staticmethod
    def from_list(input):
        """
        Return an n-d array VectorObjective from a list of Objectives whose lengths will total n.
        Parameters
        ----------
        input: list:
            a list of Objectives.

        Returns
        -------
        VectorObjective:
            VectorObjective representing the stacked Objectives of input.
        """
        assert isinstance(input,list)
        assert all([isinstance(i, VectorObjective) or isinstance(i,Objective) for i in input])
        argsets=[]
        transformations=[]
        for i in input:
            if isinstance(i,VectorObjective):
                argsets.extend(i.argsets)
                transformations.extend(i.transformations)
            if isinstance(i,Objective):
                argsets.append(i.args)
                transformations.append(i.transformation)
        return VectorObjective(argsets=argsets, transformations=transformations)


def ExpectationValue(U, H, optimize_measurements: bool = False, *args, **kwargs) -> VectorObjective:
    """
    Initialize an VectorObjective which is just a single expectationvalue
    """
    if optimize_measurements:
        binary_H = BinaryHamiltonian.init_from_qubit_hamiltonian(H)
        commuting_parts = binary_H.commuting_groups()
        result = 0.0
        for cH in commuting_parts:
            qwc, Um = cH.get_qubit_wise()
            Etmp = ExpectationValue(H=qwc, U=U + Um, optimize_measurements=False)
            result += Etmp
        return result
    else:
        return Objective.ExpectationValue(U=U, H=H, *args, **kwargs)


def vectorize(objectives):
    """
    Combine several objectives in order, into one longer vector.

    Parameters
    ----------
    objectives: iterable:
        the objectives to combine as a vector. Note that this is not addition, but the 'end to end' combination of
        vectors; the new objective will have length Sum(len(x) for x in objectives)

    Returns
    -------
    VectorObjective:
        Objectives stacked together.
    """
    l = list_assignment(objectives)
    argsets = []
    trans = []
    for o in l:
        for s in o.argsets:
            argsets.append(s)
        for t in o.transformations:
            trans.append(t)
    return VectorObjective(argsets=argsets, transformations=trans)


class TequilaVariableException(TequilaException):
    def __str__(self):
        return "Error in tequila variable:" + self.message


class Variable:
    """
    Hashable class representing generalized variables.

    E.g, variables are the stand-in parameters of parametrized gates. We implement, here, the ability to manipulate
    this class as if they were floats, in order to allow for symbolic and functional operations on this type.

    Attributes
    ----------
    name:
        the name, identifying the variable.

    Methods
    -------
    extract_variables:
        returns a list containing only self.
    apply:
        generate an objective which applies a callable to self.
    wrap:
        alias for apply.
    """
    @property
    def name(self):
        return self._name

    def __lt__(self, other):
        return hash(self.name) < hash(other.name)

    def __hash__(self):
        return hash(self.name)

    def __init__(self, name: typing.Union[str, typing.Hashable]):
        """
        Parameters
        ----------
        name: hashable:
            a unique identifier for this variable. All Variable instances with the same name are considered the same.

        Raises
        ------
        TequilaVariableException
        """
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
        if hasattr(other, "name"):
            return self.name == other.name
        else:
            return self.name == other

    def _left_helper(self, op, other):
        """
        function for use by magic methods, which all have an identical structure, differing only by the
        external operator they call.

        left helper is responsible for all 'self # other' operations. Note similarity
        to the same function in VectorObjective.

        Parameters
        ----------
        op: callable:
            the operation to be performed
        other:
            the right-hand argument of the operation to be performed

        Returns
        -------
        VectorObjective:
            an VectorObjective, who transform is op, acting on self and other
        """
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
        else:
            raise TequilaException("unknown type in left_helper of objective arithmetics with operation {}: {}".format(type(op), type(other)))
        return new

    def _right_helper(self, op, other):
        """
        see _left_helper for details.
        """
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
        else:
            raise TequilaException("unknown type in left_helper of objective arithmetics with operation {}: {}".format(type(op),type(other)))
        return new

    def __mul__(self, other):
        return self._left_helper(numpy.multiply, other)

    def __add__(self, other):
        return self._left_helper(numpy.add, other)

    def __sub__(self, other):
        return self._left_helper(numpy.subtract, other)

    def __truediv__(self, other):
        return self._left_helper(numpy.true_divide, other)

    def __neg__(self):
        return Objective(args=[self], transformation=lambda v: numpy.multiply(v, -1.))

    def __pow__(self, other):
        return self._left_helper(numpy.float_power, other)

    def __rpow__(self, other):
        return self._right_helper(numpy.float_power, other)

    def __rmul__(self, other):
        return self._right_helper(numpy.multiply, other)

    def __radd__(self, other):
        return self._right_helper(numpy.add, other)

    def __rtruediv__(self, other):
        return self._right_helper(numpy.true_divide, other)

    def __invert__(self):
        new = Objective(args=[self])
        return new ** -1.0

    def __len__(self):
        return 1

    def __ne__(self, other):
        if self.__eq__(other):
            return False
        else:
            return True

    def apply(self, other):
        assert (callable(other))
        return Objective(args=[self], transformation=other)

    def wrap(self, other):
        return self.apply(other)

    def __repr__(self):
        return str(self.name)

    def toJson(self):
        import json
        return json.dumps(self, default=lambda o: o.__dict__)

class FixedVariable(float):
    """
    Wrapper over floats, to allow them to mimic Variable objects, for 'duck typing' of gate parameters.

    Methods
    -------
    apply:
        generate an objective with one argument -- self -- and other as transformation.
    wrap:
        alias for apply.
    """

    def __call__(self, *args, **kwargs):
        return self

    def apply(self, other):
        assert (callable(other))
        return Objective(args=[self], transformation=other)

    def wrap(self, other):
        return self.apply(other)


def format_variable_list(variables: typing.List[typing.Hashable]) -> typing.List[Variable]:
    """
    Convenience functions to assign tequila variables.
    Parameters
    ----------
    variables:
        a list with Hashables as elements.

    Returns
    -------
    list:
        a list with tq.Variable types as keys
    """
    if variables is None:
        return variables
    else:
        return [assign_variable(k) for k in variables]


def format_variable_dictionary(variables: typing.Dict[typing.Hashable, typing.Any]) -> typing.Dict[
    Variable, typing.Any]:
    """
    Convenience function to assign tequila variables.
    Parameters
    ----------
    variables:
        a dictionary with Hashables as keys

    Returns
    -------
    dict:
        a dictionary with tq.Variable types as keys
    """
    if variables is None:
        return variables
    else:
        return Variables(variables)


def assign_variable(variable: typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable]) -> typing.Union[
    Variable, FixedVariable]:
    """
    Convenience function; maps various objects into Variable, FixedVariable, or Variables, for easy duck-typing.

    Parameters
    ----------
    variable:
        a string, a number or a variable.

    Raises
    ------
    TequilaVariableException


    Returns
    -------
    Variable or FixedVariable:
        A duck-typing adjusted version of the input. If hashable, a Variable. If a number, a FixedVariable.
    """
    if isinstance(variable, str):
        return Variable(name=variable)
    elif isinstance(variable, Variable):
        return variable
    elif isinstance(variable, Objective):
        return variable
    elif isinstance(variable, VectorObjective):
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
    Dictionary-like object for tequila variables.

    Allows hashable types and variable types as keys

    Attributes
    ----------
    store: dict:
        the internal dictionary around which this structure is built.

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
        for k, v in self.items():
            result += "{} : {}\n".format(str(k), str(v))
        return result

    def __repr__(self):
        return self.__str__()
