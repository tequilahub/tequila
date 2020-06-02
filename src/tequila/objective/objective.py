import typing, copy, numbers

from tequila import TequilaException
from tequila.utils import JoinedTransformation, to_float
from tequila.tools.convenience import list_assignment
from tequila.hamiltonian import paulis
import numpy as onp
from tequila.autograd_imports import numpy as numpy


import collections


class ExpectationValueImpl:
    """
    Implements the (uncompiled) Expectation Value as a class. Should not be called directly.

    common arithmetical operations like addition, multiplication, etc. are defined, to return Objective objects.

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
    def H(self):
        if self._hamiltonian is None:
            return paulis.QubitHamiltonian.unit()
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
    fundamental class for arithmetic on quantum data; the core of tequila.

    A callable class which returns either a scalar or a vector. Implements all tools needed for arithmetic,
    as well as automatic differentiation, of user defined objectives, which may be either quantum or classical
    in nature.

    """
    def __init__(self, argsets: typing.Iterable = None, transformations: typing.Iterable[callable] = None):
        """

        Parameters
        ----------
        argsets: iterable:
            a (list, tuple) of (list,tuple) of arguments; each the argument of a transformation.
        transformations: iterable:
            a (list, tuple) of callables; these determine the output of call.
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
        Extract all variables on which the objective depends
        :return: List of all Variables
        """
        variables = []
        for arg in self.args:
            if hasattr(arg, 'extract_variables'):
                variables += arg.extract_variables()
            else:
                variables += []

        return list(set(variables))

    def is_expectationvalue(self):
        """
        :return: bool: whether or not this objective is just a wrapped ExpectationValue
        """
        return len(self.args) == 1 and self._transformations is None and type(self.args[0]) is ExpectationValueImpl

    def has_expectationvalues(self):
        """
        :return: bool: wether or not this objective has expectationvalues or is just a function of the variables
        """
        # testing if all arguments are only variables and give back the negative
        return any([hasattr(arg, "U") for arg in self.args])

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
        Objective:
            an Objective, who transform is the joined_transform of self with op, acting on self and other.
        """
        sized=self._size_helper(len(self),other)
        if isinstance(sized, numpy.ndarray):
            ops=[lambda v: op(v, s) for s in sized]
            new = self.unary_operator(left=self, ops=ops)
        elif isinstance(sized, Objective):
            new = self.binary_operator(left=self, right=other, op=op)
        else:
            t = op
            nother = Objective(argsets=[[assign_variable(other)]])
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
        elif isinstance(other, Objective):
            new = self.binary_operator(left=other, right=self, op=op)
        else:
            t = op
            nother = Objective(argsets=[[assign_variable(other)]])
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
        new = Objective(argsets=self.argsets, transformations=self.transformations)
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
            if ls == 1:
                return other
            lo = len(other)
            if lo == 1:
                argset=other.argsets[0]
                argsets=[argset for i in range(ls)]
                transform=other.transformations[0]
                transforms=[transform for i in range(ls)]
                return Objective(argsets=argsets, transformations=transforms)
            if ls != lo:
                raise TequilaException('cannot combine objectives of len  {} and {}!'.format(ls,len(other)))
            else:
                return other
        elif isinstance(other,ExpectationValueImpl):
            return Objective(argsets=[[other] for i in range(ls)])
        elif isinstance(other,Variable):
            return Objective(argsets=[[other] for i in range(ls)])
        else:
            raise TequilaException('Received unknown object of type {}'.format(type(other)))

    @classmethod
    def unary_operator(cls, left, ops):
        """
        Arithmetical function for unary operations.
        Generally, called by the magic methods of Objective itself.
        Parameters
        ----------
        left: Objective:
            the objective to which op will be applied

        ops: list of Callable:
            operations to apply to left

        Returns
        -------
        Objective:
            Objective representing ops applied to objective left.

        """
        transformations=[]
        for i,op in enumerate(ops):
            transformations.append(lambda *args: op(left.transformations[i](*args)))
        return Objective(argsets=left.argsets,
                         transformations=transformations)

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
            op acting on left, right. Will arguments appropriately if one is scalar.
        """
        if isinstance(right, numbers.Number) or isinstance(right, numpy.ndarray):
            if isinstance(left, Objective):
                sized_r=left._size_helper(len(left),right)
                sized_l=left._size_helper(len(right),left)
                ops=[lambda E: op(E, s) for s in sized_r]
                return cls.unary_operator(left=sized_l, ops=ops)
            else:
                raise TequilaException(
                    'BinaryOperator method called on types ' + str(type(left)) + ',' + str(type(right)))
        elif isinstance(left, numbers.Number) or isinstance(left, numpy.ndarray):
            if isinstance(right, Objective):
                sized_r = right._size_helper(len(left), right)
                sized_l = right._size_helper(len(right), left)
                ops = [lambda E: op(s, E) for s in sized_l]
                return cls.unary_operator(left=sized_r, ops=ops)
            else:
                raise TequilaException(
                    'BinaryOperator method called on types ' + str(type(left)) + ',' + str(type(right)))
        else:
            sized_r=Objective._size_helper(len(left),right)
            sized_l=Objective._size_helper(len(right),left)
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
            return Objective(argsets=sets,
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
        Objective:
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
        Objective:
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

        return Objective(argsets=argsets,transformations=new_transformations)

    def apply_op_list(self,oplist):
        """
        Apply a list of operations to each output of self.
        Parameters
        ----------
        oplist:  list of Callabe:
            the list of operations to compose with the current transformations. Must have equal length.

        Returns
        -------
        Objective:
            an Objective, corresponding to oplist[i] composed with self.transformations[i] for all i.
        """
        assert len(oplist) == len(self)
        argsets = self.argsets
        transformations = self.transformations
        new_transformations = []
        for i, t in enumerate(transformations):
            new = lambda *args: oplist[i](t(*args))
            new_transformations.append(new)

        return Objective(argsets=argsets, transformations=new_transformations)


    def get_expectationvalues(self):
        """
        Returns
        -------
        list:
            all the expectation values that make up the objective.
        """
        return [arg for arg in self.args if hasattr(arg, "U")]

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

    def contract(self):
        argsets=self.argsets
        trans=self.transformations
        group=[]
        for i, a in enumerate(argsets):
            o = Objective(argsets=[a], transformations=[trans[i]])
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
        return "Objective with {} unique expectation values\n" \
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

        eved=[]
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
        called=[]
        for i,f in enumerate(self.transformations):
            called.append(f(*eved[i]))
        if len(called) == 1:
            return called[0]
        else:
            return onp.asarray(called)

    @staticmethod
    def empty(length: int = 1):
        f = lambda *x: 0.0
        trans = [f for i in range(length)]
        return Objective(argsets=None, transformations=trans)

    @staticmethod
    def from_list(input):
        assert isinstance(input,list)
        assert all([isinstance(i, Objective) for i in input])
        argsets=[]
        transformations=[]
        for i in input:
            argsets.extend(i.argsets)
            transformations.extend(i.transformations)
        return Objective(argsets=argsets, transformations=transformations)

    @staticmethod
    def ExpectationValue(U,H,*args,**kwargs):
        ev=ExpectationValueImpl(U=U,H=H,*args,**kwargs)
        return Objective(argsets=[ev])


def ExpectationValue(U, H, *args, **kwargs) -> Objective:
    """
    Initialize an Objective which is just a single expectationvalue
    """
    return Objective.ExpectationValue(U=U, H=H, *args, **kwargs)


def stack_objectives(objectives):
    l=list_assignment(objectives)
    argsets=[]
    trans=[]
    for o in l:
        for s in o.argsets:
            argsets.append(s)
        for t in o.transformations:
            trans.append(t)
    return Objective(argsets=argsets,transformations=trans)


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

    def _left_helper(self, op, other):
        """
        function for use by magic methods, which all have an identical structure, differing only by the
        external operator they call.

        left helper is responsible for all 'self # other' operations. Note similarity
        to the same function in Objective.

        Parameters
        ----------
        op: callable:
            the operation to be performed
        other:
            the right-hand argument of the operation to be performed

        Returns
        -------
        Objective:
            an Objective, who transform is op, acting on self and other
        """
        as_obj=Objective(argsets=[[self]])
        return as_obj._left_helper(op,other)


    def _right_helper(self, op, other):
        """
        see _left_helper for details.
        """

        as_obj=Objective(argsets=[[self]])
        return as_obj._right_helper(op,other)

    def __mul__(self, other):
        return self._left_helper(numpy.multiply, other)

    def __add__(self, other):
        return self._left_helper(numpy.add, other)

    def __sub__(self, other):
        return self._left_helper(numpy.subtract, other)

    def __truediv__(self, other):
        return self._left_helper(numpy.true_divide, other)

    def __neg__(self):
        return Objective(argsets=[[self]], transformations=[lambda v: numpy.multiply(v, -1.)])

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
        new = Objective(argsets=[[self]])
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
        return Objective(argsets=[[self]], transformations=[other])

    def wrap(self, other):
        return self.apply(other)

    def __repr__(self):
        return str(self.name)


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
        return Objective(argsets=[[self]], transformations=[other])

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
