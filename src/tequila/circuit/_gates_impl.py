import typing
import copy
import numbers
from abc import ABC
from tequila import TequilaException
from tequila.circuit.variable import Variable, SympyVariable
from tequila.hamiltonian import PauliString, QubitHamiltonian
from tequila.tools import number_to_string, list_assignement
from tequila.circuit.variable import has_variable


class QGateImpl:
    def __init__(self, name, target: list, control: list = None):
        self.name = name
        self.target = tuple(list_assignement(target))
        self.control = tuple(list_assignement(control))
        self.finalize()

    def copy(self):
        return copy.deepcopy(self)

    def is_frozen(self):
        raise Exception(
            'unparametrized gates cannot be frozen because there is nothing to freeze. \n If you want to iterate over all your gates, use is_differentiable as a criterion before or in addition to is_frozen')

    def dagger(self):
        """
        :return: return the hermitian conjugate of the gate.
        """

        return QGateImpl(name=copy.copy(self.name), target=self.target,
                         control=self.control)

    def is_controlled(self) -> bool:
        """
        :return: True if the gate is controlled
        """
        if self.control:
            return True
        else:
            return False

    def is_parametrized(self) -> bool:
        """
        :return: True if the gate is parametrized
        """
        return False

    def is_single_qubit_gate(self) -> bool:
        """
        Convenience and easier to interpret
        :return: True if the Gate only acts on one qubit (not controlled)
        """
        return ((not self.control) and (len(self.target) == 1))

    def is_differentiable(self) -> bool:
        '''
        defaults to False, overwridden by ParametrizedGate
        '''
        return False

    def finalize(self):
        if not self.target:
            raise Exception('Received no targets upon initialization')
        if self.is_controlled():
            for c in self.target:
                if c in self.control:
                    raise Exception("control and target are the same qubit: " + self.__str__())

        # Set the active qubits
        if self.control:
            self.qubits = self.target + self.control
        else:
            self.qubits = self.target

        self.max_qubit = self.compute_max_qubit()

    def __str__(self):
        result = str(self.name) + "(target=" + str(self.target)
        if not self.is_single_qubit_gate():
            result += ", control=" + str(self.control)
        result += ")"
        return result

    def __repr__(self):
        """
        Todo: Add Nice stringification
        """
        return self.__str__()

    def compute_max_qubit(self):
        """
        :return: highest qubit index used by this gate
        """
        if self.control is None:
            return max(self.target)
        else:
            return max(self.target + self.control)

    def __eq__(self, other):
        if self.name != other.name:
            return False
        if self.target != other.target:
            return False
        if self.control != other.control:
            return False
        return True


class MeasurementImpl(QGateImpl):

    def __init__(self, name, target):
        self.name = name
        self.target = tuple(sorted(list_assignement(target)))
        self.control = tuple()
        self.finalize()


class ParametrizedGateImpl(QGateImpl, ABC):
    '''
    the base class from which all parametrized gates inherit. User defined gates, when implemented, are liable to be members of this class directly.
    Has su
    '''

    def dagger(self):
        raise TequilaException("should not be called from ABC")

    def update_variables(self, variables: typing.Dict[str, numbers.Real]):
        for k, v in variables.items():
            if has_variable(self.parameter, k):
                self.parameter.update({k: v})

    def extract_variables(self):
        if hasattr(self.parameter, "variables"):
            return self.parameter.variables

    @property
    def parameter(self):
        return self._parameter

    @parameter.setter
    def parameter(self, other):
        if isinstance(other, numbers.Number):
            self._parameter = Variable(value=other)
        elif isinstance(other, str):
            self._parameter = Variable(name=other, value=0.0)
        elif hasattr(other, "evalf"):
            self._parameter = SympyVariable(value=other)
        else:
            self._parameter = other

    def __init__(self, name, parameter: Variable, target: list, control: list = None, frozen: bool = None):
        super().__init__(name, target, control)

        # failsafe:
        if frozen is not None and not frozen and isinstance(parameter, numbers.Number):
            raise TequilaException(
                "\nYou explicitly demanded a parametrized gate with frozen=False\n"
                "but have not passed down a Variable object but a simple number.\n"
                "initialize the gate with a Variable object like Variable(name=\'pick_a_name\', value=number)")
        elif frozen is None and isinstance(parameter, numbers.Number):
            self.frozen = True
        else:
            self.frozen = frozen

        self.parameter = parameter

    def is_frozen(self):
        '''
        :return: return wether this gate is frozen or not.
        '''
        return self.frozen

    def is_parametrized(self) -> bool:
        """
        :return: True if the gate is parametrized
        """
        if self._parameter is None:
            return False
        else:
            return True

    def is_differentiable(self) -> bool:
        """
        :return: True if the gate is differentiable
        """
        return True

    def __str__(self):
        result = str(self.name) + "(target=" + str(self.target)
        if not self.is_single_qubit_gate():
            result += ", control=" + str(self.control)

        result += ", parameter=" + str(self._parameter)
        result += ")"
        return result

    def __eq__(self, other):
        if not isinstance(other, ParametrizedGateImpl):
            return False
        if not super().__eq__(other):
            return False
        if self._parameter != other._parameter:
            return False
        return True


class RotationGateImpl(ParametrizedGateImpl):
    axis_to_string = {0: "x", 1: "y", 2: "z"}
    string_to_axis = {"x": 0, "y": 1, "z": 2}

    @staticmethod
    def get_name(axis):
        axis = RotationGateImpl.assign_axis(axis)
        return "R" + RotationGateImpl.axis_to_string[axis]

    @property
    def axis(self):
        return self._axis

    @axis.setter
    def axis(self, value):
        self._axis = self.assign_axis(value)

    @property
    def angle(self):
        return self.parameter

    @angle.setter
    def angle(self, angle):
        self.parameter = angle

    def __ipow__(self, power, modulo=None):
        self.angle *= power
        return self

    def __pow__(self, power, modulo=None):
        result = copy.deepcopy(self)
        result.angle *= power
        return result

    def __mul__(self, other) -> list:
        """
        Helper function for QCircuit, should not be used on its own
        As every class in _gates_impl.py
        Tries to optimize if two rotation gates are combined
        """
        if hasattr(other,
                   "angle") and other.axis == self._axis and other.target == self.target and other.control == self.control:
            result = copy.deepcopy(self)
            result.angle = self.angle + other.angle
            result.frozen = self.frozen or other.frozen
            return [result]
        else:
            return [self, other]

    def __init__(self, axis, angle, target: list, control: list = None, frozen: bool = None):
        assert (angle is not None)
        super().__init__(name=self.get_name(axis=axis), parameter=angle, target=target, control=control, frozen=frozen)
        self._axis = self.assign_axis(axis)

    @staticmethod
    def assign_axis(axis):
        if axis in RotationGateImpl.string_to_axis:
            return RotationGateImpl.string_to_axis[axis]
        elif hasattr(axis, "lower") and axis.lower() in RotationGateImpl.string_to_axis:
            return RotationGateImpl.string_to_axis[axis.lower()]
        else:
            assert (axis in [0, 1, 2])
            return axis

    def dagger(self):
        result = copy.deepcopy(self)
        result.angle = -self.angle
        return result


class PowerGateImpl(ParametrizedGateImpl):

    @property
    def power(self):
        if self.parameter is None:
            return 1
        else:
            return self.parameter

    @power.setter
    def power(self, power):
        self.parameter = power

    def __ipow__(self, other):
        if self.parameter is None:
            self.power = other
        else:
            self.power = self.power * other
        return self

    def __pow__(self, power, modulo=None):
        result = copy.deepcopy(self)
        result.power *= power
        return result

    def __mul__(self, other) -> list:
        """
        Helper function for QCircuit, should not be used on its own
        As every class in _gates_impl.py
        Tries to optimize if two rotation gates are combined
        """
        if hasattr(other,
                   "power") and other.name == self.name and other.target == self.target and other.control == self.control:
            result = copy.deepcopy(self)
            result.power = self.power + other.power
            result.frozen = self.frozen or other.frozen
            return [result]
        else:
            return [self, other]

    def __init__(self, name, target: list, power=None, control: list = None, frozen: bool = None):
        super().__init__(name=name, parameter=power, target=target, control=control, frozen=frozen)

    def dagger(self):
        result = copy.deepcopy(self)
        return result


class ExponentialPauliGateImpl(ParametrizedGateImpl):
    """
    Same convention as for rotation gates:
    Exp(-i angle/2 * paulistring)
    """

    @property
    def angle(self):
        return self.parameter

    @angle.setter
    def angle(self, angle):
        self.parameter = angle

    @property
    def name(self):
        return "Exp(" + number_to_string(self.angle() * 1j) + "/2 PS)"

    def __init__(self, paulistring: PauliString, angle: float, control: typing.List[int] = None, frozen: bool = False):
        self.paulistring = paulistring
        self.parameter = angle
        self.target = tuple(t for t in paulistring.keys())
        self.control = tuple(list_assignement(control))
        self.frozen = frozen
        self.finalize()

    def __str__(self):
        result = str(self.name) + "(target=" + str(self.target)
        if not self.is_single_qubit_gate():
            result += ", control=" + str(self.control)

        result += ", parameter=" + str(self._parameter)
        result += ", paulistring=" + str(self.paulistring)
        result += ")"
        return result


class TrotterizedGateImpl(ParametrizedGateImpl):

    def update_variables(self, variables: typing.Dict[str, numbers.Real]):
        for k, v in variables.items():
            for angle in self.angles:
                if has_variable(angle, k):
                    angle.update({k: v})

    def extract_variables(self) -> typing.Dict[str, numbers.Number]:
        tmp = dict()
        for angle in self.angles:
            if hasattr(angle, "variables"):
                for k, v in angle.variables.items():
                    tmp[k] = v
        return tmp

    @property
    def parameter(self):
        return self.angles

    @parameter.setter
    def parameter(self, other):
        assert (len(other) == len(self.generators))
        self._parameter = other

    @property
    def angles(self):
        return self._parameter

    @angles.setter
    def angles(self, other):
        if other is None:
            self._parameter = tuple([1] * len(self.generators))
        elif hasattr(other, "__len__"):
            if len(other) == 1:
                self._parameter = tuple([other[0]] * len(self.generators))
            else:
                assert (len(other) == len(self.generators))
                self._parameter = tuple(other)
        else:
            self._parameter = tuple([other] * len(self.generators))

    def __init__(self, generators: typing.Union[QubitHamiltonian, typing.List[QubitHamiltonian]],
                 steps: int = 1,
                 angles: typing.Union[list, numbers.Real, Variable] = None,
                 control: typing.Union[list, int] = None,
                 frozen: bool = None,
                 threshold: numbers.Real = 0.0,
                 join_components: bool = True,
                 randomize_component_order: bool = True,
                 randomize: bool = True):
        """
        :param generators: list of generators
        :param angles: coefficients for each generator
        :param steps: Trotter Steps
        :param control: control qubits
        :param frozen: freeze the gate (optimizers ingnore it)
        :param threshold: neglect terms in the given Hamiltonians if their coefficients are below this threshold
        :param join_components: The generators are trotterized together. If False the first generator is trotterized, then the second etc
        Note that for steps==1 as well as len(generators)==1 this has no effect
        :param randomize_component_order: randomize the order in the generators order before trotterizing
        :param randomize: randomize the trotter decomposition of each generator
        """
        self.generators = list_assignement(generators)
        self.target = self.extract_targets()
        self.angles = angles
        self.control = tuple(list_assignement(control))

        # failsafe for now
        all_variable = True
        all_number = True
        for a in self.angles:
            if isinstance(a, numbers.Number):
                all_variable = False
            if hasattr(a, "has_var"):
                all_number = False
        assert (all_variable != all_number)
        if all_number:
            self.frozen = True
        else:
            self.frozen = frozen

        self.steps = steps
        self.threshold = threshold
        self.join_components = join_components
        self.randomize_component_order = randomize_component_order
        self.randomize = randomize
        self.name = "Trotterized"
        self.finalize()

    def __str__(self):
        result = str(self.name) + "(target=" + str(self.target)
        if not self.is_single_qubit_gate():
            result += ", control=" + str(self.control)

        result += ", angles=" + str(self._parameter)
        result += ", generators=" + str(self.generators)
        result += ")"
        return result

    def extract_targets(self):
        targets = []
        for g in self.generators:
            for ps in g.paulistrings:
                targets += [k for k in ps.keys()]
        return tuple(set(targets))
