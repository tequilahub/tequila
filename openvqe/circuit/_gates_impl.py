from abc import ABC
from openvqe import OpenVQEException
from openvqe import typing
from openvqe import numpy
from openvqe import copy


class QGateImpl:

    @staticmethod
    def list_assignement(o):
        """
        --> moved to tools
        Helper function to make initialization with lists and single elements possible
        :param o: iterable object or single element
        :return: Gives back a list if a single element was given
        """
        if o is None:
            return None
        elif hasattr(o, "__get_item__"):
            return o
        elif hasattr(o, "__iter__"):
            return o
        else:
            return [o]

    def __init__(self, name, target: list, control: list = None, phase=1.0):
        self.name = name
        self.phase = phase
        self.target = self.list_assignement(target)
        self.control = self.list_assignement(control)
        self.verify()

    def is_frozen(self):
        raise Exception(
            'unparametrized gates cannot be frozen because there is nothing to freeze. \n If you want to iterate over all your gates, use is_differentiable as a criterion before or in addition to is_frozen')

    def dagger(self):
        """
        :return: return the hermitian conjugate of the gate.
        """

        return QGateImpl(name=copy.copy(self.name), target=copy.deepcopy(self.target),
                         control=copy.deepcopy(self.control), phase=numpy.conj(self.phase))

    def is_controlled(self) -> bool:
        """
        :return: True if the gate is controlled
        """
        return self.control is not None

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
        return (self.control is None or len(self.control) == 0) and len(self.target) == 1

    def is_differentiable(self) -> bool:
        '''
        defaults to False, overwridden by ParametrizedGate
        '''
        return False

    def verify(self):
        if self.target is None:
            raise Exception('Recieved no targets upon initialization')
        if len(self.list_assignement(self.target)) < 1:
            raise Exception('Recieved no targets upon initialization')
        if self.is_controlled():
            for c in self.target:
                if c in self.control:
                    raise Exception("control and target are the same qubit: " + self.__str__())
        if not numpy.isclose(numpy.abs(self.phase), 1.0):
            raise Exception('Phase must lie on the complex unit circle (I.E, have modulus of 1)')

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

    @property
    def qubits(self) -> typing.List[int]:
        if self.control is not None:
            return self.target + self.control
        else:
            return self.target


    def max_qubit(self):
        """
        :return: highest qubit index used by this gate
        """
        result = max(self.target)
        if self.control is not None:
            result = max(result, max(self.control))
        return result

    def is_phased(self):
        '''
        TODO: make sure this is functional.
        '''
        return self.phase not in [1.0, 1.0 + 0.j]

    def __eq__(self, other):
        if self.name != other.name:
            return False
        if self.target != other.target:
            return False
        if self.control != other.control:
            return False
        if self.phase != other.phase:
            return False
        return True

class MeasurementImpl(QGateImpl):

    def __init__(self, name, target):
        self.name = name
        self.target = sorted(self.list_assignement(target))
        self.control = None


class ParametrizedGateImpl(QGateImpl, ABC):
    '''
    the base class from which all parametrized gates inherit. User defined gates, when implemented, are liable to be members of this class directly.
    Has su
    '''

    def dagger(self):
        raise OpenVQEException("should not be called from ABC")
        return self

    def __init__(self, name, parameter, target: list, control: list = None, frozen: bool = False, phase=1.0):
        super().__init__(name, target, control, phase=phase)
        self.parameter = parameter
        self.frozen = frozen

    def is_frozen(self):
        '''
        :return: return wether this gate is frozen or not.
        '''
        return self.frozen

    def is_parametrized(self) -> bool:
        """
        :return: True if the gate is parametrized
        """
        if self.parameter is None:
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

        result += ", parameter=" + str(self.parameter)
        result += ")"
        return result

    def __eq__(self, other):
        if not isinstance(other, ParametrizedGateImpl):
            return False
        if not super().__eq__(other):
            return False
        if self.parameter != other.parameter:
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

    def __init__(self, axis, angle, target: list, control: list = None, frozen: bool = False, phase=1.0):
        assert (angle is not None)
        super().__init__(name=self.get_name(axis=axis), parameter=angle, target=target, control=control, frozen=frozen,
                         phase=phase)
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
        result.phase = self.phase.conjugate()
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

    def __init__(self, name, target: list, power=None, control: list = None, frozen: bool = False, phase=1.0):
        super().__init__(name=name, parameter=power, target=target, control=control, frozen=frozen,
                         phase=phase)

    def dagger(self):
        result = copy.deepcopy(self)
        result.phase = self.phase.conjugate()
        return result
