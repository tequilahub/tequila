from openvqe.circuit.circuit import QCircuit
from openvqe.circuit._gates_impl import RotationGateImpl, PowerGateImpl, QGateImpl, MeasurementImpl
from openvqe import OpenVQEException
from openvqe import typing
import functools


def wrap_gate(func):
    @functools.wraps(func)
    def doit(*args, **kwargs):
        return QCircuit.wrap_gate(func(*args, **kwargs))

    return doit


@wrap_gate
def RotationGate(axis, angle, target: typing.Union[list, int], control: typing.Union[list, int] = None,
                 frozen: bool = None):
    return RotationGateImpl(axis=axis, angle=angle, target=target, control=control, frozen=frozen)


@wrap_gate
def PowerGate(name, target: typing.Union[list, int], power: bool = None, control: typing.Union[list, int] = None,
              frozen: bool = None):
    return PowerGateImpl(name=name, power=power, target=target, control=control, frozen=frozen)


@wrap_gate
def QGate(name, target: typing.Union[list, int], control: typing.Union[list, int] = None):
    return QGateImpl(name=name, target=target, control=control)


@wrap_gate
def Rx(angle, target: typing.Union[list, int], control: typing.Union[list, int] = None, frozen: bool = None):
    return RotationGateImpl(axis=0, angle=angle, target=target, control=control, frozen=frozen)


@wrap_gate
def Ry(angle, target: typing.Union[list, int], control: typing.Union[list, int] = None, frozen: bool = None):
    return RotationGateImpl(axis=1, angle=angle, target=target, control=control, frozen=frozen)


@wrap_gate
def Rz(angle, target: typing.Union[list, int], control: typing.Union[list, int] = None, frozen: bool = None):
    return RotationGateImpl(axis=2, angle=angle, target=target, control=control, frozen=frozen)


@wrap_gate
def X(target: typing.Union[list, int], control: typing.Union[list, int] = None, power=None, frozen: bool = None):
    return PowerGateImpl(name="X", power=power, target=target, control=control, frozen=frozen)


@wrap_gate
def H(target: typing.Union[list, int], control: typing.Union[list, int] = None, power=None, frozen: bool = None):
    return PowerGateImpl(name="H", power=power, target=target, control=control, frozen=frozen)


@wrap_gate
def Y(target: typing.Union[list, int], control: typing.Union[list, int] = None, power=None, frozen: bool = None):
    return PowerGateImpl(name="Y", power=power, target=target, control=control, frozen=frozen)


@wrap_gate
def Z(target: typing.Union[list, int], control: typing.Union[list, int] = None, power=None, frozen: bool = None):
    return PowerGateImpl(name="Z", power=power, target=target, control=control, frozen=frozen)


@wrap_gate
def Measurement(target, name=None):
    if name is None:
        return MeasurementImpl(name="", target=target)
    else:
        return MeasurementImpl(name=name, target=target)


"""
Convenience for Two Qubit Gates
iSWAP will only work with cirq, the others will be recompiled
"""


@wrap_gate
def SWAP(q0: int, q1: int, control: typing.Union[int, list] = None, power: float = None,
         frozen: bool = None):
    return PowerGateImpl(name="SWAP", target=[q0, q1], control=control, power=power, frozen=frozen)


@wrap_gate
def iSWAP(q0: int, q1: int, control: typing.Union[int, list] = None):
    return PowerGateImpl(name="ISWAP", target=[q0, q1], control=control)


"""
Convenience Initialization Routines for controlled gates
All following the patern: Gate(control_qubit, target_qubit, possible_parameter)
"""


def enforce_integer(function) -> int:
    """
    Replace if we have a qubit class at some point
    :param obj:
    :return: int(obj)
    """

    def wrapper(control, target, *args):
        try:
            control = int(control)
        except ValueError as e:
            raise OpenVQEException(
                "Could not initialize gate: Conversion of input type for control-qubit failed\n" + str(e))
        try:
            target = int(target)
        except ValueError as e:
            raise OpenVQEException(
                "Could not initialize gate: Conversion of input type for target-qubit failed\n" + str(e))
        return function(control, target, *args)

    return wrapper


@enforce_integer
def CNOT(control: int, target: int, frozen: bool = None) -> QCircuit:
    return X(target=target, control=control, frozen=frozen)


@enforce_integer
def CX(control: int, target: int, frozen: bool = None) -> QCircuit:
    return X(target=target, control=control, frozen=frozen)


@enforce_integer
def CY(control: int, target: int, frozen: bool = None) -> QCircuit:
    return Y(target=target, control=control, frozen=frozen)


@enforce_integer
def CZ(control: int, target: int, frozen: bool = None) -> QCircuit:
    return Z(target=target, control=control, frozen=frozen)


@enforce_integer
def CRx(control: int, target: int, angle: float, frozen: bool = None) -> QCircuit:
    return Rx(target=target, control=control, angle=angle, frozen=frozen)


@enforce_integer
def CRy(control: int, target: int, angle: float, frozen: bool = None) -> QCircuit:
    return Ry(target=target, control=control, angle=angle, frozen=frozen)


@enforce_integer
def CRz(control: int, target: int, angle: float, frozen: bool = None) -> QCircuit:
    return Rz(target=target, control=control, angle=angle, frozen=frozen)


if __name__ == "__main__":
    G = CRx(1, 0, 2.0)

    print(G)
