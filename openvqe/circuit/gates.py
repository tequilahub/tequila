from openvqe.circuit.circuit import QCircuit
from openvqe.circuit._gates_impl import RotationGateImpl, PowerGateImpl, QGateImpl
import functools


def wrap_gate(func):
    @functools.wraps(func)
    def doit(*args, **kwargs):
        return QCircuit.wrap_gate(func(*args, **kwargs))

    return doit


@wrap_gate
def RotationGate(axis, angle, target: list, control: list = None, frozen: bool = False, phase=1.0):
    return RotationGateImpl(axis=axis, angle=angle, target=target, control=control, frozen=frozen, phase=phase)


@wrap_gate
def PowerGate(name, target: list, power=1.0, control: list = None, frozen: bool = False, phase=1.0):
    return PowerGateImpl(name=name, power=power, target=target, control=control, frozen=frozen, phase=phase)


@wrap_gate
def QGate(name, target: list, control: list = None, phase=1.0):
    return QGateImpl(name=name, target=target, control=control, phase=phase)


@wrap_gate
def Rx(angle, target, control=None, frozen=None, phase=1.0):
    return RotationGateImpl(axis=0, angle=angle, target=target, control=control, frozen=frozen, phase=phase)


@wrap_gate
def Ry(angle, target, control=None, frozen=None, phase=1.0):
    return RotationGateImpl(axis=1, angle=angle, target=target, control=control, frozen=frozen, phase=phase)


@wrap_gate
def Rz(angle, target, control=None, frozen=None, phase=1.0):
    return RotationGateImpl(axis=2, angle=angle, target=target, control=control, frozen=frozen, phase=phase)


@wrap_gate
def X(target, control=None, power=None, frozen=None, phase=1.0):
    if power is None:
        return QGate(name="X", target=target, control=control, phase=phase)
    else:
        return PowerGateImpl(name="X", power=power, target=target, control=control, frozen=frozen, phase=phase)


def CNOT(target, control=None):
    if control is None:
        assert (len(target) == 2)
        control = target[1]
        target = target[0]
    return X(target=target, control=control)


@wrap_gate
def H(target, control=None, power=None, frozen=None, phase=1.0):
    if power is None:
        return QGate(name="H", target=target, control=control, phase=phase)
    else:
        return PowerGateImpl(name="H", power=power, target=target, control=control, frozen=frozen, phase=phase)


@wrap_gate
def Y(target, control=None, power=None, frozen=None, phase=1.0):
    if power is None:
        return QGate(name="Y", target=target, control=control, phase=phase)
    else:
        return PowerGateImpl(name="Y", power=power, target=target, control=control, frozen=frozen, phase=phase)


@wrap_gate
def Z(target, control=None, power=None, frozen=None, phase=1.0):
    if power is None:
        return QGate(name="Z", target=target, control=control, phase=phase)
    else:
        return PowerGateImpl(name="Z", power=power, target=target, control=control, frozen=frozen, phase=phase)


@wrap_gate
def SWAP(target, control=None, power=None, frozen=None, phase=1.0):
    assert (len(target) >= 2)
    if power is None:
        return QGate(name="SWAP", target=target, control=control, phase=phase)
    else:
        return PowerGateImpl(name="SWAP", power=power, target=target, control=control, frozen=frozen, phase=phase)
