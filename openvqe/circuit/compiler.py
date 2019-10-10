"""
Primitive Compiler from Qubit-Operators to evolution operators
Replace with fancier external packages at some point
"""
from openvqe import OpenVQEException
from openvqe.circuit.circuit import QCircuit
from openvqe import numpy
from openvqe.circuit.gates import Rx, H
from openvqe.circuit._gates_impl import RotationGateImpl, QGateImpl
import copy

def compile_multitarget(gate) -> QCircuit:
    # for the case that gate is actually a whole circuit
    if hasattr(gate, "gates"):
        result = QCircuit()
        for g in gate.gates:
            result *= compile_multitarget(gate=g)
        return result

    targets = gate.target

    result = QCircuit()
    for t in targets:
        gx = copy.deepcopy(gate)
        gx.target = [t]
        result *= gx

    return result


def change_basis(target, axis, daggered=False):
    if isinstance(axis, str):
        axis = RotationGateImpl.string_to_axis[axis.lower()]

    if axis == 0:
        return H(target=target, frozen=True)
    elif axis == 1 and daggered:
        return Rx(angle=-numpy.pi / 2, target=target, frozen=True)
    elif axis == 1:
        return Rx(angle=numpy.pi / 2, target=target, frozen=True)
    else:
        return QCircuit()


def compile_controlled_rotation_gate(gate: RotationGateImpl, angles: list = None) -> QCircuit:
    """
    Recompilation of a controlled-rotation gate
    Basis change into Rz then recompilation of controled Rz, then change basis back
    :param gate: The rotational gate
    :param angles: new angles to set, given as a list of two. If None the angle in the gate is used (default)
    :return: set of gates wrapped in QCircuit class
    """

    # for the case that gate is actually a whole circuit
    if hasattr(gate, "gates"):
        result = QCircuit()
        for g in gate.gates:
            result *= compile_controlled_rotation_gate(gate=g, angles=angles)
        return result

    if gate.control is None:
        return QCircuit.wrap_gate(gate)

    if angles is None:
        angles = [gate.angle / 2.0, -gate.angle / 2.0]

    assert (len(angles) == 2)

    if len(gate.target) > 1:
        result = QCircuit()
        return compile_controlled_rotation_gate(gate=compile_multitarget(gate=g), angles=angles)

    target = gate.target
    control = gate.control

    result = QCircuit()
    result *= change_basis(target=target, axis=gate._axis)
    result *= RotationGateImpl(axis="z", target=target, angle=angles[0])
    result *= QGateImpl(name="X", target=target, control=control)
    result *= RotationGateImpl(axis="Z", target=target, angle=angles[1])
    result *= QGateImpl(name="X", target=target, control=control)
    result *= change_basis(target=target, axis=gate._axis, daggered=True)

    result.n_qubits = result.max_qubit() + 1
    return result
