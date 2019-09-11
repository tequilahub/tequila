"""
Primitive Compiler from Qubit-Operators to evolution operators
Replace with fancier external packages at some point
"""

from openvqe.circuit.circuit import QCircuit
import numpy
from openfermion import QubitOperator
from openvqe.circuit.gates import Rx, H, CNOT, Rz
from openvqe.circuit._gates_impl import RotationGateImpl, QGateImpl
import copy


def compile_trotter_evolution(cluster_operator: QubitOperator, steps: int = 1, anti_hermitian=True) -> QCircuit:
    circuit = QCircuit()
    factor = 1.0 / steps
    if anti_hermitian:
        factor = 1.0j / steps
    for index in range(steps):
        for key, value in cluster_operator.terms.items():
            if key == ():
                # dont implement the constant part
                continue
            elif not numpy.isclose(value, 0.0, rtol=1.e-8, atol=1.e-8):
                # don;t make circuit for too small values
                # @todo include ampltidude_neglect_threshold into parameters
                circuit *= exponential_pauli_gate(paulistring=key, angle=value * factor)
    return circuit


def exponential_pauli_gate(paulistring, angle) -> QCircuit:
    """
    Returns the circuit: exp(i*angle*paulistring)
    primitively compiled into X,Y Basis Changes and CNOTs and Z Rotations
    :param paulistring: The paulistring in given as tuple of tuples (openfermion format)
    like e.g  ( (0, 'Y'), (1, 'X'), (5, 'Z') )
    :param angle: The angle which parametrizes the gate -> should be real
    :returns: the above mentioned circuit as abstract structure
    """

    if not numpy.isclose(numpy.imag(angle), 0.0):
        raise Warning("angle is not real, angle=" + str(angle))

    circuit = QCircuit()

    # the general circuit will look like:
    # series which changes the basis if necessary
    # series of CNOTS associated with basis changes
    # Rz gate parametrized on the angle
    # series of CNOT (inverted direction compared to before)
    # series which changes the basis back
    ubasis = QCircuit()
    ubasis_t = QCircuit()
    cnot_cascade = QCircuit()
    reversed_cnot = QCircuit()

    last_qubit = None
    previous_qubit = None
    for pq in paulistring:
        pauli = pq[1]
        qubit = [pq[0]]  # wrap in list for targets= ...

        # see if we need to change the basis
        axis = 2
        if pauli.upper()=="X":
            axis =0
        elif pauli.upper()=="Y":
            axis=1
        ubasis *= change_basis(target=qubit, axis=axis)
        ubasis_t *= change_basis(target=qubit, axis=axis, daggered=True)

        if previous_qubit is not None:
            cnot_cascade *= CNOT(target=qubit, control=previous_qubit)
        previous_qubit = qubit
        last_qubit = qubit

    reversed_cnot = cnot_cascade.dagger()

    # assemble the circuit
    circuit *= ubasis
    circuit *= cnot_cascade
    # factor 2 is since gates are defined with angle/2
    circuit *= Rz(target=last_qubit, angle=2.0 * angle)
    circuit *= reversed_cnot
    circuit *= ubasis_t

    return circuit


def compile_multitarget(gate) -> QCircuit:
    # for the case that gate is actually a whole circuit
    if hasattr(gate, "gates"):
        result = QCircuit()
        for g in gate.gates:
            result *= compile_multitarget(gate=g)
        return result

    targets = g.target

    result = QCircuit()
    for t in targets:
        gx = copy.deepcopy(gate)
        gx.target = [t]
        result += gx

    return result


def change_basis(target, axis, daggered=False):
    if isinstance(axis, str):
        axis = RotationGateImpl.string_to_axis[axis.lower()]

    if axis == 0:
        return H(target=target, frozen=True)
    elif axis == 1 and daggered:
        return Rx(angle=-numpy.pi/2, target=target, frozen=True)
    elif axis == 1:
        return Rx(angle=numpy.pi/2, target=target, frozen=True)
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
        angles = [-gate.angle / 2.0, gate.angle / 2.0]

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

    result.n_qubits = result.max_qubit()
    return result
