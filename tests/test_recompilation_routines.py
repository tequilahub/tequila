import numpy as np

import tequila.simulators.simulator_api
from tequila.circuit import gates
from tequila.circuit._gates_impl import QGateImpl
from tequila.circuit.compiler import (
    compile_controlled_rotation,
    change_basis,
    compile_phase,
    compile_swap,
    compile_ry,
    compile_y,
    compile_ch,
    CircuitCompiler,
)
from numpy.random import uniform, randint
from numpy import pi, isclose

from tequila.circuit.gates import RotationGate
from tequila.hamiltonian import paulis
from tequila import simulators, QubitWaveFunction, compile_circuit, QCircuit
from tequila.simulators.simulator_api import simulate
from tequila.objective.objective import ExpectationValue
import pytest
import numpy

# Get QC backends for parametrized testing
import select_backends

simulators = select_backends.get()
samplers = select_backends.get(sampler=True)

PX = paulis.X
PY = paulis.Y
PZ = paulis.Z


@pytest.mark.parametrize("simulator", simulators)
@pytest.mark.parametrize("angle", numpy.random.uniform(0, 2 * numpy.pi, 1))
@pytest.mark.parametrize("axis", ["X", "Y", "Z"])
@pytest.mark.parametrize("control", [None, 1])
def test_exponential_pauli_wfn(simulator, angle, axis, control):
    U1 = gates.RotationGate(axis=axis, angle=angle, target=0, control=control)
    U2 = gates.ExpPauli(paulistring=axis + "(0)", angle=angle, control=control)

    wfn1 = simulate(U1, backend=simulator)
    wfn2 = simulate(U2, backend=simulator)
    wfn3 = simulate(U2, backend=None)

    assert isclose(numpy.abs(wfn1.inner(wfn2)) ** 2, 1.0, atol=1.0e-4)
    assert isclose(numpy.abs(wfn2.inner(wfn3)) ** 2, 1.0, atol=1.0e-4)


@pytest.mark.parametrize("simulator", simulators)
def test_controlled_rotations(simulator):
    angles = uniform(0, 2 * pi, 5)
    gs = [gates.Rx, gates.Ry, gates.Rz]
    for angle in angles:
        for gate in gs:
            qubit = randint(0, 1)
            control = randint(2, 3)
            U = gates.X(target=control) + gate(target=qubit, control=control, angle=angle)
            RCU = compile_controlled_rotation(gate=U)
            wfn1 = simulate(U, initial_state=0, backend=simulator)
            wfn2 = simulate(RCU, initial_state=0, backend=simulator)
            assert isclose(numpy.abs(wfn1.inner(wfn2)) ** 2, 1.0, atol=1.0e-4)


@pytest.mark.parametrize("simulator", simulators)
def test_basis_change(simulator):
    for angle in list(uniform(0, 2 * pi, 5)):
        EX = simulate(ExpectationValue(U=gates.Rx(target=0, angle=angle), H=PX(0)), backend=simulator)
        EY = simulate(ExpectationValue(U=gates.Rx(target=0, angle=angle), H=PY(0)), backend=simulator)
        EZ = simulate(ExpectationValue(U=gates.Rx(target=0, angle=angle), H=PZ(0)), backend=simulator)

        EXX = simulate(
            ExpectationValue(U=gates.Rx(target=0, angle=angle) + change_basis(target=0, axis=0), H=PZ(0)),
            backend=simulator,
        )
        EYY = simulate(
            ExpectationValue(U=gates.Rx(target=0, angle=angle) + change_basis(target=0, axis=1), H=PZ(0)),
            backend=simulator,
        )
        EZZ = simulate(
            ExpectationValue(U=gates.Rx(target=0, angle=angle) + change_basis(target=0, axis=2), H=PZ(0)),
            backend=simulator,
        )

        assert isclose(EX, EXX, atol=1.0e-4)
        assert isclose(EY, EYY, atol=1.0e-4)
        assert isclose(EZ, EZZ, atol=1.0e-4)

    for i, gate in enumerate([gates.Rx, gates.Ry, gates.Rz]):
        angle = uniform(0, 2 * pi)
        U1 = gate(target=0, angle=angle)
        U2 = (
            change_basis(target=0, axis=i)
            + gates.Rz(target=0, angle=angle)
            + change_basis(target=0, axis=i, daggered=True)
        )
        wfn1 = simulate(U1, backend=simulator)
        wfn2 = simulate(U2, backend=simulator)
        assert isclose(numpy.abs(wfn1.inner(wfn2)) ** 2, 1.0, atol=1.0e-4)

        if simulator == "qiskit":
            return  # initial state not yet supported
        wfn1 = simulate(U1, initial_state=1, backend=simulator)
        wfn2 = simulate(U2, initial_state=1, backend=simulator)
        assert isclose(numpy.abs(wfn1.inner(wfn2)) ** 2, 1.0, atol=1.0e-4)


def test_compile_swap():
    circuit = gates.SWAP(first=0, second=3)
    equivalent_circuit = compile_swap(circuit)

    equivalent_swap = gates.X(target=0, control=3) + gates.X(target=3, control=0) + gates.X(target=0, control=3)

    assert equivalent_circuit == equivalent_swap


@pytest.mark.parametrize(
    "target,control,angle", [(2, 4, 3.14), (1, 0, numpy.pi / 7), (1, None, numpy.pi / 5), (5, None, 1.093)]
)
def test_compile_ry(target, control, angle):
    circuit = gates.Ry(target=target, control=control, angle=angle)
    equivalent_circuit = compile_ry(circuit)

    equivalent_ry = (
        gates.Rz(target=target, control=None, angle=-numpy.pi / 2)
        + gates.Rx(target=target, control=control, angle=angle)
        + gates.Rz(target=target, control=None, angle=numpy.pi / 2)
    )

    assert equivalent_circuit == equivalent_ry


@pytest.mark.parametrize(
    "target,control,power", [(2, 4, 1.5), (4, 0, 1.0), (0, 5, 2.9), (1, None, 4.2), (5, None, 3.9)]
)
def test_compile_y(target, control, power):
    circuit = gates.Y(target=target, control=control, power=power)
    equivalent_circuit_y = compile_y(circuit)

    equivalent_y = (
        gates.Rz(target=target, control=None, angle=-numpy.pi / 2)
        + gates.X(target=target, control=control, power=power)
        + gates.Rz(target=target, control=None, angle=numpy.pi / 2)
    )

    assert equivalent_circuit_y == equivalent_y


@pytest.mark.parametrize(
    "target,control,power", [(2, 4, 1.5), (4, 2, 1.0), (0, 5, 2.9), (1, None, 4.2), (5, None, 3.9)]
)
def test_compile_ch(target, control, power):
    circuit = gates.H(target=target, control=control, power=power)
    equivalent_circuit = compile_ch(circuit)

    equivalent_ch = (
        gates.Ry(target=target, control=None, angle=-numpy.pi / 4)
        + gates.Z(target=target, control=control, power=power)
        + gates.Ry(target=target, control=None, angle=numpy.pi / 4)
    )

    if control is not None:
        assert equivalent_circuit == equivalent_ch


@pytest.mark.parametrize("type", [gates.Rx, gates.Ry, gates.Rz])
@pytest.mark.parametrize("angle", np.linspace(0, 2 * np.pi, 10))
def test_compile_pauli_rotations(type, angle: float):
    gate = type(target=0, angle=angle)
    compiler = CircuitCompiler(pauli_rotations=True, epsilon=1e-6)
    compiled = compiler.compile_circuit(gate)
    assert np.allclose(gate.to_matrix(), compiled.to_matrix(), atol=1e-6)


# Check if the gate is in the set {H, X, S, CNOT, T}
def is_error_correctable_gate(gate: QGateImpl) -> bool:
    if len(gate.control) > 1:
        return False

    if len(gate.control) == 1:
        return gate.name.lower() == "x"

    if gate.name.lower() in ["x", "h", "globalphase"]:
        return True

    if (gate.name.lower() == "rz" or gate.name.lower() == "phase") and (
        isclose(gate.parameter, pi / 2) or isclose(gate.parameter, pi / 4)
    ):
        return True

    return False


def test_error_correctable_compilation():
    U = QCircuit()
    U += gates.Ry(target=1, angle=0.4)
    U += gates.Toffoli(first=0, second=1, target=2)
    U += gates.H(target=2, power=2.5, control=0)
    U += gates.X(target=0)
    U += gates.Rz(target=0, angle=1.2, control=1)
    U += gates.CNOT(control=2, target=0)

    compiler = CircuitCompiler.error_correctable_gate_set(1e-6)
    compiled = compiler.compile_circuit(U)

    assert all(is_error_correctable_gate(g) for g in compiled.gates)
    assert np.allclose(U.to_matrix(), compiled.to_matrix(), atol=1e-5)


def test_compile_qubit_excitations():
    # Test 3-qubit excitation

    q = [5, 3, 7, 8, 2, 9, 2, 4]

    state_circ = gates.X([q[0], q[1], q[2]])

    # optimized decomposition
    circuit = state_circ + gates.QubitExcitation(angle=numpy.pi / 2, target=[q[0], q[3], q[1], q[4], q[2], q[5]])
    U1 = compile_circuit(circuit)

    # non-optimized decomposition
    circuit = state_circ + gates.QubitExcitation(
        angle=numpy.pi / 2,
        target=[q[0], q[3], q[1], q[4], q[2], q[5]],
        compile_options="pauli",
    )
    U2 = compile_circuit(circuit)
    wfn1 = simulate(U1)
    wfn2 = simulate(U2)

    assert U1.depth < U2.depth
    assert isclose(numpy.abs(wfn1.inner(wfn2)) ** 2, 1.0)

    # Test 5-qubit excitation with a random state

    q = list(range(10))
    state = numpy.random.randn(2**10) + 1j * numpy.random.randn(2**10)
    state = QubitWaveFunction.from_array(state).normalize()
    angle = numpy.pi * numpy.random.randn()

    U1 = compile_circuit(gates.QubitExcitation(angle=angle, target=q))  # optimized
    U2 = compile_circuit(gates.QubitExcitation(angle=angle, target=q, compile_options="pauli"))

    wfn1 = simulate(U1, initial_state=state)
    wfn2 = simulate(U2, initial_state=state)

    assert U1.depth < U2.depth
    assert isclose(numpy.abs(wfn1.inner(wfn2)) ** 2, 1.0)
