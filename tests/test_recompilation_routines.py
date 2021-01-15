import tequila.simulators.simulator_api
from tequila.circuit import gates
from tequila.circuit.compiler import compile_controlled_rotation, change_basis, compile_phase, compile_swap, \
                                     compile_ry, compile_y, compile_ch
from numpy.random import uniform, randint
from numpy import pi, isclose
from tequila.hamiltonian import paulis
from tequila import simulators
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
@pytest.mark.parametrize('angle', numpy.random.uniform(0, 2 * numpy.pi, 1))
@pytest.mark.parametrize('axis', ['X', 'Y', 'Z'])
@pytest.mark.parametrize('control', [None, 1])
def test_exponential_pauli_wfn(simulator, angle, axis, control):
    U1 = gates.RotationGate(axis=axis, angle=angle, target=0, control=control)
    U2 = gates.ExpPauli(paulistring=axis + "(0)", angle=angle, control=control)

    wfn1 = simulate(U1, backend=simulator)
    wfn2 = simulate(U2, backend=simulator)
    wfn3 = simulate(U2, backend=None)

    assert (isclose(numpy.abs(wfn1.inner(wfn2))**2, 1.0, atol=1.e-4))
    assert (isclose(numpy.abs(wfn2.inner(wfn3))**2, 1.0, atol=1.e-4))

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
            assert (isclose(numpy.abs(wfn1.inner(wfn2)) ** 2, 1.0, atol=1.e-4))

@pytest.mark.parametrize("simulator", simulators)
def test_basis_change(simulator):
    for angle in list(uniform(0, 2 * pi, 5)):
        EX = simulate(ExpectationValue(U=gates.Rx(target=0, angle=angle), H=PX(0)), backend=simulator)
        EY = simulate(ExpectationValue(U=gates.Rx(target=0, angle=angle), H=PY(0)), backend=simulator)
        EZ = simulate(ExpectationValue(U=gates.Rx(target=0, angle=angle), H=PZ(0)), backend=simulator)

        EXX = simulate(ExpectationValue(U=gates.Rx(target=0, angle=angle) + change_basis(target=0, axis=0),
                                   H=PZ(0)), backend=simulator)
        EYY = simulate(ExpectationValue(U=gates.Rx(target=0, angle=angle) + change_basis(target=0, axis=1),
                                   H=PZ(0)), backend=simulator)
        EZZ = simulate(ExpectationValue(U=gates.Rx(target=0, angle=angle) + change_basis(target=0, axis=2),
                                   H=PZ(0)), backend=simulator)

        assert (isclose(EX, EXX, atol=1.e-4))
        assert (isclose(EY, EYY, atol=1.e-4))
        assert (isclose(EZ, EZZ, atol=1.e-4))

    for i, gate in enumerate([gates.Rx, gates.Ry, gates.Rz]):
        angle = uniform(0, 2 * pi)
        U1 = gate(target=0, angle=angle)
        U2 = change_basis(target=0, axis=i) + gates.Rz(target=0, angle=angle) + change_basis(target=0, axis=i,
                                                                                             daggered=True)
        wfn1 = simulate(U1, backend=simulator)
        wfn2 = simulate(U2, backend=simulator)
        assert (isclose(numpy.abs(wfn1.inner(wfn2)) ** 2, 1.0, atol=1.e-4))

        if simulator == "qiskit":
            return # initial state not yet supported
        wfn1 = simulate(U1, initial_state=1, backend=simulator)
        wfn2 = simulate(U2, initial_state=1, backend=simulator)
        assert (isclose(numpy.abs(wfn1.inner(wfn2)) ** 2, 1.0, atol=1.e-4))


def test_compile_swap():
    circuit = gates.SWAP(first=0, second=3)
    equivalent_circuit = compile_swap(circuit)

    equivalent_swap = gates.X(target=0, control=3) + gates.X(target=3, control=0) + gates.X(target=0, control=3)

    assert (equivalent_circuit == equivalent_swap)


@pytest.mark.parametrize(
    "target,control,angle",
    [
        (2, 4, 3.14),
        (1, 0, numpy.pi / 7),
        (1, None, numpy.pi / 5),
        (5, None, 1.093)
    ]
)
def test_compile_ry(target, control, angle):

    circuit = gates.Ry(target=target, control=control, angle=angle)
    equivalent_circuit = compile_ry(circuit)

    equivalent_ry = gates.Rz(target=target, control=None, angle=-numpy.pi / 2) + \
                    gates.Rx(target=target, control=control, angle=angle) + \
                    gates.Rz(target=target, control=None, angle=numpy.pi / 2)

    assert (equivalent_circuit == equivalent_ry)


@pytest.mark.parametrize(
    "target,control,power",
    [
        (2, 4, 1.5),
        (4, 0, 1.0),
        (0, 5, 2.9),
        (1, None, 4.2),
        (5, None, 3.9)
    ]
)
def test_compile_y(target, control, power):

    circuit = gates.Y(target=target, control=control, power=power)
    equivalent_circuit_y = compile_y(circuit)

    equivalent_y = gates.Rz(target=target, control=None, angle=-numpy.pi / 2) + \
                   gates.X(target=target, control=control, power=power) + \
                   gates.Rz(target=target, control=None, angle=numpy.pi / 2)

    assert (equivalent_circuit_y == equivalent_y)


@pytest.mark.parametrize(
    "target,control,power",
    [
        (2, 4, 1.5),
        (4, 2, 1.0),
        (0, 5, 2.9),
        (1, None, 4.2),
        (5, None, 3.9)
    ]
)
def test_compile_ch(target, control, power):

    circuit = gates.H(target=target, control=control, power=power)
    equivalent_circuit = compile_ch(circuit)

    equivalent_ch = gates.Ry(target=target, control=None, angle=-numpy.pi / 4) + \
                    gates.Z(target=target, control=control, power=power) + \
                    gates.Ry(target=target, control=None, angle=numpy.pi / 4)

    if control is not None:
        assert (equivalent_circuit == equivalent_ch)

