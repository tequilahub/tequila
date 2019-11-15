system_has_cirq = None
try:
    from openvqe.simulators.simulator_cirq import SimulatorCirq

    system_has_cirq = True
except ImportError:
    system_has_cirq = False

from openvqe.circuit.circuit import QCircuit
from openvqe.circuit.gates import X, Y, Z, Rx, Ry, Rz, SWAP, H, iSWAP
from numpy import pi, random, isclose, sqrt
from openvqe.hamiltonian import PauliString
from openvqe.objective import Objective
from openvqe.circuit.gradient import grad
from openvqe.circuit import Variable
import pytest

supported_primitive_gates = [X, Y, Z, H]
supported_two_qubit_gates = [SWAP, iSWAP]
supported_rotations = [Rx, Ry, Rz]
supported_powers = (X, Y, Z, H)


@pytest.mark.skipif(condition=not system_has_cirq, reason="cirq not found")
def test_simple_execution():
    ac = QCircuit()
    ac *= X(0)
    ac *= Ry(target=1, control=0, angle=pi / 2)

    simulator = SimulatorCirq()

    result = simulator.simulate_wavefunction(abstract_circuit=ac, initial_state=0)


@pytest.mark.skipif(condition=not system_has_cirq, reason="cirq not found")
@pytest.mark.parametrize("g", supported_primitive_gates)
def test_primitive_gates(g):
    qubit = random.randint(0, 10)
    incr = random.randint(1, 5)
    init = random.randint(0, 1)
    result = SimulatorCirq().simulate_wavefunction(abstract_circuit=g(target=qubit), initial_state=init)
    result = SimulatorCirq().simulate_wavefunction(abstract_circuit=g(target=qubit, control=qubit + incr),
                                                   initial_state=init)
    controls = [11, 12, 13]
    result = SimulatorCirq().simulate_wavefunction(abstract_circuit=g(target=qubit, control=controls),
                                                   initial_state=init)


@pytest.mark.skipif(condition=not system_has_cirq, reason="cirq not found")
@pytest.mark.parametrize("g", supported_two_qubit_gates)
def test_two_qubit_gates(g):
    init = random.randint(0, 1)
    result = SimulatorCirq().simulate_wavefunction(abstract_circuit=g(2,1), initial_state=init)
    result = SimulatorCirq().simulate_wavefunction(abstract_circuit=g(1,2, control=3),
                                                   initial_state=init)


@pytest.mark.skipif(condition=not system_has_cirq, reason="cirq not found")
@pytest.mark.parametrize("g", supported_rotations)
def test_rotations(g):
    qubit = random.randint(0, 10)
    incr = random.randint(1, 5)
    init = random.randint(0, 1)
    angle = random.uniform(0, 2 * pi)
    result = SimulatorCirq().simulate_wavefunction(abstract_circuit=g(target=qubit, angle=angle),
                                                   initial_state=init)
    result = SimulatorCirq().simulate_wavefunction(
        abstract_circuit=g(target=qubit, control=qubit + incr, angle=angle), initial_state=init)
    controls = [11, 12, 13]
    result = SimulatorCirq().simulate_wavefunction(abstract_circuit=g(target=qubit, control=controls, angle=angle),
                                                   initial_state=init)


@pytest.mark.skipif(condition=not system_has_cirq, reason="cirq not found")
@pytest.mark.parametrize("g", supported_powers)
def test_power_gates(g):
    qubit = random.randint(0, 10)
    incr = random.randint(1, 5)
    init = random.randint(0, 1)
    power = random.uniform(0, 2 * pi)
    result = SimulatorCirq().simulate_wavefunction(abstract_circuit=g(target=qubit, power=power),
                                                   initial_state=init)
    result = SimulatorCirq().simulate_wavefunction(
        abstract_circuit=g(target=qubit, control=qubit + incr, power=power), initial_state=init)
    controls = [11, 12, 13]
    result = SimulatorCirq().simulate_wavefunction(abstract_circuit=g(target=qubit, control=controls, power=power),
                                                   initial_state=init)


@pytest.mark.skipif(condition=not system_has_cirq, reason="cirq not found")
def test_expectation_values():
    from openvqe.hamiltonian.paulis import X as PX
    hamiltonian = PX(qubit=0)

    U = Ry(target=0, angle=pi / 4)

    simulator = SimulatorCirq()

    state = simulator.simulate_wavefunction(abstract_circuit=U)

    O = Objective(observable=hamiltonian, unitaries=U)

    E = simulator.simulate_objective(objective=O)
    assert (isclose(E, 1.0 / sqrt(2)))

    U1 = X(0)
    U2 = Y(0)
    O = Objective(observable=hamiltonian, unitaries=[U1, U2])
    E = simulator.simulate_objective(objective=O)
    assert (isclose(E, 0.0))

    dU1 = Ry(target=0, angle=pi / 2 + pi / 2)
    dU1.weight = 0.5
    dU2 = Ry(target=0, angle=pi / 2 - pi / 2)
    dU2.weight = -0.5
    O = Objective(observable=hamiltonian, unitaries=[dU1, dU2])
    dE = simulator.simulate_objective(objective=O)
    assert (isclose(dE, 0.0))

    U = Ry(target=0, angle=Variable(name="angle",value=pi / 2))
    dU = grad(U)
    for k, v in dU.items():
        dU[k].observable = hamiltonian
        dEx = simulator.simulate_objective(objective=dU[k])
    assert (isclose(dEx, dE))
