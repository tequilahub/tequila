system_has_pyquil = True
try:
    from openvqe.simulator.simulator_pyquil import SimulatorPyquil
    system_has_pyquil = True
except ImportError:
    system_has_pyquil = False

from openvqe.circuit.circuit import QCircuit
from openvqe.circuit.gates import X, Y, Z, Rx, Ry, Rz, CNOT, SWAP, H
from numpy import pi, random

import pytest
from shutil import which
system_has_qvm = which("qvm") is not None


supported_primitive_gates = [X, Y, Z, H]
supported_two_qubit_gates = []
supported_rotations = [Rx, Ry, Rz]
supported_powers = []

@pytest.mark.skipif(condition=not system_has_qvm, reason="no qvm found")
@pytest.mark.skipif(condition=not system_has_pyquil, reason="pyquil not found")
def test_simple_execution():
    ac = QCircuit()
    ac *= X(0)
    ac *= Ry(target=1, control=0, angle=pi / 2)

    simulator = SimulatorPyquil()

    result = simulator.simulate_wavefunction(abstract_circuit=ac, initial_state=0)


@pytest.mark.skipif(condition=not system_has_qvm, reason="no qvm found")
@pytest.mark.skipif(condition=not system_has_pyquil, reason="pyquil not found")
@pytest.mark.parametrize("g", supported_primitive_gates)
def test_primitive_gates(g):
    qubit = random.randint(0, 10)
    incr = random.randint(1, 5)
    init = random.randint(0, 1)
    result = SimulatorPyquil().simulate_wavefunction(abstract_circuit=g(target=qubit), initial_state=init)
    result = SimulatorPyquil().simulate_wavefunction(abstract_circuit=g(target=qubit, control=qubit + incr),
                                                     initial_state=init)
    controls = [11, 13]
    targets = qubit
    result = SimulatorPyquil().simulate_wavefunction(abstract_circuit=g(target=targets, control=controls),
                                                     initial_state=init)


@pytest.mark.skipif(condition=not system_has_qvm, reason="no qvm found")
@pytest.mark.skipif(condition=not system_has_pyquil, reason="pyquil not found")
@pytest.mark.parametrize("g", supported_two_qubit_gates)
def test_primitive_two_qubit_gates(g):
    qubit = random.randint(0, 10)
    qubits = [qubit, qubit + 3]
    init = random.randint(0, 1)
    result = SimulatorPyquil().simulate_wavefunction(abstract_circuit=g(target=qubits), initial_state=init)
    result = SimulatorPyquil().simulate_wavefunction(abstract_circuit=g(target=qubits, control=qubit + 2),
                                                     initial_state=init)


@pytest.mark.skipif(condition=not system_has_qvm, reason="no qvm found")
@pytest.mark.skipif(condition=not system_has_pyquil, reason="pyquil not found")
@pytest.mark.parametrize("g", supported_rotations)
def test_rotations(g):
    for g in supported_rotations:
        qubit = random.randint(0, 10)
        incr = random.randint(1, 5)
        init = random.randint(0, 1)
        angle = random.uniform(0, 2 * pi)
        result = SimulatorPyquil().simulate_wavefunction(abstract_circuit=g(target=qubit, angle=angle),
                                                         initial_state=init)
        result = SimulatorPyquil().simulate_wavefunction(
            abstract_circuit=g(target=qubit, control=qubit + incr, angle=angle), initial_state=init)
        controls = [11, 13]
        targets = qubit
        result = SimulatorPyquil().simulate_wavefunction(
            abstract_circuit=g(target=targets, control=controls, angle=angle), initial_state=init)


@pytest.mark.skipif(condition=not system_has_qvm, reason="no qvm found")
@pytest.mark.skipif(condition=not system_has_pyquil, reason="pyquil not found")
@pytest.mark.parametrize("g", supported_powers)
def test_power_gates(g):
    qubit = random.randint(0, 10)
    incr = random.randint(1, 5)
    init = random.randint(0, 1)
    power = random.uniform(0, 2 * pi)
    result = SimulatorPyquil().simulate_wavefunction(abstract_circuit=g(target=qubit, power=power),
                                                     initial_state=init)
    result = SimulatorPyquil().simulate_wavefunction(
        abstract_circuit=g(target=qubit, control=qubit + incr, power=power), initial_state=init)
    controls = [11, 13]
    targets = qubit
    result = SimulatorPyquil().simulate_wavefunction(
        abstract_circuit=g(target=targets, control=controls, power=power), initial_state=init)
