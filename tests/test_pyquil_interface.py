from openvqe.simulator.simulator_pyquil import SimulatorPyquil
from openvqe.circuit.circuit import QCircuit
from openvqe.circuit.gates import X, Y, Z, Rx, Ry, Rz, CNOT, SWAP, H
from numpy import pi, random

# Note
# multi control works
# multi target does not

supported_primitive_gates = [X, Y, Z, H]
supported_two_qubit_gates = []
supported_rotations = [Rx, Ry, Rz]
supported_powers = []


def test_simple_execution():
    ac = QCircuit()
    ac *= X(0)
    ac *= Ry(target=1, control=0, angle=pi / 2)

    simulator = SimulatorPyquil()

    result = simulator.simulate_wavefunction(abstract_circuit=ac, initial_state=0)


def test_primitive_gates():
    for g in supported_primitive_gates:
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


def test_primitive_two_qubit_gates():
    for g in supported_two_qubit_gates:
        qubit = random.randint(0, 10)
        qubits = [qubit, qubit + 3]
        init = random.randint(0, 1)
        result = SimulatorPyquil().simulate_wavefunction(abstract_circuit=g(target=qubits), initial_state=init)
        result = SimulatorPyquil().simulate_wavefunction(abstract_circuit=g(target=qubits, control=qubit + 2),
                                                         initial_state=init)


def test_rotations():
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
        targets =  qubit
        result = SimulatorPyquil().simulate_wavefunction(
            abstract_circuit=g(target=targets, control=controls, angle=angle), initial_state=init)


def test_power_gates():
    for g in supported_powers:
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
