from openvqe.simulator.simulator_cirq import SimulatorCirq
from openvqe.circuit.circuit import QCircuit
from openvqe.circuit.gates import X, Y, Z, Rx, Ry, Rz, CNOT, SWAP, H
from numpy import pi, random, isclose, sqrt
from openvqe.hamiltonian import HamiltonianBase
from openfermion import QubitOperator
from openvqe.objective import Objective
from openvqe.circuit.gradient import grad

# Note
# multi control works
# multi target does not

supported_primitive_gates = [X, Y, Z, H]
supported_two_qubit_gates = [SWAP]
supported_rotations = [Rx, Ry, Rz]
supported_powers = (X, Y, Z, H)


def test_simple_execution():
    ac = QCircuit()
    ac *= X(0)
    ac *= Ry(target=1, control=0, angle=pi / 2)

    simulator = SimulatorCirq()

    result = simulator.simulate_wavefunction(abstract_circuit=ac, initial_state=0)


def test_primitive_gates():
    for g in supported_primitive_gates:
        qubit = random.randint(0, 10)
        incr = random.randint(1, 5)
        init = random.randint(0, 1)
        result = SimulatorCirq().simulate_wavefunction(abstract_circuit=g(target=qubit), initial_state=init)
        result = SimulatorCirq().simulate_wavefunction(abstract_circuit=g(target=qubit, control=qubit + incr),
                                                       initial_state=init)
        controls = [11, 12, 13]
        result = SimulatorCirq().simulate_wavefunction(abstract_circuit=g(target=qubit, control=controls),
                                                       initial_state=init)


def test_primitive_two_qubit_gates():
    for g in supported_two_qubit_gates:
        qubit = random.randint(0, 10)
        qubits = [qubit, qubit + 3]
        init = random.randint(0, 1)
        result = SimulatorCirq().simulate_wavefunction(abstract_circuit=g(target=qubits), initial_state=init)
        result = SimulatorCirq().simulate_wavefunction(abstract_circuit=g(target=qubits, control=qubit + 2),
                                                       initial_state=init)


def test_rotations():
    for g in supported_rotations:
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


def test_power_gates():
    for g in supported_powers:
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


class MyQubitHamiltonian(HamiltonianBase):

    # Hamiltonian needs to be aware of this
    def n_qubits(self):
        return 1

    def my_trafo(self, H):
        """
        Here we define the hamiltonian to be already in qubit form, so no transformation will be needed
        """
        return H

    def get_fermionic_hamiltonian(self):
        H = QubitOperator()
        H.terms[((0, 'X'),)] = 1.0
        return H


def test_expectation_values():
    hamiltonian = MyQubitHamiltonian()
    hamiltonian.parameters.transformation = "my_trafo"

    U = Ry(target=0, angle=pi / 4)

    simulator = SimulatorCirq()

    state = simulator.simulate_wavefunction(abstract_circuit=U)

    O = Objective(observable=hamiltonian, unitaries=U)

    E = simulator.expectation_value(objective=O)
    assert (isclose(E, 1.0 / sqrt(2)))

    U1 = X(0)
    U2 = Y(0)
    O = Objective(observable=hamiltonian, unitaries=[U1, U2])
    E = simulator.expectation_value(objective=O)
    assert (isclose(E, 0.0))

    dU1 = Ry(target=0, angle=pi / 4 + pi)
    dU1.weight = 0.5
    dU2 = Ry(target=0, angle=pi / 4 - pi)
    dU2.weight = -0.5
    O = Objective(observable=hamiltonian, unitaries=[dU1, dU2])
    dE = simulator.expectation_value(objective=O)
    assert (isclose(dE, 0.0))

    U = Ry(target=0, angle=pi / 4)
    dU = grad(U)
    dU[0].observable = hamiltonian
    dEx = simulator.expectation_value(objective=dU[0])
    assert (isclose(dEx, dE))
