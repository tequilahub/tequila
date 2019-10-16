from openvqe.simulator.simulator_cirq import SimulatorCirq
from openvqe.circuit.circuit import QCircuit
from openvqe.circuit.gates import X, Y, Z, Rx, Ry, Rz, SWAP, H
from numpy import pi, random, isclose, sqrt
from openvqe.hamiltonian import PauliString
from openvqe.objective import Objective
from openvqe.circuit.gradient import grad

# Note
# multi control works
# multi target does not

supported_primitive_gates = [X, Y, Z, H]
supported_two_qubit_gates = [SWAP]
supported_rotations = [Rx, Ry, Rz]
supported_powers = (X, Y, Z, H)

def test_measurment():
    ps=[]
    data = {0:'x'}
    ps.append(PauliString(data=data, coeff=-1.2345))
    simulator = SimulatorCirq()
    ac = H(target=0)
    result = simulator.measure_paulistrings(abstract_circuit=ac, paulistrings=ps, samples=1)[0]
    assert(isclose(result,-1.2345))

    ps=[]
    data = {0:'Z'}
    ps.append(PauliString(data=data, coeff=-1.2345))
    simulator = SimulatorCirq()
    ac = QCircuit()
    result = simulator.measure_paulistrings(abstract_circuit=ac, paulistrings=ps, samples=1)[0]
    assert(isclose(result,-1.2345))

    ps=[]
    data = {0:'x', 1:'z', 2:'z'}
    ps.append(PauliString(data=data, coeff=-1.2345))
    simulator = SimulatorCirq()
    ac = H(target=0)
    result = simulator.measure_paulistrings(abstract_circuit=ac, paulistrings=ps, samples=2)[0]
    assert(isclose(result,-1.2345))

    ps=[]
    data = {0:'x', 1:'z', 2:'x'}
    ps.append(PauliString(data=data, coeff=-1.2345))
    simulator = SimulatorCirq()
    ac = H(target=0)*X(target=1)*X(target=1)*H(target=2)
    result = simulator.measure_paulistrings(abstract_circuit=ac, paulistrings=ps, samples=5)[0]
    assert(isclose(result,-1.2345))

    ps=[]
    data = {0:'z', 1:'z', 2:'z'}
    ps.append(PauliString(data=data, coeff=-1.2345))
    simulator = SimulatorCirq()
    ac = X(target=0)*X(target=1)*X(target=2)
    result = simulator.measure_paulistrings(abstract_circuit=ac, paulistrings=ps, samples=5)[0]
    assert(isclose(result,1.2345))


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

    dU1 = Ry(target=0, angle=pi / 2 + pi/2)
    dU1.weight = 0.5
    dU2 = Ry(target=0, angle=pi / 2 - pi/2)
    dU2.weight = -0.5
    O = Objective(observable=hamiltonian, unitaries=[dU1, dU2])
    dE = simulator.simulate_objective(objective=O)
    assert (isclose(dE, 0.0))

    U = Ry(target=0, angle=pi / 2)
    dU = grad(U)
    dU[0].observable = hamiltonian
    dEx = simulator.simulate_objective(objective=dU[0])
    assert (isclose(dEx, dE))
