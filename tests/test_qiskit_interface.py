from openvqe.simulator.simulator_qiskit import SimulatorQiskit
from openvqe.circuit import gates
from openvqe.circuit.circuit import QCircuit
from openvqe.circuit.gates import X, Y, Z, Rx, Ry, Rz, CNOT, SWAP, H
from numpy import pi, random, isclose, sqrt

# Note
# multi controls do not work
# single and double controls work for some gates
# -- ccx, ccz work, ccy is not there (need automatic recompilation at some point)
# multi target does not

supported_primitive_gates = [gates.X, gates.Y, gates.Z, gates.H]
supported_controlled_gates = [gates.X, gates.Z, gates.H]
supported_two_qubit_gates = [None]
supported_rotations = [gates.Rx, gates.Ry, gates.Rz]
supported_powers = (gates.X, gates.Y, gates.Z, gates.H)


def test_simple_execution():
    ac = QCircuit()
    ac *= gates.X(0)
    ac *= gates.Ry(target=1, control=0, angle=pi / 2)

    simulator = SimulatorQiskit()
    simulator.run(abstract_circuit=ac, samples=2)


def test_primitive_gates():
    for g in supported_primitive_gates:
        qubit = random.randint(0, 10)
        incr = random.randint(1, 5)
        SimulatorQiskit().run(abstract_circuit=g(target=qubit))
        SimulatorQiskit().run(abstract_circuit=g(target=qubit, control=qubit + incr))
        if g(0).gates[0].name in supported_controlled_gates:
            controls = [11]
            SimulatorQiskit().run(abstract_circuit=g(target=qubit, control=controls))

            controls = [11, 12]
            SimulatorQiskit().run(abstract_circuit=g(target=qubit, control=controls))


def test_bell_state():
    for i in range(10):
        c = gates.H(target=0) * gates.CNOT(target=1, control=0) * gates.Measurement(target=[0, 1])
        result = SimulatorQiskit().run(abstract_circuit=c, samples=1)
        assert (len(result.measurements._result) ==1)
        keys=[k for k in result.measurements._result.keys()]
        assert (len(keys)==1)
        assert (keys[0] in [0,3])

def test_notation():
    c = gates.X(target=0)*gates.Measurement(target=0)
    result = SimulatorQiskit().run(abstract_circuit=c, samples=1)

    assert (len(result.measurements._result) == 1)
    keys = [k for k in result.measurements._result.keys()]
    assert (len(keys) == 1)
    assert (keys[0] == 1)

    c = gates.X(target=0)*gates.Measurement(target=[0,1])
    result = SimulatorQiskit().run(abstract_circuit=c, samples=1)

    assert (len(result.measurements._result) == 1)
    keys = [k for k in result.measurements._result.keys()]
    assert (len(keys) == 1)
    assert (keys[0] == 1)
