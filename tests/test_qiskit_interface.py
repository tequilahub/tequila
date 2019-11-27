system_has_qiskit = None
try:
    from tequila.simulators.simulator_qiskit import SimulatorQiskit

    system_has_qiskit = True
except ImportError:
    system_has_qiskit = False
from tequila.circuit import gates
from tequila.circuit.circuit import QCircuit
from numpy import pi, random

import pytest

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


@pytest.mark.skipif(condition=not system_has_qiskit, reason="qiskit not found")
def test_simple_execution():
    ac = QCircuit()
    ac *= gates.X(0)
    ac *= gates.Ry(target=1, control=0, angle=pi / 2)

    simulator = SimulatorQiskit()
    simulator.run(abstract_circuit=ac, samples=2)


@pytest.mark.skipif(condition=not system_has_qiskit, reason="qiskit not found")
@pytest.mark.parametrize("g", supported_primitive_gates)
def test_primitive_gates(g):
    qubit = random.randint(0, 10)
    incr = random.randint(1, 5)
    SimulatorQiskit().run(abstract_circuit=g(target=qubit))
    SimulatorQiskit().run(abstract_circuit=g(target=qubit, control=qubit + incr))
    if g(0).gates[0].name in supported_controlled_gates:
        controls = [11]
        SimulatorQiskit().run(abstract_circuit=g(target=qubit, control=controls))

        controls = [11, 12]
        SimulatorQiskit().run(abstract_circuit=g(target=qubit, control=controls))


@pytest.mark.skipif(condition=not system_has_qiskit, reason="qiskit not found")
@pytest.mark.parametrize("i", range(10))
def test_bell_state(i):
    c = gates.H(target=0) * gates.CNOT(target=1, control=0) * gates.Measurement(target=[0, 1])
    result = SimulatorQiskit().run(abstract_circuit=c, samples=1)
    assert (len(result.measurements['']) == 1)
    keys = [k for k in result.measurements[''].keys()]
    assert (len(keys) == 1)
    assert (keys[0].integer in [0, 3])


@pytest.mark.skipif(condition=not system_has_qiskit, reason="qiskit not found")
def test_notation():
    c = gates.X(target=0) * gates.Measurement(name="", target=0)
    result = SimulatorQiskit().run(abstract_circuit=c, samples=1)

    assert (len(result.measurements['']) == 1)
    keys = [k for k in result.measurements[''].keys()]
    assert (len(keys) == 1)
    assert (keys[0].integer == 1)

    c = gates.X(target=0) * gates.Measurement(target=[0, 1])
    result = SimulatorQiskit().run(abstract_circuit=c, samples=1)

    assert (len(result.measurements['']) == 1)
    keys = [k for k in result.measurements[''].keys()]
    assert (len(keys) == 1)
    assert (keys[0].integer == 2)
