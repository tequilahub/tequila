"""
All Backends need to be installed for full testing
"""

import pytest
import random
from tequila.circuit import gates
from tequila import simulators

"""
Warn if Simulators are not installed
"""
import warnings


@pytest.mark.parametrize("name", simulators.supported_simulators())
def test_backend_availability(name):
    installed = getattr(simulators, "HAS_" + name.upper())
    if not installed:
        warnings.warn(name + " is not installed!", UserWarning)

def create_random_circuit():
    primitive_gates = [gates.X, gates.Y, gates.Z, gates.H]
    rot_gates_gates = [gates.Rx, gates.Ry, gates.Rz]
    circuit = gates.QCircuit()
    for x in range(4):
        target = random.randint(1, 2)
        control = random.randint(3, 4)
        circuit += random.choice(primitive_gates)(target=target, control=control)
        target = random.randint(1, 2)
        control = random.randint(3, 4)
        angle = random.uniform(0.0, 4.0)
        circuit += random.choice(rot_gates_gates)(target=target, control=control, angle=angle)
    return circuit


@pytest.mark.parametrize("simulator", simulators.get_all_wfn_simulators())
def test_wfn_simple_execution(simulator):
    ac = gates.X(0)
    ac += gates.Ry(target=1, control=0, angle=2.3 / 2)
    ac += gates.H(target=1, control=None)
    simulator().simulate_wavefunction(abstract_circuit=ac, initial_state=0)


@pytest.mark.parametrize("simulator", simulators.get_all_wfn_simulators())
def test_wfn_multitarget(simulator):
    ac = gates.X([0, 1, 2])
    ac += gates.Ry(target=[1, 2], control=0, angle=2.3 / 2)
    ac += gates.H(target=[1], control=None)
    simulator().simulate_wavefunction(abstract_circuit=ac, initial_state=0)


@pytest.mark.parametrize("simulator", simulators.get_all_wfn_simulators())
def test_wfn_multi_control(simulator):
    # currently no compiler, so that test can not succeed
    if simulators.HAS_QULACS and isinstance(simulator(), simulators.SimulatorQulacs):
        return
    ac = gates.X([0, 1, 2])
    ac += gates.Ry(target=[0], control=[1, 2], angle=2.3 / 2)
    ac += gates.H(target=[0], control=[1, 2])
    simulator().simulate_wavefunction(abstract_circuit=ac, initial_state=0)


@pytest.mark.parametrize("simulator", simulators.get_all_wfn_simulators())
def test_wfn_simple_consistency(simulator):
    ac = create_random_circuit()
    wfn0 = simulator().simulate_wavefunction(abstract_circuit=ac).wavefunction
    wfn1 = simulators.SimulatorSymbolic().simulate_wavefunction(abstract_circuit=ac).wavefunction
    assert (wfn0 == wfn1)


@pytest.mark.parametrize("simulator", simulators.get_all_samplers())
def test_shot_simple_execution(simulator):
    ac = gates.X(0)
    ac += gates.Ry(target=1, control=0, angle=1.2 / 2)
    ac += gates.H(target=1, control=None)
    ac += gates.Measurement([0, 1])
    simulator().run(abstract_circuit=ac)


@pytest.mark.parametrize("simulator", simulators.get_all_samplers())
def test_shot_multitarget(simulator):
    ac = gates.X([0, 1, 2])
    ac += gates.Ry(target=[1, 2], control=0, angle=2.3 / 2)
    ac += gates.H(target=[1], control=None)
    ac += gates.Measurement([0, 1])
    simulator().run(abstract_circuit=ac)


@pytest.mark.parametrize("simulator", simulators.get_all_samplers())
def test_shot_multi_control(simulator):
    ac = gates.X([0, 1, 2])
    ac += gates.X(target=[0], control=[1, 2])
    ac += gates.Ry(target=[0], control=[1, 2], angle=2.3 / 2)
    ac += gates.Rz(target=[0], control=[1, 2], angle=2.3 / 2)
    ac += gates.Rx(target=[0], control=[1, 2], angle=2.3 / 2)
    ac += gates.Measurement([0, 1])
    simulator().run(abstract_circuit=ac)


@pytest.mark.skipif(condition=not simulators.HAS_CIRQ or not simulators.HAS_QISKIT, reason="need qiskit and cirq")
def test_shot_simple_consistency():
    ac = create_random_circuit()
    ac += gates.Measurement([0, 1, 2, 3, 4, 5])
    wfn0 = simulators.SimulatorQiskit().run(abstract_circuit=ac).counts
    wfn1 = simulators.SimulatorCirq().run(abstract_circuit=ac).counts
    assert (wfn0 == wfn1)
