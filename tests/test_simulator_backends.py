"""
All Backends need to be installed for full testing
"""

import numpy
import pytest
import random
import numbers
import tequila as tq

"""
Warn if Simulators are not installed
"""
import warnings

@pytest.mark.parametrize("backend", list(set([None] + [k for k in tq.simulators.INSTALLED_FULL_WFN_SIMULATORS.keys()] + [k for k in tq.simulators.INSTALLED_SAMPLERS.keys()])))
def test_interface(backend):
    H = tq.paulis.X(0)
    U = tq.gates.X(target=0)
    a = tq.simulators.simulate(objective=U, backend=backend)
    assert(isinstance(a, tq.QubitWaveFunction))
    E = tq.ExpectationValue(H=H, U=U)
    a = tq.simulators.simulate(objective=E, backend=backend)
    assert(isinstance(a, numbers.Number))

@pytest.mark.parametrize("name", tq.simulators.supported_simulators())
def test_backend_availability(name):
    installed = getattr(tq.simulators, "HAS_" + name.upper())
    if not installed:
        warnings.warn(name + " is not installed!", UserWarning)


def create_random_circuit():
    primitive_gates = [tq.gates.X, tq.gates.Y, tq.gates.Z, tq.gates.H]
    rot_gates = [tq.gates.Rx, tq.gates.Ry, tq.gates.Rz]
    circuit = tq.gates.QCircuit()
    for x in range(4):
        target = random.randint(1, 2)
        control = random.randint(3, 4)
        circuit += random.choice(primitive_gates)(target=target, control=control)
        target = random.randint(1, 2)
        control = random.randint(3, 4)
        angle = random.uniform(0.0, 4.0)
        circuit += random.choice(rot_gates)(target=target, control=control, angle=angle)
    return circuit


@pytest.mark.parametrize("simulator", tq.simulators.get_all_wfn_simulators())
def test_wfn_simple_execution(simulator):
    ac = tq.gates.X(0)
    ac += tq.gates.Ry(target=1, control=0, angle=2.3 / 2)
    ac += tq.gates.H(target=1, control=None)
    simulator().simulate_wavefunction(abstract_circuit=ac, initial_state=0)


@pytest.mark.parametrize("simulator", tq.simulators.get_all_wfn_simulators())
def test_wfn_multitarget(simulator):
    ac = tq.gates.X([0, 1, 2])
    ac += tq.gates.Ry(target=[1, 2], control=0, angle=2.3 / 2)
    ac += tq.gates.H(target=[1], control=None)
    simulator().simulate_wavefunction(abstract_circuit=ac, initial_state=0)


@pytest.mark.parametrize("simulator", tq.simulators.get_all_wfn_simulators())
def test_wfn_multi_control(simulator):
    # currently no compiler, so that test can not succeed
    if tq.simulators.HAS_QULACS and isinstance(simulator(), tq.simulators.SimulatorQulacs):
        return
    ac = tq.gates.X([0, 1, 2])
    ac += tq.gates.Ry(target=[0], control=[1, 2], angle=2.3 / 2)
    ac += tq.gates.H(target=[0], control=[1, 2])
    simulator().simulate_wavefunction(abstract_circuit=ac, initial_state=0)


@pytest.mark.parametrize("simulator", tq.simulators.get_all_wfn_simulators())
def test_wfn_simple_consistency(simulator):
    ac = create_random_circuit()
    wfn0 = simulator().simulate_wavefunction(abstract_circuit=ac).wavefunction
    wfn1 = tq.simulators.SimulatorSymbolic().simulate_wavefunction(abstract_circuit=ac).wavefunction
    assert (wfn0 == wfn1)


@pytest.mark.parametrize("simulator", tq.simulators.get_all_samplers())
def test_shot_simple_execution(simulator):
    ac = tq.gates.X(0)
    ac += tq.gates.Ry(target=1, control=0, angle=1.2 / 2)
    ac += tq.gates.H(target=1, control=None)
    ac += tq.gates.Measurement([0, 1])
    simulator().run(abstract_circuit=ac)


@pytest.mark.parametrize("simulator", tq.simulators.get_all_samplers())
def test_shot_multitarget(simulator):
    ac = tq.gates.X([0, 1, 2])
    ac += tq.gates.Ry(target=[1, 2], control=0, angle=2.3 / 2)
    ac += tq.gates.H(target=[1], control=None)
    ac += tq.gates.Measurement([0, 1])
    simulator().run(abstract_circuit=ac)


@pytest.mark.parametrize("simulator", tq.simulators.get_all_samplers())
def test_shot_multi_control(simulator):
    ac = tq.gates.X([0, 1, 2])
    ac += tq.gates.X(target=[0], control=[1, 2])
    ac += tq.gates.Ry(target=[0], control=[1, 2], angle=2.3 / 2)
    ac += tq.gates.Rz(target=[0], control=[1, 2], angle=2.3 / 2)
    ac += tq.gates.Rx(target=[0], control=[1, 2], angle=2.3 / 2)
    ac += tq.gates.Measurement([0, 1])
    simulator().run(abstract_circuit=ac)


@pytest.mark.skipif(condition=not tq.simulators.HAS_CIRQ or not tq.simulators.HAS_QISKIT, reason="need qiskit and cirq")
def test_shot_simple_consistency():
    ac = create_random_circuit()
    ac += tq.gates.Measurement([0, 1, 2, 3, 4, 5])
    wfn0 = tq.simulators.SimulatorQiskit().run(abstract_circuit=ac).counts
    wfn1 = tq.simulators.SimulatorCirq().run(abstract_circuit=ac).counts
    assert (wfn0 == wfn1)


@pytest.mark.parametrize("simulator", tq.simulators.get_all_wfn_simulators())
@pytest.mark.parametrize("initial_state", numpy.random.randint(0, 31, 5))
def test_initial_state_from_integer(simulator, initial_state):
    U = tq.gates.QCircuit()
    for i in range(6):
        U += tq.gates.X(target=i) + tq.gates.X(target=i)

    wfn = simulator().simulate_wavefunction(abstract_circuit=U, initial_state=initial_state).wavefunction
    print(wfn)
    for k,v in wfn.items():
        print(k, " ", k.integer, " ", type(k), " ", v)
    assert (initial_state in wfn)
    assert (numpy.isclose(wfn[initial_state],1.0))
