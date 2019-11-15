"""
All Backends need to be installed for full testing
"""

import pytest
import random
from openvqe.circuit import gates

system_has_pyquil = True
try:
    from openvqe.simulator.simulator_pyquil import SimulatorPyquil

    system_has_pyquil = True
except ImportError:
    system_has_pyquil = False

from shutil import which

system_has_qvm = which("qvm") is not None

system_has_qiskit = True
try:
    from openvqe.simulator.simulator_qiskit import SimulatorQiskit

    system_has_qiskit = True
except ImportError:
    system_has_qiskit = False

system_has_cirq = True
try:
    from openvqe.simulator.simulator_cirq import SimulatorCirq

    system_has_cirq = True
except ImportError:
    system_has_cirq = False

system_has_qulacs = True
try:
    from openvqe.simulator.simulator_qulacs import SimulatorQulacs

    system_has_qulacs = True
except ImportError:
    system_has_qulacs = False

from openvqe.simulator.simulator_symbolic import SimulatorSymbolic

do_pyquil = system_has_qvm and system_has_pyquil
do_qiskit = system_has_qiskit
do_cirq = system_has_cirq
do_qulacs = system_has_qulacs

wfn_backends = [SimulatorSymbolic]
if do_cirq: wfn_backends.append(SimulatorCirq)
if do_pyquil: wfn_backends.append(SimulatorPyquil)
if do_qulacs: wfn_backends.append(SimulatorQulacs)

shot_backends = []
if do_cirq: shot_backends.append(SimulatorCirq)
if do_qiskit: shot_backends.append(SimulatorQiskit)


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


@pytest.mark.parametrize("simulator", wfn_backends)
def test_wfn_simple_execution(simulator):
    ac = gates.X(0)
    ac += gates.Ry(target=1, control=0, angle=2.3 / 2)
    ac += gates.H(target=1, control=None)
    simulator().simulate_wavefunction(abstract_circuit=ac, initial_state=0)


@pytest.mark.parametrize("simulator", wfn_backends)
def test_wfn_multitarget(simulator):
    ac = gates.X([0, 1, 2])
    ac += gates.Ry(target=[1, 2], control=0, angle=2.3 / 2)
    ac += gates.H(target=[1], control=None)
    simulator().simulate_wavefunction(abstract_circuit=ac, initial_state=0)


@pytest.mark.parametrize("simulator", wfn_backends)
def test_wfn_multi_control(simulator):
    if system_has_qulacs and isinstance(simulator(), SimulatorQulacs):
        # does not support multi-control
        return
    ac = gates.X([0, 1, 2])
    ac += gates.Ry(target=[0], control=[1, 2], angle=2.3 / 2)
    ac += gates.H(target=[0], control=[1, 2])
    simulator().simulate_wavefunction(abstract_circuit=ac, initial_state=0)


@pytest.mark.parametrize("simulator", wfn_backends)
def test_wfn_simple_consistency(simulator):
    ac = create_random_circuit()
    wfn0 = simulator().simulate_wavefunction(abstract_circuit=ac).wavefunction
    wfn1 = SimulatorSymbolic().simulate_wavefunction(abstract_circuit=ac).wavefunction
    assert (wfn0 == wfn1)


@pytest.mark.parametrize("simulator", shot_backends)
def test_shot_simple_execution(simulator):
    ac = gates.X(0)
    ac += gates.Ry(target=1, control=0, angle=1.2 / 2)
    ac += gates.H(target=1, control=None)
    ac += gates.Measurement([0, 1])
    simulator().run(abstract_circuit=ac)


@pytest.mark.parametrize("simulator", shot_backends)
def test_shot_multitarget(simulator):
    ac = gates.X([0, 1, 2])
    ac += gates.Ry(target=[1, 2], control=0, angle=2.3 / 2)
    ac += gates.H(target=[1], control=None)
    ac += gates.Measurement([0, 1])
    simulator().run(abstract_circuit=ac)


@pytest.mark.parametrize("simulator", shot_backends)
def test_shot_multi_control(simulator):
    ac = gates.X([0, 1, 2])
    ac += gates.X(target=[0], control=[1, 2])
    ac += gates.Ry(target=[0], control=[1, 2], angle=2.3 / 2)
    ac += gates.Rz(target=[0], control=[1, 2], angle=2.3 / 2)
    ac += gates.Rx(target=[0], control=[1, 2], angle=2.3 / 2)
    ac += gates.Measurement([0, 1])
    simulator().run(abstract_circuit=ac)


@pytest.mark.skipif(condition=not do_qiskit or not do_cirq, reason="need qiskit and cirq")
def test_shot_simple_consistency():
    ac = create_random_circuit()
    ac += gates.Measurement([0,1,2,3,4,5])
    wfn0 = SimulatorQiskit().run(abstract_circuit=ac).counts
    wfn1 = SimulatorCirq().run(abstract_circuit=ac).counts
    assert (wfn0 == wfn1)
