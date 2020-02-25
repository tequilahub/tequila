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

@pytest.mark.dependencies
def test_dependencies():
    for package in tq.simulators.SUPPORTED_BACKENDS:
        assert(package in tq.simulators.INSTALLED_BACKENDS)

@pytest.mark.parametrize("backend", list(set(
    [None] + [k for k in tq.simulators.INSTALLED_SIMULATORS.keys()] + [k for k in
                                                                       tq.simulators.INSTALLED_SAMPLERS.keys()])))
def test_interface(backend):
    H = tq.paulis.X(0)
    U = tq.gates.X(target=0)
    CU = tq.compile(objective=U, backend=backend)
    a = tq.simulate(objective=U, backend=backend)
    assert (isinstance(a, tq.QubitWaveFunction))
    E = tq.ExpectationValue(H=H, U=U)
    CE = tq.compile(objective=E, backend=backend)
    a = tq.simulate(objective=E, backend=backend)
    assert (isinstance(a, numbers.Number))


@pytest.mark.parametrize("name", tq.simulators.SUPPORTED_BACKENDS)
def test_backend_availability(name):
    installed = getattr(tq.simulators, "HAS_" + name.upper())
    if not installed:
        warnings.warn(name + " is not installed!", UserWarning)


@pytest.mark.parametrize("simulator", tq.simulators.INSTALLED_SIMULATORS.keys())
@pytest.mark.parametrize("angle", numpy.random.uniform(0.0, 2.0*numpy.pi,2))
def test_rotations(simulator, angle):
    U1 = tq.gates.X(target=1) + tq.gates.X(target=0, control=1) + tq.gates.Rx(angle=angle, target=0)
    U2 = tq.gates.X(target=1) + tq.gates.X(target=0, control=1) + tq.gates.ExpPauli(angle=angle, paulistring="X(0)")
    wfn1 = tq.simulate(U1, backend=None)
    wfn2 = tq.simulate(U2, backend=None)
    wfn3 = tq.simulate(U2, backend=simulator)
    wfn4 = tq.simulate(U2, backend=simulator)

    assert (numpy.isclose(numpy.abs(wfn1.inner(wfn2)) ** 2, 1.0, atol=1.e-4))
    assert (numpy.isclose(numpy.abs(wfn3.inner(wfn4)) ** 2, 1.0, atol=1.e-4))
    assert (numpy.isclose(numpy.abs(wfn1.inner(wfn3)) ** 2, 1.0, atol=1.e-4))

    U = tq.gates.X(target=1) + tq.gates.X(target=0, control=1) + tq.gates.ExpPauli(angle=angle, paulistring="X(0)Y(3)")
    wfn1 = tq.simulate(U2, backend=None)
    wfn2 = tq.simulate(U2, backend=simulator)

    assert (numpy.isclose(numpy.abs(wfn1.inner(wfn2)) ** 2, 1.0, atol=1.e-4))


@pytest.mark.parametrize("simulator", tq.simulators.INSTALLED_SIMULATORS.keys())
@pytest.mark.parametrize("angle", numpy.random.uniform(0.0, 2.0 * numpy.pi, 2))
@pytest.mark.parametrize("ps", ["X(0)Y(3)", "Y(2)X(4)"]) # it is important to test paulistrings on qubits which are not explicitly initialized through other gates
def test_multi_pauli_rotation(simulator, angle, ps):

    U = tq.gates.X(target=1) + tq.gates.X(target=0, control=1) + tq.gates.ExpPauli(angle=angle, paulistring=ps)
    wfn1 = tq.simulate(U, backend=None)
    wfn2 = tq.simulate(U, backend=simulator)

    assert (numpy.isclose(numpy.abs(wfn1.inner(wfn2)) ** 2, 1.0, atol=1.e-4))

@pytest.mark.parametrize("simulator", tq.simulators.INSTALLED_SIMULATORS.keys())
@pytest.mark.parametrize("angle", numpy.random.uniform(0.0, 2.0*numpy.pi,2))
def test_parametrized_rotations(simulator, angle):
    U1 = tq.gates.X(target=1) + tq.gates.X(target=0, control=1) + tq.gates.Rx(angle="a", target=0)
    U2 = tq.gates.X(target=1) + tq.gates.X(target=0, control=1) + tq.gates.ExpPauli(angle="a", paulistring="X(0)")
    variables = {"a": angle}
    wfn1 = tq.simulate(U1, variables, backend=None)
    wfn2 = tq.simulate(U2, variables, backend=None)
    wfn3 = tq.simulate(U2, variables, backend=simulator)
    wfn4 = tq.simulate(U2, variables, backend=simulator)

    assert (numpy.isclose(numpy.abs(wfn1.inner(wfn2)) ** 2, 1.0, atol=1.e-4))
    assert (numpy.isclose(numpy.abs(wfn3.inner(wfn4)) ** 2, 1.0, atol=1.e-4))
    assert (numpy.isclose(numpy.abs(wfn1.inner(wfn3)) ** 2, 1.0, atol=1.e-4))

@pytest.mark.parametrize("simulator", tq.simulators.INSTALLED_SIMULATORS.keys())
@pytest.mark.parametrize("angle", numpy.random.uniform(0.0, 2.0 * numpy.pi, 2))
@pytest.mark.parametrize("ps", ["X(0)Z(3)", "Y(2)X(4)"])
def test_parametrized_multi_pauli_rotation(simulator, angle, ps):
    a = tq.Variable("a")
    variables = {a:angle}
    U = tq.gates.X(target=1) + tq.gates.X(target=0, control=1) + tq.gates.ExpPauli(angle=a, paulistring=ps)
    wfn1 = tq.simulate(U, variables, backend=None)
    wfn2 = tq.simulate(U, variables, backend=simulator)

    print(wfn1)
    print(wfn2)

    assert (numpy.isclose(numpy.abs(wfn1.inner(wfn2)) ** 2, 1.0, atol=1.e-4))

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


@pytest.mark.parametrize("simulator", tq.simulators.INSTALLED_SIMULATORS.keys())
def test_wfn_simple_execution(simulator):
    ac = tq.gates.X(0)
    ac += tq.gates.Ry(target=1, control=0, angle=2.3 / 2)
    ac += tq.gates.H(target=1, control=None)
    tq.simulate(ac, backend=simulator)


@pytest.mark.parametrize("simulator", tq.simulators.INSTALLED_SIMULATORS.keys())
def test_wfn_multitarget(simulator):
    ac = tq.gates.X([0, 1, 2])
    ac += tq.gates.Ry(target=[1, 2], control=0, angle=2.3 / 2)
    ac += tq.gates.H(target=[1], control=None)
    tq.simulate(ac, backend=simulator)


@pytest.mark.parametrize("simulator", tq.simulators.INSTALLED_SIMULATORS.keys())
def test_wfn_multi_control(simulator):
    # currently no compiler, so that test can not succeed
    if simulator == "qulacs":
        return
    ac = tq.gates.X([0, 1, 2])
    ac += tq.gates.Ry(target=[0], control=[1, 2], angle=2.3 / 2)
    ac += tq.gates.H(target=[0], control=[1])
    tq.simulate(ac, backend=simulator)

    if simulator == "qiskit":  # can't compile the CCH currently ... but throws error
        return

    ac = tq.gates.X([0, 1, 2])
    ac += tq.gates.Ry(target=[0], control=[1, 2], angle=2.3 / 2)
    ac += tq.gates.H(target=[0], control=[1, 2])
    tq.simulate(ac, backend=simulator)


@pytest.mark.parametrize("simulator", tq.simulators.INSTALLED_SIMULATORS.keys())
def test_wfn_simple_consistency(simulator):
    ac = create_random_circuit()
    wfn0 = tq.simulate(ac, backend=simulator)
    wfn1 = tq.simulate(ac, backend=None)
    assert (wfn0 == wfn1)


@pytest.mark.parametrize("simulator", tq.simulators.INSTALLED_SAMPLERS.keys())
def test_shot_simple_execution(simulator):
    ac = tq.gates.X(0)
    ac += tq.gates.Ry(target=1, control=0, angle=1.2 / 2)
    ac += tq.gates.H(target=1, control=None)
    ac += tq.gates.Measurement([0, 1])
    tq.simulate(ac, samples=1)


@pytest.mark.parametrize("simulator", tq.simulators.INSTALLED_SAMPLERS.keys())
def test_shot_multitarget(simulator):
    ac = tq.gates.X([0, 1, 2])
    ac += tq.gates.Ry(target=[1, 2], control=0, angle=2.3 / 2)
    ac += tq.gates.H(target=[1], control=None)
    ac += tq.gates.Measurement([0, 1])
    tq.simulate(ac, samples=1)


@pytest.mark.parametrize("simulator", tq.simulators.INSTALLED_SAMPLERS.keys())
def test_shot_multi_control(simulator):
    ac = tq.gates.X([0, 1, 2])
    ac += tq.gates.X(target=[0], control=[1, 2])
    ac += tq.gates.Ry(target=[0], control=[1, 2], angle=2.3 / 2)
    ac += tq.gates.Rz(target=[0], control=[1, 2], angle=2.3 / 2)
    ac += tq.gates.Rx(target=[0], control=[1, 2], angle=2.3 / 2)
    ac += tq.gates.Measurement([0, 1])
    tq.simulate(ac, samples=1)


@pytest.mark.skipif(condition=not tq.simulators.HAS_CIRQ or not tq.simulators.HAS_QISKIT, reason="need qiskit and cirq")
def test_shot_simple_consistency():
    ac = create_random_circuit()
    ac += tq.gates.Measurement([0, 1, 2, 3, 4, 5])
    wfn0 = tq.simulate(ac, backend="cirq")
    wfn1 = tq.simulate(ac, backend="qiskit")
    assert (wfn0 == wfn1)


@pytest.mark.parametrize("simulator", tq.simulators.INSTALLED_SIMULATORS.keys())
@pytest.mark.parametrize("initial_state", numpy.random.randint(0, 31, 5))
def test_initial_state_from_integer(simulator, initial_state):
    U = tq.gates.QCircuit()
    for i in range(6):
        U += tq.gates.X(target=i) + tq.gates.X(target=i)

    wfn = tq.simulate(U, initial_state=initial_state)
    print(wfn)
    for k, v in wfn.items():
        print(k, " ", k.integer, " ", type(k), " ", v)
    assert (initial_state in wfn)
    assert (numpy.isclose(wfn[initial_state], 1.0))
