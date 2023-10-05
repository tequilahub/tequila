import tequila.simulators.simulator_api
from tequila.circuit import gates
from tequila.circuit.gradient import grad
from tequila.objective import ExpectationValue
from tequila.objective.objective import Variable
from tequila.hamiltonian import paulis
from tequila.simulators.simulator_api import simulate
from tequila import simulators
import numpy
import pytest

import tequila as tq

# Get QC backends for parametrized testing
import select_backends

simulators = select_backends.get()
samplers = select_backends.get(sampler=True)

# special tests
def test_gradient_swap():
    U1 = tq.gates.X(2) + tq.gates.SWAP(1,2,angle="a")
    U2 = tq.compile_circuit(U1)
    H = tq.paulis.Z(1) - tq.paulis.Z(2) 
    E1 = tq.ExpectationValue(H=H, U=U1)
    E2 = tq.ExpectationValue(H=H, U=U2)
    dE1 = tq.grad(E1, "a")
    dE2 = tq.grad(E1, "a")
    for angle in numpy.random.uniform(0.0,numpy.pi*4,10):
        g1 = tq.simulate(dE1, variables={"a":angle})
        g2 = tq.simulate(dE2, variables={"a":angle})
        assert numpy.isclose(g1,g2,atol=1.e-5)

# special tests
def test_gradient_genrot():
    G = tq.paulis.KetBra(ket="|101>", bra="|010>")
    G = G + G.dagger()
    P0 = 1.0 - tq.paulis.Projector("|101>") - tq.paulis.Projector("|010>")
    U1 = tq.gates.X([0,2]) + tq.gates.GeneralizedRotation(generator=G, p0=P0 ,angle="a")
    U2 = tq.gates.X([0,2]) + tq.gates.GeneralizedRotation(generator=G, p0=P0 ,angle="a", assume_real=True)
    U3 = tq.compile_circuit(U1)
    H = tq.paulis.Z(1) - tq.paulis.Z(2)
    E1 = tq.ExpectationValue(H=H, U=U1)
    E2 = tq.ExpectationValue(H=H, U=U2)
    E3 = tq.ExpectationValue(H=H, U=U3)
    dE1 = tq.grad(E1, "a")
    dE2 = tq.grad(E2, "a")
    dE3 = tq.grad(E3, "a")
    for angle in numpy.random.uniform(0.0,numpy.pi*4,10):
        g1 = tq.simulate(dE1, variables={"a":angle})
        g2 = tq.simulate(dE2, variables={"a":angle})
        g3 = tq.simulate(dE3, variables={"a":angle})
        assert numpy.isclose(g1,g2,atol=1.e-5)
        assert numpy.isclose(g1,g3,atol=1.e-5)

@pytest.mark.parametrize("simulator", simulators)
@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("assume_real", [False, True])
@pytest.mark.parametrize("angle_value", numpy.random.uniform(0.0, 2.0 * numpy.pi, 1))
def test_gradient_UY_HX(simulator, angle_value, controlled, assume_real, silent=True):
    # case X Y
    # U = cos(angle/2) + sin(-angle/2)*i*Y
    # <0|Ud H U |0> = cos^2(angle/2)*<0|X|0>
    # + sin^2(-angle/2) <0|YXY|0>
    # + cos(angle/2)*sin(angle/2)*i<0|XY|0>
    # + sin(-angle/2)*cos(angle/2)*(-i) <0|YX|0>
    # = cos^2*0 + sin^2*0 + cos*sin*i(<0|[XY,YX]|0>)
    # = 0.5*sin(-angle)*i <0|[XY,YX]|0> = -0.5*sin(angle)*i * 2 i <0|Z|0>
    # = sin(angle)

    angle = Variable(name="angle")
    variables = {angle: angle_value}

    qubit = 0
    H = paulis.X(qubit=qubit)
    if controlled:
        control = 1
        U = gates.X(target=control) + gates.Ry(target=qubit, control=control, assume_real=assume_real, angle=angle)
    else:
        U = gates.X(target=qubit) + gates.X(target=qubit) + gates.Ry(target=qubit, angle=angle)
    O = ExpectationValue(U=U, H=H)
    E = simulate(O, variables=variables, backend=simulator)
    print("O={type}".format(type=type(O)))
    dO = grad(objective=O, variable=angle)
    dE = simulate(dO, variables=variables, backend=simulator)
    assert (numpy.isclose(E, numpy.sin(angle(variables)), atol=1.e-4))
    assert (numpy.isclose(dE, numpy.cos(angle(variables)), atol=1.e-4))
    if not silent:
        print("E         =", E)
        print("sin(angle)=", numpy.sin(angle()))
        print("dE        =", dE)
        print("cos(angle)=", numpy.cos(angle()))


@pytest.mark.parametrize("simulator", simulators)
@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("assume_real", [False, True])
@pytest.mark.parametrize("angle_value", numpy.random.uniform(0.0, 2.0 * numpy.pi, 1))
def test_gradient_UY_HX_sample(simulator, angle_value, controlled, assume_real, silent=True):
    # case X Y
    # U = cos(angle/2) + sin(-angle/2)*i*Y
    # <0|Ud H U |0> = cos^2(angle/2)*<0|X|0>
    # + sin^2(-angle/2) <0|YXY|0>
    # + cos(angle/2)*sin(angle/2)*i<0|XY|0>
    # + sin(-angle/2)*cos(angle/2)*(-i) <0|YX|0>
    # = cos^2*0 + sin^2*0 + cos*sin*i(<0|[XY,YX]|0>)
    # = 0.5*sin(-angle)*i <0|[XY,YX]|0> = -0.5*sin(angle)*i * 2 i <0|Z|0>
    # = sin(angle)

    angle = Variable(name="angle")
    variables = {angle: angle_value}

    qubit = 0
    H = paulis.X(qubit=qubit)
    if controlled:
        control = 1
        U = gates.X(target=control) + gates.Ry(target=qubit, control=control, assume_real=assume_real, angle=angle)
    else:
        U = gates.X(target=qubit) + gates.X(target=qubit) + gates.Ry(target=qubit, angle=angle)
    O = ExpectationValue(U=U, H=H)
    E = simulate(O, variables=variables, backend=simulator, samples=10000)
    print("O={type}".format(type=type(O)))
    dO = grad(objective=O, variable=angle)
    dE = simulate(dO, variables=variables, backend=simulator, samples=10000)
    assert (numpy.isclose(E, numpy.sin(angle(variables)), atol=3.e-2))
    assert (numpy.isclose(dE, numpy.cos(angle(variables)), atol=3.e-2))
    if not silent:
        print("E         =", E)
        print("sin(angle)=", numpy.sin(angle()))
        print("dE        =", dE)
        print("cos(angle)=", numpy.cos(angle()))


@pytest.mark.parametrize("simulator", simulators)
@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("assume_real", [False, True])
@pytest.mark.parametrize("angle_value", numpy.random.uniform(0.0, 2.0 * numpy.pi, 1))
def test_gradient_UX_HY(simulator, angle_value, controlled, assume_real, silent=True):
    # case YX
    # U = cos(angle/2) + sin(-angle/2)*i*X
    # O = cos*sin*i*<0|YX|0> + sin*cos*(-i)<0|XY|0>
    #   = 0.5*sin(-angle)*i <0|[YX,XY]|0>
    #   = -sin(angle)

    angle = Variable(name="angle")
    variables = {angle: angle_value}

    qubit = 0
    H = paulis.Y(qubit=qubit)
    if controlled:
        control = 1
        U = gates.X(target=control) + gates.Rx(target=qubit, control=control, assume_real=assume_real, angle=angle)
    else:
        U = gates.Rx(target=qubit, angle=angle)
    O = ExpectationValue(U=U, H=H)
    E = simulate(O, variables=variables, backend=simulator)
    dO = grad(objective=O, variable='angle')
    dE = simulate(dO, variables=variables)
    assert (numpy.isclose(E, -numpy.sin(angle(variables)), atol=1.e-4))
    assert (numpy.isclose(dE, -numpy.cos(angle(variables)), atol=1.e-4))
    if not silent:
        print("E         =", E)
        print("-sin(angle)=", -numpy.sin(angle(variables)))
        print("dE        =", dE)
        print("-cos(angle)=", -numpy.cos(angle(variables)))


@pytest.mark.parametrize("simulator", simulators)
@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("assume_real", [False, True])
@pytest.mark.parametrize("angle_value", numpy.random.uniform(0.0, 2.0 * numpy.pi, 1))
def test_gradient_UHZH_HY(simulator, angle_value, controlled, assume_real, silent=True):
    angle = Variable(name="angle")
    variables = {angle: angle_value}

    qubit = 0
    H = paulis.Y(qubit=qubit)
    if controlled:
        control = 1
        U = gates.X(target=control) + gates.H(target=qubit) + gates.Rz(target=qubit, control=control,
                                                                       assume_real=assume_real,
                                                                       angle=angle) + gates.H(target=qubit)
    else:
        U = gates.H(target=qubit) + gates.Rz(target=qubit, angle=angle) + gates.H(target=qubit)
    O = ExpectationValue(U=U, H=H)
    E = simulate(O, variables=variables, backend=simulator)
    dO = grad(objective=O, variable='angle')
    dE = simulate(dO, variables=variables)
    assert (numpy.isclose(E, -numpy.sin(angle(variables)), atol=1.e-4))
    assert (numpy.isclose(dE, -numpy.cos(angle(variables)), atol=1.e-4))
    if not silent:
        print("E         =", E)
        print("-sin(angle)=", -numpy.sin(angle(variables)))
        print("dE        =", dE)
        print("-cos(angle)=", -numpy.cos(angle(variables)))


@pytest.mark.parametrize("simulator", simulators)
@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("assume_real", [False, True])
@pytest.mark.parametrize("angle_value", numpy.random.uniform(0.0, 2.0 * numpy.pi, 1))
def test_gradient_PHASE_HY(simulator, angle_value, controlled, assume_real, silent=True):
    angle = Variable(name="angle")
    variables = {angle: angle_value}

    qubit = 0
    H = paulis.Y(qubit=qubit)
    if controlled:
        control = 1
        U = gates.X(target=control) + gates.H(target=qubit) + gates.Phase(target=qubit, control=control,
                                                                          assume_real=assume_real,
                                                                          phi=angle) + gates.H(target=qubit)
    else:
        U = gates.H(target=qubit) + gates.Phase(target=qubit, phi=angle) + gates.H(target=qubit)

    O = ExpectationValue(U=U, H=H)
    E = simulate(O, variables=variables, backend=simulator)
    dO = grad(objective=O, variable='angle')
    dE = simulate(dO, variables=variables)
    assert (numpy.isclose(E, -numpy.sin(angle(variables)), atol=1.e-4))
    assert (numpy.isclose(dE, -numpy.cos(angle(variables)), atol=1.e-4))
    if not silent:
        print("E         =", E)
        print("-sin(angle)=", -numpy.sin(angle(variables)))
        print("dE        =", dE)
        print("-cos(angle)=", -numpy.cos(angle(variables)))


@pytest.mark.parametrize("simulator", simulators)
@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("assume_real", [False, True])
@pytest.mark.parametrize("angle_value", numpy.random.uniform(0.0, 2.0 * numpy.pi, 1))
def test_gradient_UY_HX_wfnsim(simulator, angle_value, controlled, assume_real, silent=True):
    # same as before just with wavefunction simulation

    # case X Y
    # U = cos(angle/2) + sin(-angle/2)*i*Y
    # <0|Ud H U |0> = cos^2(angle/2)*<0|X|0>
    # + sin^2(-angle/2) <0|YXY|0>
    # + cos(angle/2)*sin(angle/2)*i<0|XY|0>
    # + sin(-angle/2)*cos(angle/2)*(-i) <0|YX|0>
    # = cos^2*0 + sin^2*0 + cos*sin*i(<0|[XY,YX]|0>)
    # = 0.5*sin(-angle)*i <0|[XY,YX]|0> = -0.5*sin(angle)*i * 2 i <0|Z|0>
    # = sin(angle)

    angle = Variable(name="angle")
    variables = {angle: angle_value}

    qubit = 0
    H = paulis.X(qubit=qubit)
    if controlled:
        control = 1
        U = gates.X(target=control) + gates.Ry(target=qubit, control=control, assume_real=assume_real, angle=angle)
    else:
        U = gates.Ry(target=qubit, angle=angle)
    O = ExpectationValue(U=U, H=H)
    E = simulate(O, variables=variables, backend=simulator)
    dO = grad(objective=O, variable='angle')
    dE = simulate(dO, variables=variables, backend=simulator)
    E = float(E)  # for isclose
    dE = float(dE)  # for isclose
    assert (numpy.isclose(E, numpy.sin(angle(variables)), atol=0.0001))
    assert (numpy.isclose(dE, numpy.cos(angle(variables)), atol=0.0001))
    if not silent:
        print("E         =", E)
        print("sin(angle)=", numpy.sin(angle(variables)))
        print("dE        =", dE)
        print("cos(angle)=", numpy.cos(angle(variables)))


@pytest.mark.parametrize("simulator", simulators)
@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("assume_real", [False, True])
@pytest.mark.parametrize("angle", numpy.random.uniform(0.0, 2.0 * numpy.pi, 1))
def test_gradient_UX_HY_wfnsim(simulator, angle, controlled, assume_real, silent=True):
    # same as before just with wavefunction simulation

    # case YX
    # U = cos(angle/2) + sin(-angle/2)*i*X
    # O = cos*sin*i*<0|YX|0> + sin*cos*(-i)<0|XY|0>
    #   = 0.5*sin(-angle)*i <0|[YX,XY]|0>
    #   = -sin(angle)

    angle_value = angle
    angle = Variable(name="angle")
    variables = {angle: angle_value}

    qubit = 0
    H = paulis.Y(qubit=qubit)
    if controlled:
        control = 1
        U = gates.X(target=control) + gates.Rx(target=qubit, control=control, assume_real=assume_real, angle=angle)
    else:
        U = gates.Rx(target=qubit, angle=angle)
    O = ExpectationValue(U=U, H=H)
    E = simulate(O, variables=variables, backend=simulator)
    dO = grad(objective=O, variable=angle)
    dE = simulate(dO, variables=variables)
    assert (numpy.isclose(E, -numpy.sin(angle(variables)), atol=0.0001))
    assert (numpy.isclose(dE, -numpy.cos(angle(variables)), atol=0.0001))
    if not silent:
        print("E         =", E)
        print("-sin(angle)=", -numpy.sin(angle(variables)))
        print("dE        =", dE)
        print("-cos(angle)=", -numpy.cos(angle(variables)))


@pytest.mark.parametrize("simulator", simulators)
@pytest.mark.parametrize("power", numpy.random.uniform(0.0, 2.0, 1))
@pytest.mark.parametrize("controlled", [False, True])
def test_gradient_X(simulator, power, controlled):
    qubit = 0
    control = 1
    angle = Variable(name="angle")
    if controlled:
        U = gates.X(target=control) + gates.X(target=qubit, power=angle, control=control)
    else:
        U = gates.X(target=qubit, power=angle)
    angle = Variable(name="angle")
    variables = {angle: power}
    H = paulis.Y(qubit=qubit)
    O = ExpectationValue(U=U, H=H)
    E = simulate(O, variables=variables, backend=simulator)
    dO = grad(objective=O, variable=angle)
    dE = simulate(dO, variables=variables, backend=simulator)
    assert (numpy.isclose(E, -numpy.sin(angle(variables) * (numpy.pi)), atol=1.e-4))
    assert (numpy.isclose(dE, -numpy.pi * numpy.cos(angle(variables) * (numpy.pi)), atol=1.e-4))


@pytest.mark.parametrize("simulator", simulators)
@pytest.mark.parametrize("power", numpy.random.uniform(0.0, 2.0, 1))
@pytest.mark.parametrize("controls", [1, 2, 3])
def test_gradient_deep_controlled_X(simulator, power, controls):
    if controls > 2 and simulator == "qiskit":
        # does not work yet
        return
    qubit = 0
    control = [i for i in range(1, controls + 1)]
    angle = Variable(name="angle")
    U = gates.X(target=control) + gates.X(target=qubit, power=angle, control=control)
    angle = Variable(name="angle")
    variables = {angle: power}
    H = paulis.Y(qubit=qubit)
    O = ExpectationValue(U=U, H=H)
    E = simulate(O, variables=variables, backend=simulator)
    dO = grad(objective=O, variable=angle)
    dE = simulate(dO, variables=variables, backend=simulator)
    assert (numpy.isclose(E, -numpy.sin(angle(variables) * (numpy.pi)), atol=1.e-4))
    assert (numpy.isclose(dE, -numpy.pi * numpy.cos(angle(variables) * (numpy.pi)), atol=1.e-4))


@pytest.mark.parametrize("simulator", simulators)
@pytest.mark.parametrize("power", numpy.random.uniform(0.0, 2.0 * numpy.pi, 1))
@pytest.mark.parametrize("controlled", [False, True])
def test_gradient_Y(simulator, power, controlled):
    if simulator != "cirq":
        return
    qubit = 0
    control = 1
    angle = Variable(name="angle")
    if controlled:
        U = gates.X(target=control) + gates.Y(target=qubit, power=angle, control=control)
    else:
        U = gates.Y(target=qubit, power=angle)
    angle = Variable(name="angle")
    variables = {angle: power}
    H = paulis.X(qubit=qubit)
    O = ExpectationValue(U=U, H=H)
    E = simulate(O, variables=variables, backend=simulator)
    dO = grad(objective=O, variable=angle)
    dE = simulate(dO, variables=variables, backend=simulator)
    assert (numpy.isclose(E, numpy.sin(angle(variables) * (numpy.pi)), atol=1.e-4))
    assert (numpy.isclose(dE, numpy.pi * numpy.cos(angle(variables) * (numpy.pi)), atol=1.e-4))


@pytest.mark.parametrize("simulator", simulators)
@pytest.mark.parametrize("power", numpy.random.uniform(0.0, 2.0, 1))
@pytest.mark.parametrize("controls", [1, 2, 3])
def test_gradient_deep_controlled_Y(simulator, power, controls):
    if controls > 2 and simulator == "qiskit":
        # does not work yet
        return
    qubit = 0
    control = [i for i in range(1, controls + 1)]
    angle = Variable(name="angle")
    U = gates.X(target=control) + gates.Y(target=qubit, power=angle, control=control)
    angle = Variable(name="angle")
    variables = {angle: power}
    H = paulis.X(qubit=qubit)
    O = ExpectationValue(U=U, H=H)
    E = simulate(O, variables=variables, backend=simulator)
    dO = grad(objective=O, variable=angle)
    dE = simulate(dO, variables=variables, backend=simulator)
    assert (numpy.isclose(E, numpy.sin(angle(variables) * (numpy.pi)), atol=1.e-4))
    assert (numpy.isclose(dE, numpy.pi * numpy.cos(angle(variables) * (numpy.pi)), atol=1.e-4))


@pytest.mark.parametrize("simulator", simulators)
@pytest.mark.parametrize("power", numpy.random.uniform(0.0, 2.0, 1))
@pytest.mark.parametrize("controlled", [False, True])
def test_gradient_Z(simulator, power, controlled):
    qubit = 0
    control = 1
    angle = Variable(name="angle")
    if controlled:
        U = gates.X(target=control) + gates.H(target=qubit) + gates.Z(target=qubit, power=angle,
                                                                      control=control) + gates.H(target=qubit)
    else:
        U = gates.H(target=qubit) + gates.Z(target=qubit, power=angle) + gates.H(target=qubit)
    angle = Variable(name="angle")
    variables = {angle: power}
    H = paulis.Y(qubit=qubit)
    O = ExpectationValue(U=U, H=H)
    E = simulate(O, variables=variables, backend=simulator)
    dO = grad(objective=O, variable=angle)
    dE = simulate(dO, variables=variables, backend=simulator)
    assert (numpy.isclose(E, -numpy.sin(angle(variables) * (numpy.pi)), atol=1.e-4))
    assert (numpy.isclose(dE, -numpy.pi * numpy.cos(angle(variables) * (numpy.pi)), atol=1.e-4))


@pytest.mark.parametrize("simulator", simulators)
@pytest.mark.parametrize("power", numpy.random.uniform(0.0, 2.0, 1))
@pytest.mark.parametrize("controls", [1, 2, 3])
def test_gradient_deep_controlled_Z(simulator, power, controls):
    if controls > 2 and simulator == "qiskit":
        # does not work yet
        return
    qubit = 0
    control = [i for i in range(1, controls + 1)]
    angle = Variable(name="angle")
    U = gates.X(target=control) + gates.H(target=qubit) + gates.Z(target=qubit, power=angle, control=control) + gates.H(
        target=qubit)
    angle = Variable(name="angle")
    variables = {angle: power}
    H = paulis.Y(qubit=qubit)
    O = ExpectationValue(U=U, H=H)
    E = simulate(O, variables=variables, backend=simulator)
    dO = grad(objective=O, variable=angle)
    dE = simulate(dO, variables=variables, backend=simulator)
    assert (numpy.isclose(E, -numpy.sin(angle(variables) * (numpy.pi)), atol=1.e-4))
    assert (numpy.isclose(dE, -numpy.pi * numpy.cos(angle(variables) * (numpy.pi)), atol=1.e-4))


@pytest.mark.parametrize("simulator", simulators)
@pytest.mark.parametrize("power", numpy.random.uniform(0.0, 2.0 * numpy.pi, 1))
@pytest.mark.parametrize("controlled", [False, True])
def test_gradient_H(simulator, power, controlled):
    qubit = 0
    control = 1
    angle = Variable(name="angle")
    variables = {angle: power}

    H = paulis.X(qubit=qubit)
    if not controlled:
        U = gates.H(target=qubit, power=angle)
    else:
        U = gates.X(target=control) + gates.H(target=qubit, control=control, power=angle)

    O = ExpectationValue(U=U, H=H)
    E = simulate(O, variables=variables, backend=simulator)
    assert (numpy.isclose(E, -numpy.cos(angle(variables) * (numpy.pi)) / 2 + 0.5, atol=1.e-4))
    dO = grad(objective=O, variable=angle)
    dE = simulate(dO, variables=variables, backend=simulator)

    assert (numpy.isclose(dE, numpy.pi * numpy.sin(angle(variables) * (numpy.pi)) / 2, atol=1.e-4))


@pytest.mark.parametrize("simulator", simulators)
@pytest.mark.parametrize("power", numpy.random.uniform(0.0, 2.0 * numpy.pi, 1))
@pytest.mark.parametrize("controls", [1, 2, 3])
def test_gradient_deep_H(simulator, power, controls):
    if controls > 2 and simulator == "qiskit":
        # does not work yet
        return
    qubit = 0
    angle = Variable(name="angle")
    variables = {angle: power}
    control = [i for i in range(1, controls + 1)]
    H = paulis.X(qubit=qubit)

    U = gates.X(target=control) + gates.H(target=qubit, control=control, power=angle)

    O = ExpectationValue(U=U, H=H)
    E = simulate(O, variables=variables, backend=simulator)
    assert (numpy.isclose(E, -numpy.cos(angle(variables) * (numpy.pi)) / 2 + 0.5, atol=1.e-4))
    dO = grad(objective=O, variable=angle)
    dE = simulate(dO, variables=variables, backend=simulator)

    assert (numpy.isclose(dE, numpy.pi * numpy.sin(angle(variables) * (numpy.pi)) / 2, atol=1.e-4))


def test_qubit_excitations():
    H = paulis.Projector("1.0*|100>")
    U1 = gates.X(0) + gates.QubitExcitation(target=[0, 1], angle="a", assume_real=True)
    U2 = gates.X(0) + gates.Trotterized(generators=[U1.gates[1].make_generator()], angles=["a"], steps=1)
    E1 = ExpectationValue(H=H, U=U1)
    E2 = ExpectationValue(H=H, U=U2)
    dE1 = grad(E1, "a")
    dE2 = grad(E2, "a")

    for a in numpy.random.uniform(-numpy.pi, numpy.pi, 5):
        a = float(a)
        variables = {"a": a}
        wfn1 = simulate(U1, variables=variables)
        wfn2 = simulate(U2, variables=variables)
        F = numpy.abs(wfn1.inner(wfn2)) ** 2
        assert numpy.isclose(F, 1.0, 1.e-4)
        eval1 = simulate(dE1, variables=variables)
        eval2 = simulate(dE2, variables=variables)
        assert numpy.isclose(eval1, eval2, 1.e-4)

    H = paulis.Projector("1.0*|0110>")
    U1 = gates.X([1, 2]) + gates.QubitExcitation(target=[0, 1, 2, 3], angle="a", assume_real=True)
    U2 = gates.X([1, 2]) + gates.Trotterized(generators=[U1.gates[2].make_generator()], angles=["a"], steps=1)
    E1 = ExpectationValue(H=H, U=U1)
    E2 = ExpectationValue(H=H, U=U2)
    dE1 = grad(E1, "a")
    dE2 = grad(E2, "a")

    for a in numpy.random.uniform(-numpy.pi, numpy.pi, 5):
        a = float(a)
        variables = {"a": a}
        wfn1 = simulate(U1, variables=variables)
        wfn2 = simulate(U2, variables=variables)
        F = numpy.abs(wfn1.inner(wfn2)) ** 2
        assert numpy.isclose(F, 1.0, 1.e-4)
        eval1 = simulate(dE1, variables=variables)
        eval2 = simulate(dE2, variables=variables)
        print(dE1.get_expectationvalues()[1].U)
        print(eval1)
        print(eval2)
        assert numpy.isclose(eval1, eval2, atol=1.e-4)
