from tequila.circuit import gates
from tequila.circuit.gradient import grad
from tequila.objective import Objective, ExpectationValue
from tequila.circuit.variable import Variable
from tequila.hamiltonian import paulis
from tequila.simulators.simulator_qiskit import SimulatorQiskit
from tequila.simulators.simulator_cirq import SimulatorCirq
import numpy
import pytest


@pytest.mark.parametrize("simulator", [SimulatorCirq, SimulatorQiskit])
@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("angle_value", [(i / 1000.0) * (numpy.pi / 2.0) for i in
                                   numpy.random.randint(0, 1000, 3)])
def test_gradient_UY_HX(simulator, angle_value, controlled, silent=True):
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
        U = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle)
    else:
        U = gates.Ry(target=qubit, angle=angle)
    O = ExpectationValue(U=U, H=H)
    E = simulator().measure_objective(objective=O, variables=variables, samples=10000)
    dO = grad(objective=O, variable=angle)
    dE = simulator().measure_objective(objective=dO, variables=variables, samples=10000)
    assert (numpy.isclose(E, numpy.sin(angle(variables)), atol=0.03))
    assert (numpy.isclose(dE, numpy.cos(angle(variables)), atol=0.03))
    if not silent:
        print("E         =", E)
        print("sin(angle)=", numpy.sin(angle()))
        print("dE        =", dE)
        print("cos(angle)=", numpy.cos(angle()))


@pytest.mark.parametrize("simulator", [SimulatorCirq, SimulatorQiskit])
@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("angle_value", [(i / 1000.0) * (numpy.pi / 2.0) for i in
                                   numpy.random.randint(0, 1000, 3)])
def test_gradient_UX_HY(simulator, angle_value, controlled, silent=False):
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
        U = gates.X(target=control) + gates.Rx(target=qubit, control=control, angle=angle)
    else:
        U = gates.Rx(target=qubit, angle=angle)
    O = ExpectationValue(U=U, H=H)
    E = simulator().measure_objective(objective=O, variables=variables, samples=10000)
    dO = grad(objective=O, variable='angle')
    dE = simulator().measure_objective(objective=dO,variables=variables, samples=10000)
    assert (numpy.isclose(E, -numpy.sin(angle(variables)), atol=0.03))
    assert (numpy.isclose(dE, -numpy.cos(angle(variables)), atol=0.03))
    if not silent:
        print("E         =", E)
        print("-sin(angle)=", -numpy.sin(angle(variables)))
        print("dE        =", dE)
        print("-cos(angle)=", -numpy.cos(angle(variables)))


@pytest.mark.parametrize("simulator", [SimulatorCirq])
@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("angle_value", [(i / 1000.0) * (numpy.pi / 2.0) for i in
                                   numpy.random.randint(0, 1000, 3)])
def test_gradient_UY_HX_wfnsim(simulator, angle_value, controlled, silent=True):
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
        U = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle)
    else:
        U = gates.Ry(target=qubit, angle=angle)
    O = ExpectationValue(U=U, H=H)
    E = simulator().simulate_objective(O, variables)
    dO = grad(objective=O, variable='angle')
    print(dO)
    dE = simulator().simulate_objective(dO, variables=variables)
    E = numpy.float(E)  # for isclose
    dE = numpy.float(dE)  # for isclose
    assert (numpy.isclose(E, numpy.sin(angle(variables)), atol=0.0001))
    assert (numpy.isclose(dE, numpy.cos(angle(variables)), atol=0.0001))
    if not silent:
        print("E         =", E)
        print("sin(angle)=", numpy.sin(angle(variables)))
        print("dE        =", dE)
        print("cos(angle)=", numpy.cos(angle(variables)))


@pytest.mark.parametrize("simulator", [SimulatorCirq])
@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("angle", [(i / 1000.0) * (numpy.pi / 2.0) for i in
                                   numpy.random.randint(0, 1000, 3)])
def test_gradient_UX_HY_wfnsim(simulator, angle, controlled, silent=True):
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
        U = gates.X(target=control) + gates.Rx(target=qubit, control=control, angle=angle)
    else:
        U = gates.Rx(target=qubit, angle=angle)
    O = ExpectationValue(U=U, H=H)
    E = simulator().simulate_objective(O, variables=variables)
    dO = grad(objective=O, variable=angle)
    dE = simulator().simulate_objective(dO,variables=variables)
    assert (numpy.isclose(E, -numpy.sin(angle(variables)), atol=0.0001))
    assert (numpy.isclose(dE, -numpy.cos(angle(variables)), atol=0.0001))
    if not silent:
        print("E         =", E)
        print("-sin(angle)=", -numpy.sin(angle(variables)))
        print("dE        =", dE)
        print("-cos(angle)=", -numpy.cos(angle(variables)))


@pytest.mark.parametrize("simulator", [SimulatorCirq])
@pytest.mark.parametrize("power", [(i / 1000.0) for i in
                                   numpy.random.randint(0, 1000, 3)])
def test_gradient_X(simulator, power):
    qubit = 0
    angle = Variable(name="angle")
    variables = {angle: power}
    H = paulis.Y(qubit=qubit)
    U = gates.X(target=qubit, power=angle)
    O = ExpectationValue(U=U, H=H)
    E = simulator().measure_objective(objective=O, variables=variables, samples=10000)
    dO = grad(objective=O, variable=angle)
    dE = simulator().measure_objective(objective=dO, variables=variables, samples=10000)
    assert (numpy.isclose(E, -numpy.sin(angle(variables) * (numpy.pi)), atol=0.03))
    assert (numpy.isclose(dE, -numpy.cos(angle(variables) * (numpy.pi)), atol=0.03))


@pytest.mark.parametrize("simulator", [SimulatorCirq])
@pytest.mark.parametrize("power", [(i / 1000.0) for i in
                                   numpy.random.randint(0, 1000, 3)])
def test_gradient_Y(simulator, power):
    qubit = 0

    angle = Variable(name="angle")
    variables = {angle:1.0}

    H = paulis.X(qubit=qubit)
    U = gates.Y(target=qubit, power=angle)
    O = ExpectationValue(U=U, H=H)
    E = simulator().measure_objective(objective=O, variables=variables, samples=10000)
    dO = grad(objective=O, variable=angle)
    dE = simulator().measure_objective(objective=dO, variables=variables, samples=10000)
    assert (numpy.isclose(E, numpy.sin(angle(variables) * (numpy.pi)), atol=0.03))
    assert (numpy.isclose(dE, numpy.cos(angle(variables) * (numpy.pi)), atol=0.03))
