from openvqe.circuit import gates, Variable
from openvqe.circuit.gradient import grad
from openvqe.objective import Objective
from openvqe.hamiltonian import paulis
from openvqe.simulators.simulator_qiskit import SimulatorQiskit
from openvqe.simulators.simulator_cirq import SimulatorCirq
#from openvqe.simulators.simulator_qulacs import SimulatorQulacs
import numpy
import pytest


@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("angle", [Variable(name="angle", value=(i / 1000.0 )* (numpy.pi / 2.0)) for i in
                                   numpy.random.randint(0, 1000, 3)])
def test_gradient_UY_HX(angle, controlled, silent=True):
    # case X Y
    # U = cos(angle/2) + sin(-angle/2)*i*Y
    # <0|Ud H U |0> = cos^2(angle/2)*<0|X|0>
    # + sin^2(-angle/2) <0|YXY|0>
    # + cos(angle/2)*sin(angle/2)*i<0|XY|0>
    # + sin(-angle/2)*cos(angle/2)*(-i) <0|YX|0>
    # = cos^2*0 + sin^2*0 + cos*sin*i(<0|[XY,YX]|0>)
    # = 0.5*sin(-angle)*i <0|[XY,YX]|0> = -0.5*sin(angle)*i * 2 i <0|Z|0>
    # = sin(angle)

    qubit = 0
    H = paulis.X(qubit=qubit)
    if controlled:
        control = 1
        U = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle)
    else:
        U = gates.Ry(target=qubit, angle=angle)
    O = Objective(unitaries=U, observable=H)
    E = SimulatorCirq().measure_objective(objective=O, samples=100000)
    dO = grad(obj=O)
    assert (len(dO) == 1)
    for k,v in dO.items():
        dE = SimulatorCirq().measure_objective(objective=v, samples=100000)
    assert (numpy.isclose(E, numpy.sin(angle()), atol=0.03))
    assert (numpy.isclose(dE, numpy.cos(angle()), atol=0.03))
    if not silent:
        print("E         =", E)
        print("sin(angle)=", numpy.sin(angle()))
        print("dE        =", dE)
        print("cos(angle)=", numpy.cos(angle()))


@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("angle", [Variable(name="angle", value=(i / 1000.0 )* (numpy.pi / 2.0)) for i in
                                   numpy.random.randint(0, 1000, 3)])
def test_gradient_UX_HY(angle, controlled, silent=False):
    # case YX
    # U = cos(angle/2) + sin(-angle/2)*i*X
    # O = cos*sin*i*<0|YX|0> + sin*cos*(-i)<0|XY|0>
    #   = 0.5*sin(-angle)*i <0|[YX,XY]|0>
    #   = -sin(angle)

    qubit = 0
    H = paulis.Y(qubit=qubit)
    if controlled:
        control = 1
        U = gates.X(target=control) + gates.Rx(target=qubit, control=control, angle=angle)
    else:
        U = gates.Rx(target=qubit, angle=angle)
    O = Objective(unitaries=U, observable=H)
    E = SimulatorCirq().measure_objective(objective=O, samples=100000)
    dO = grad(obj=O)
    assert (len(dO) == 1)
    for k,v in dO.items():
        dE = SimulatorCirq().measure_objective(objective=v, samples=100000)
    assert (numpy.isclose(E, -numpy.sin(angle()), atol=0.03))
    assert (numpy.isclose(dE, -numpy.cos(angle()), atol=0.03))
    if not silent:
        print("E         =", E)
        print("-sin(angle)=", -numpy.sin(angle()))
        print("dE        =", dE)
        print("-cos(angle)=", -numpy.cos(angle()))


@pytest.mark.parametrize("simulator", [SimulatorCirq])
@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("angle", [Variable(name="angle", value=(i / 1000.0 )* (numpy.pi / 2.0)) for i in
                                   numpy.random.randint(0, 1000, 3)])
def test_gradient_UY_HX_wfnsim(simulator, angle, controlled, silent=True):
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

    qubit = 0
    H = paulis.X(qubit=qubit)
    if controlled:
        control = 1
        U = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle)
    else:
        U = gates.Ry(target=qubit, angle=angle)
    O = Objective(unitaries=U, observable=H)
    E = simulator().simulate_objective(objective=O)
    dO = grad(obj=O)
    assert (len(dO) == 1)
    for k,v in dO.items():
        dE = simulator().simulate_objective(objective=v)
    E = numpy.float(E)  # for isclose
    dE = numpy.float(dE)  # for isclose
    assert (numpy.isclose(E, numpy.sin(angle()), atol=0.0001))
    assert (numpy.isclose(dE, numpy.cos(angle()), atol=0.0001))
    if not silent:
        print("E         =", E)
        print("sin(angle)=", numpy.sin(angle()))
        print("dE        =", dE)
        print("cos(angle)=", numpy.cos(angle()))


@pytest.mark.parametrize("simulator", [SimulatorCirq])
@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("angle", [Variable(name="angle", value=(i / 1000.0 )* (numpy.pi / 2.0)) for i in
                                   numpy.random.randint(0, 1000, 3)])
def test_gradient_UX_HY_wfnsim(simulator, angle, controlled, silent=True):
    # same as before just with wavefunction simulation

    # case YX
    # U = cos(angle/2) + sin(-angle/2)*i*X
    # O = cos*sin*i*<0|YX|0> + sin*cos*(-i)<0|XY|0>
    #   = 0.5*sin(-angle)*i <0|[YX,XY]|0>
    #   = -sin(angle)

    qubit = 0
    H = paulis.Y(qubit=qubit)
    if controlled:
        control = 1
        U = gates.X(target=control) + gates.Rx(target=qubit, control=control, angle=angle)
    else:
        U = gates.Rx(target=qubit, angle=angle)
    O = Objective(unitaries=U, observable=H)
    E = simulator().simulate_objective(objective=O)
    dO = grad(obj=O)
    assert (len(dO) == 1)
    for k,v in dO.items():
        dE = simulator().simulate_objective(objective=v)
        assert (numpy.isclose(E, -numpy.sin(angle()), atol=0.0001))
        assert (numpy.isclose(dE, -numpy.cos(angle()), atol=0.0001))
    if not silent:
        print("E         =", E)
        print("-sin(angle)=", -numpy.sin(angle()))
        print("dE        =", dE)
        print("-cos(angle)=", -numpy.cos(angle()))
