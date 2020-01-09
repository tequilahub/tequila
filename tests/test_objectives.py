from tequila.circuit import gates
from tequila.objective import Objective, ExpectationValue
from tequila.circuit.variable import Variable
from tequila.hamiltonian import paulis
from tequila.circuit.gradient import grad
import jax.numpy as np
import numpy
import pytest
from jax import grad as jg
from tequila import simulators


### these 8 tests test add,mult,div, and power, with the expectationvalue on the left and right.

@pytest.mark.parametrize("simulator", simulators.get_all_wfn_simulators())
def test_l_addition(simulator):
    angle1 = Variable(name="angle1", value=(numpy.random.randint(0, 1000) / 1000.0 * (numpy.pi / 2.0)))
    qubit = 0
    control = 1
    H1 = paulis.X(qubit=qubit)
    U1 = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle1)
    e1 = ExpectationValue(U=U1, H=H1)
    added = e1 + 1
    val = simulator().simulate_objective(added)
    en1 = simulator().simulate_objective(e1) + 1.
    an1 = np.sin(angle1()) + 1.
    assert bool(np.isclose(val, en1)) is True
    assert bool(np.isclose(val, an1)) is True


@pytest.mark.parametrize("simulator", simulators.get_all_wfn_simulators())
def test_r_addition(simulator):
    angle1 = Variable(name="angle1", value=(numpy.random.randint(0, 1000) / 1000.0 * (numpy.pi / 2.0)))
    qubit = 0
    control = 1
    H1 = paulis.X(qubit=qubit)
    U1 = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle1)
    e1 = ExpectationValue(U=U1, H=H1)
    added = 1 + e1
    val = simulator().simulate_objective(added)
    en1 = 1 + simulator().simulate_objective(e1)
    an1 = np.sin(angle1()) + 1.
    assert bool(np.isclose(val, en1)) is True
    assert bool(np.isclose(val, an1)) is True


@pytest.mark.parametrize("simulator", simulators.get_all_wfn_simulators())
def test_l_multiplication(simulator):
    angle1 = Variable(name="angle1", value=(numpy.random.randint(0, 1000) / 1000.0 * (numpy.pi / 2.0)))
    qubit = 0
    control = 1
    H1 = paulis.X(qubit=qubit)
    U1 = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle1)
    e1 = ExpectationValue(U=U1, H=H1)
    added = e1 * 2
    val = simulator().simulate_objective(added)
    en1 = 2 * simulator().simulate_objective(e1)
    an1 = np.sin(angle1()) * 2
    assert bool(np.isclose(val, en1)) is True
    assert bool(np.isclose(val, an1)) is True


@pytest.mark.parametrize("simulator", simulators.get_all_wfn_simulators())
def test_r_multiplication(simulator):
    angle1 = Variable(name="angle1", value=(numpy.random.randint(0, 1000) / 1000.0 * (numpy.pi / 2.0)))
    qubit = 0
    control = 1
    H1 = paulis.X(qubit=qubit)
    U1 = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle1)
    e1 = ExpectationValue(U=U1, H=H1)
    added = 2 * e1
    val = simulator().simulate_objective(added)
    en1 = 2 * simulator().simulate_objective(e1)
    an1 = np.sin(angle1()) * 2
    assert bool(np.isclose(val, en1)) is True
    assert bool(np.isclose(val, an1)) is True


@pytest.mark.parametrize("simulator", simulators.get_all_wfn_simulators())
def test_l_division(simulator):
    angle1 = Variable(name="angle1", value=(numpy.random.randint(0, 1000) / 1000.0 * (numpy.pi / 2.0)))
    qubit = 0
    control = 1
    H1 = paulis.X(qubit=qubit)
    U1 = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle1)
    e1 = ExpectationValue(U=U1, H=H1)
    added = e1 / 2
    val = simulator().simulate_objective(added)
    en1 = simulator().simulate_objective(e1) / 2
    an1 = np.sin(angle1()) / 2.
    assert bool(np.isclose(val, en1)) is True
    assert bool(np.isclose(val, an1)) is True


@pytest.mark.parametrize("simulator", simulators.get_all_wfn_simulators())
def test_r_division(simulator):
    angle1 = Variable(name="angle1", value=(numpy.random.randint(0, 1000) / 1000.0 * (numpy.pi / 2.0)))
    qubit = 0
    control = 1
    H1 = paulis.X(qubit=qubit)
    U1 = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle1)
    e1 = ExpectationValue(U=U1, H=H1)
    added = 2 / e1
    val = simulator().simulate_objective(added)
    en1 = 2 / simulator().simulate_objective(e1)
    an1 = 2 / np.sin(angle1())
    assert bool(np.isclose(val, en1)) is True
    assert bool(np.isclose(val, an1)) is True


@pytest.mark.parametrize("simulator", simulators.get_all_wfn_simulators())
def test_l_power(simulator):
    angle1 = Variable(name="angle1", value=(numpy.random.randint(0, 1000) / 1000.0 * (numpy.pi / 2.0)))
    qubit = 0
    control = 1
    H1 = paulis.X(qubit=qubit)
    U1 = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle1)
    e1 = ExpectationValue(U=U1, H=H1)
    added = e1 ** 2
    val = simulator().simulate_objective(added)
    en1 = simulator().simulate_objective(e1) ** 2
    an1 = np.sin(angle1()) ** 2.
    assert bool(np.isclose(val, en1)) is True
    assert bool(np.isclose(val, an1)) is True


@pytest.mark.parametrize("simulator", simulators.get_all_wfn_simulators())
def test_r_power(simulator):
    angle1 = Variable(name="angle1", value=(numpy.random.randint(0, 1000) / 1000.0 * (numpy.pi / 2.0)))
    qubit = 0
    control = 1
    H1 = paulis.X(qubit=qubit)
    U1 = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle1)
    e1 = ExpectationValue(U=U1, H=H1)
    added = 2 ** e1
    val = simulator().simulate_objective(added)
    en1 = 2 ** simulator().simulate_objective(e1)
    an1 = 2. ** np.sin(angle1())
    assert bool(np.isclose(val, en1)) is True
    assert bool(np.isclose(val, an1)) is True


### these four tests test mutual operations. We skip minus cuz it's not needed.

@pytest.mark.parametrize("simulator", simulators.get_all_wfn_simulators())
def test_ex_addition(simulator):
    angle1 = Variable(name="angle1", value=(numpy.random.randint(0, 1000) / 1000.0 * (numpy.pi / 2.0)))
    angle2 = Variable(name="angle2", value=(numpy.random.randint(0, 1000) / 1000.0 * (numpy.pi / 2.0)))
    qubit = 0
    control = 1
    H1 = paulis.X(qubit=qubit)
    U1 = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle1)
    e1 = ExpectationValue(U=U1, H=H1)
    H2 = paulis.Y(qubit=qubit)
    U2 = gates.X(target=control) + gates.Rx(target=qubit, control=control, angle=angle2)
    e2 = ExpectationValue(U=U2, H=H2)
    added = e1 + e2
    val = simulator().simulate_objective(added)
    en1 = simulator().simulate_objective(e1)
    en2 = simulator().simulate_objective(e2)
    an1 = np.sin(angle1())
    an2 = -np.sin(angle2())
    assert bool(np.isclose(val, en1 + en2)) is True
    assert bool(np.isclose(val, an1 + an2)) is True


@pytest.mark.parametrize("simulator", simulators.get_all_wfn_simulators())
def test_ex_multiplication(simulator):
    angle1 = Variable(name="angle1", value=(numpy.random.randint(0, 1000) / 1000.0 * (numpy.pi / 2.0)))
    angle2 = Variable(name="angle2", value=(numpy.random.randint(0, 1000) / 1000.0 * (numpy.pi / 2.0)))
    qubit = 0
    control = 1
    H1 = paulis.X(qubit=qubit)
    U1 = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle1)
    e1 = ExpectationValue(U=U1, H=H1)
    H2 = paulis.Y(qubit=qubit)
    U2 = gates.X(target=control) + gates.Rx(target=qubit, control=control, angle=angle2)
    e2 = ExpectationValue(U=U2, H=H2)
    added = e1 * e2
    val = simulator().simulate_objective(added)
    en1 = simulator().simulate_objective(e1)
    en2 = simulator().simulate_objective(e2)
    an1 = np.sin(angle1())
    an2 = -np.sin(angle2())
    assert bool(np.isclose(val, en1 * en2)) is True
    assert bool(np.isclose(val, an1 * an2)) is True


@pytest.mark.parametrize("simulator", simulators.get_all_wfn_simulators())
def test_ex_division(simulator):
    angle1 = Variable(name="angle1", value=(numpy.random.randint(0, 1000) / 1000.0 * (numpy.pi / 2.0)))
    angle2 = Variable(name="angle2", value=(numpy.random.randint(0, 1000) / 1000.0 * (numpy.pi / 2.0)))
    qubit = 0
    control = 1
    H1 = paulis.X(qubit=qubit)
    U1 = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle1)
    e1 = ExpectationValue(U=U1, H=H1)
    H2 = paulis.Y(qubit=qubit)
    U2 = gates.X(target=control) + gates.Rx(target=qubit, control=control, angle=angle2)
    e2 = ExpectationValue(U=U2, H=H2)
    added = e1 / e2
    val = simulator().simulate_objective(added)
    en1 = simulator().simulate_objective(e1)
    en2 = simulator().simulate_objective(e2)
    an1 = np.sin(angle1())
    an2 = -np.sin(angle2())
    assert bool(np.isclose(val, en1 / en2)) is True
    assert bool(np.isclose(val, an1 / an2)) is True


@pytest.mark.parametrize("simulator", simulators.get_all_wfn_simulators())
def test_ex_power(simulator):
    angle1 = Variable(name="angle1", value=(numpy.random.randint(0, 1000) / 1000.0 * (numpy.pi / 2.0)))
    angle2 = Variable(name="angle2", value=(numpy.random.randint(0, 1000) / 1000.0 * (numpy.pi / 2.0)))
    qubit = 0
    control = 1
    H1 = paulis.X(qubit=qubit)
    U1 = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle1)
    e1 = ExpectationValue(U=U1, H=H1)
    H2 = paulis.Y(qubit=qubit)
    U2 = gates.X(target=control) + gates.Rx(target=qubit, control=control, angle=angle2)
    e2 = ExpectationValue(U=U2, H=H2)
    added = e1 ** e2
    val = simulator().simulate_objective(added)
    en1 = simulator().simulate_objective(e1)
    en2 = simulator().simulate_objective(e2)
    an1 = np.sin(angle1())
    an2 = -np.sin(angle2())
    assert bool(np.isclose(val, en1 ** en2)) is True
    assert bool(np.isclose(val, an1 ** an2)) is True


### these four tests test the mixed Objective,ExpectationValue operations to ensure propriety

@pytest.mark.parametrize("simulator", simulators.get_all_wfn_simulators())
def test_mixed_addition(simulator):
    angle1 = Variable(name="angle1", value=(numpy.random.randint(0, 1000) / 1000.0 * (numpy.pi / 2.0)))
    angle2 = Variable(name="angle2", value=(numpy.random.randint(0, 1000) / 1000.0 * (numpy.pi / 2.0)))
    qubit = 0
    control = 1
    H1 = paulis.X(qubit=qubit)
    U1 = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle1)
    e1 = ExpectationValue(U=U1, H=H1)
    H2 = paulis.Y(qubit=qubit)
    U2 = gates.X(target=control) + gates.Rx(target=qubit, control=control, angle=angle2)
    e2 = ExpectationValue(U=U2, H=H2)
    added = e1 + e2
    val = simulator().simulate_objective(added)
    en1 = simulator().simulate_objective(e1)
    en2 = simulator().simulate_objective(e2)
    an1 = np.sin(angle1())
    an2 = -np.sin(angle2())
    assert bool(np.isclose(val, en1 + en2)) is True
    assert bool(np.isclose(val, float(an1 + an2))) is True


@pytest.mark.parametrize("simulator", simulators.get_all_wfn_simulators())
def test_mixed_multiplication(simulator):
    angle1 = Variable(name="angle1", value=(numpy.random.randint(0, 1000) / 1000.0 * (numpy.pi / 2.0)))
    angle2 = Variable(name="angle2", value=(numpy.random.randint(0, 1000) / 1000.0 * (numpy.pi / 2.0)))
    qubit = 0
    control = 1
    H1 = paulis.X(qubit=qubit)
    U1 = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle1)
    e1 = ExpectationValue(U=U1, H=H1)
    H2 = paulis.Y(qubit=qubit)
    U2 = gates.X(target=control) + gates.Rx(target=qubit, control=control, angle=angle2)
    e2 = ExpectationValue(U=U2, H=H2)
    added = e1 * e2
    val = simulator().simulate_objective(added)
    en1 = simulator().simulate_objective(e1)
    en2 = simulator().simulate_objective(e2)
    an1 = np.sin(angle1())
    an2 = -np.sin(angle2())
    assert bool(np.isclose(val, en1 * en2)) is True
    assert bool(np.isclose(val, an1 * an2)) is True


@pytest.mark.parametrize("simulator", simulators.get_all_wfn_simulators())
def test_mixed_division(simulator):
    angle1 = Variable(name="angle1", value=(numpy.random.randint(0, 1000) / 1000.0 * (numpy.pi / 2.0)))
    angle2 = Variable(name="angle2", value=(numpy.random.randint(0, 1000) / 1000.0 * (numpy.pi / 2.0)))
    qubit = 0
    control = 1
    H1 = paulis.X(qubit=qubit)
    U1 = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle1)
    e1 = ExpectationValue(U=U1, H=H1)
    H2 = paulis.Y(qubit=qubit)
    U2 = gates.X(target=control) + gates.Rx(target=qubit, control=control, angle=angle2)
    e2 = ExpectationValue(U=U2, H=H2)
    added = e1 / e2
    val = simulator().simulate_objective(added)
    en1 = simulator().simulate_objective(e1)
    en2 = simulator().simulate_objective(e2)
    an1 = np.sin(angle1())
    an2 = -np.sin(angle2())
    assert bool(np.isclose(val, en1 / en2)) is True
    assert bool(np.isclose(val, an1 / an2)) is True


@pytest.mark.parametrize("simulator", simulators.get_all_wfn_simulators())
def test_mixed_power(simulator):
    angle1 = Variable(name="angle1", value=(numpy.random.randint(0, 1000) / 1000.0 * (numpy.pi / 2.0)))
    angle2 = Variable(name="angle2", value=(numpy.random.randint(0, 1000) / 1000.0 * (numpy.pi / 2.0)))
    qubit = 0
    control = 1
    H1 = paulis.X(qubit=qubit)
    U1 = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle1)
    e1 = ExpectationValue(U=U1, H=H1)
    H2 = paulis.Y(qubit=qubit)
    U2 = gates.X(target=control) + gates.Rx(target=qubit, control=control, angle=angle2)
    e2 = ExpectationValue(U=U2, H=H2)
    added = e1 ** e2
    val = simulator().simulate_objective(added)
    en1 = simulator().simulate_objective(e1)
    en2 = simulator().simulate_objective(e2)
    an1 = np.sin(angle1())
    an2 = -np.sin(angle2())
    assert bool(np.isclose(val, en1 ** en2)) is True
    assert bool(np.isclose(val, an1 ** an2)) is True

@pytest.mark.parametrize("simulator", simulators.get_all_wfn_simulators())
@pytest.mark.parametrize('op',[np.add,np.subtract,np.float_power,np.true_divide,np.multiply])
def test_heterogeneous_operations_l(simulator,op):
    angle1 = Variable(name="angle1", value=(numpy.random.randint(100, 1000) / 1000.0 * (numpy.pi / 2.0)))
    angle2 = Variable(name="angle2", value=(numpy.random.randint(100, 1000) / 1000.0 * (numpy.pi / 2.0)))
    qubit = 0
    control = 1
    H2 = paulis.X(qubit=qubit)
    U2 = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle2)
    e2 = ExpectationValue(U=U2, H=H2)
    added = Objective(args=[angle1,e2.args[0]],transformation=op)
    val = simulator().simulate_objective(added)
    en2 = simulator().simulate_objective(e2)
    an1 = angle1()
    an2 = np.sin(angle2())
    assert bool(np.isclose(val, float(op(an1,en2)))) is True
    assert np.isclose(en2,an2)


@pytest.mark.parametrize("simulator", simulators.get_all_wfn_simulators())
@pytest.mark.parametrize('op',[np.add,np.subtract,np.true_divide,np.multiply])
def test_heterogeneous_operations_r(simulator,op):
    ### the reason we don't test float power here is that it keeps coming up NAN, because the argument is too small
    angle1 = Variable(name="angle1", value=(numpy.random.randint(100, 1000) / 1000.0 * (numpy.pi / 2.0)))
    angle2 = Variable(name="angle2", value=(numpy.random.randint(100, 1000) / 1000.0 * (numpy.pi / 2.0)))
    qubit = 0
    control = 1
    H1 = paulis.Y(qubit=qubit)
    U1 = gates.X(target=control) + gates.Rx(target=qubit, control=control, angle=angle1)
    e1 = ExpectationValue(U=U1, H=H1)
    added = Objective(args=[e1.args[0],angle2],transformation=op)
    val = simulator().simulate_objective(added)
    en1 = simulator().simulate_objective(e1)
    an1 = -np.sin(angle1())
    an2 = angle2()
    assert bool(np.isclose(val, float(op(en1,an2)))) is True
    assert np.isclose(en1,an1)


