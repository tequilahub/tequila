import tequila.simulators.simulator_api
from tequila.circuit import gates
from tequila import Objective, ExpectationValue, VectorObjective
from tequila.objective.objective import Variable
from tequila.hamiltonian import paulis
from tequila.circuit.gradient import grad
from tequila import numpy as np
import numpy
import pytest
import numbers
import tequila as tq
from tequila.simulators.simulator_api import simulate

# Get QC backends for parametrized testing
import select_backends
simulators = select_backends.get()
samplers = select_backends.get(sampler=True)

def test_non_quantum():
    E = tq.Objective()
    E += 1.0
    E = E + 2.0
    E *= 2.0
    E = E * 2.0
    E = 1.0 * E
    E -= 1.0
    E = 1.0 + E
    E = E.apply(lambda x: x/3.0)
    assert E() == 2.0*2.0*(1.0 + 2.0)/3.0

def test_return_type():
    U = tq.gates.H(0)
    H = tq.paulis.X(0)
    E = tq.ExpectationValue(H=H,U=U)
    result = tq.simulate(E)
    assert isinstance(result, numbers.Number)

def test_qubit_maps():
    qubit_map = {0:1, 1:2, 2:3}

    H1 = tq.paulis.X(0) + tq.paulis.Z(0)*tq.paulis.Z(1) + tq.paulis.Y(2)
    U1 = tq.gates.Ry(angle="a",target=0) + tq.gates.CNOT(0,1) + tq.gates.H(target=2)
    E1 = tq.ExpectationValue(H=H1, U=U1)

    H2 = tq.paulis.X(qubit_map[0]) + tq.paulis.Z(qubit_map[0]) * tq.paulis.Z(qubit_map[1]) + tq.paulis.Y(qubit_map[2])
    U2 = tq.gates.Ry(angle="a", target=qubit_map[0]) + tq.gates.CNOT(qubit_map[0], qubit_map[1]) + tq.gates.H(target=qubit_map[2])
    E2 = tq.ExpectationValue(H=H2, U=U2)

    E3 = E1.map_qubits(qubit_map=qubit_map)

    for angle in numpy.random.uniform(0.0, 10.0, 10):
        variables = {"a":angle}
        assert np.isclose(tq.simulate(E2, variables=variables), tq.simulate(E3, variables=variables))

def test_variable_map():
    x = tq.Variable("x")
    y = tq.Variable("y")
    z = tq.Variable("z")

    v = x**2+y+z/2
    U = tq.gates.Ry(angle="a", target=0) + tq.gates.Rx(angle="b", target=1) + tq.gates.Rz(angle="c", target=2) + tq.gates.H(angle="d", target=3) + tq.gates.ExpPauli(paulistring="X(0)Y(1)Z(2)", angle="e")
    U+= tq.gates.GeneralizedRotation(angle=v, generator=tq.paulis.X([0,1,2]))

    H = tq.paulis.X([0,1,2,3,4])
    E = tq.ExpectationValue(H=H, U=U)
    O = E**2 + v + E + 2.0

    variables = {"a":"aa", "b":"bb", "c":"cc", "d":"dd", "e":"ee", "x":"xx", "y":"yy", "z":"zz"}
    variables = {tq.assign_variable(k):tq.assign_variable(v) for k,v in variables.items()}

    U2 = U.map_variables(variables=variables)
    assert sorted([str(x) for x in U.extract_variables()]) == sorted([str(x) for x in list(variables.keys())])
    assert sorted([str(x) for x in U2.extract_variables()]) == sorted([str(x) for x in list(variables.values())])

    E2 = E.map_variables(variables=variables)
    assert sorted([str(x) for x in E.extract_variables()]) == sorted([str(x) for x in list(variables.keys())])
    assert sorted([str(x) for x in E2.extract_variables()]) == sorted([str(x) for x in list(variables.values())])

    O2 = O.map_variables(variables=variables)
    assert sorted([str(x) for x in O.extract_variables()]) == sorted([str(x) for x in list(variables.keys())])
    assert sorted([str(x) for x in O2.extract_variables()]) == sorted([str(x) for x in list(variables.values())])





@pytest.mark.parametrize("backend", simulators)
def test_compilation(backend):
    U = gates.X(target=[0,1,2,3,4,5])
    for i in range(10):
        U += gates.Ry(angle=(i,), target=numpy.random.randint(1,5,1)[0])
    U += gates.CZ(0,1) + gates.CNOT(1,2) + gates.CZ(2,3) + gates.CNOT(3,4) + gates.CZ(5,6)
    H = paulis.X(0) + paulis.X(1) + paulis.X(2) + paulis.X(3) + paulis.X(4) + paulis.X(5)
    H += paulis.Z(0) + paulis.Z(1) + paulis.Z(2) + paulis.Z(3) + paulis.Z(4) + paulis.Z(5)
    E = ExpectationValue(H=H, U=U)

    randvals = numpy.random.uniform(1.0, 1.9, 10)
    variables = {(i,): randvals[i] for i in range(10)}
    e0 = simulate(E, variables=variables, backend=backend)

    E2 = E*E
    for i in range(99):
        E2 += E*E

    compiled = tq.compile(E2, variables=variables, backend=backend)
    e2 = compiled(variables=variables)
    assert(E2.count_expectationvalues(unique=True) == 1)
    assert(compiled.count_expectationvalues(unique=True) == 1)
    assert numpy.isclose(100*e0**2, e2)

### these 8 tests test add,mult,div, and power, with the expectationvalue on the left and right.

@pytest.mark.parametrize("simulator", simulators)
def test_l_addition(simulator, value=(numpy.random.randint(0, 1000) / 1000.0 * (numpy.pi / 2.0))):
    angle1 = Variable(name="angle1")
    variables = {angle1: value}
    qubit = 0
    control = 1
    H1 = paulis.X(qubit=qubit)
    U1 = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle1)
    e1 = ExpectationValue(U=U1, H=H1)
    added = e1 + 1
    val = simulate(added, variables=variables, backend=simulator)
    en1 = simulate(e1, variables=variables, backend=simulator) + 1.
    an1 = np.sin(angle1(variables=variables)) + 1.
    assert np.isclose(val, en1, atol=1.e-4)
    assert np.isclose(val, an1, atol=1.e-4)


@pytest.mark.parametrize("simulator", simulators)
def test_r_addition(simulator, value=numpy.random.uniform(0.1, 1.9*numpy.pi, 1)[0]):
    angle1 = Variable(name="angle1")
    variables = {angle1: value}
    qubit = 0
    control = 1
    H1 = paulis.X(qubit=qubit)
    U1 = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle1)
    e1 = ExpectationValue(U=U1, H=H1)
    added = 1 + e1
    val = simulate(added, variables=variables, backend=simulator)
    en1 = 1 + simulate(e1, variables=variables, backend=simulator)
    an1 = np.sin(angle1(variables=variables)) + 1.
    assert np.isclose(val, en1, atol=1.e-4)
    assert np.isclose(val, an1, atol=1.e-4)


@pytest.mark.parametrize("simulator", simulators)
def test_l_multiplication(simulator, value=numpy.random.uniform(0.1, 1.9*numpy.pi, 1)[0]):
    angle1 = Variable(name="angle1")
    variables = {angle1: value}
    qubit = 0
    control = 1
    H1 = paulis.X(qubit=qubit)
    U1 = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle1)
    e1 = ExpectationValue(U=U1, H=H1)
    added = e1 * 2
    val = simulate(added, variables=variables, backend=simulator)
    en1 = 2 * simulate(e1, variables=variables, backend=simulator)
    an1 = np.sin(angle1(variables=variables)) * 2
    assert np.isclose(val, en1, atol=1.e-4)
    assert np.isclose(val, an1, atol=1.e-4)


@pytest.mark.parametrize("simulator", simulators)
def test_r_multiplication(simulator, value=numpy.random.uniform(0.1, 1.9*numpy.pi, 1)[0]):
    angle1 = Variable(name="angle1")
    variables = {angle1: value}
    qubit = 0
    control = 1
    H1 = paulis.X(qubit=qubit)
    U1 = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle1)
    e1 = ExpectationValue(U=U1, H=H1)
    added = 2 * e1
    val = simulate(added, variables=variables, backend=simulator)
    en1 = 2 * simulate(e1, variables=variables, backend=simulator)
    an1 = np.sin(value) * 2
    assert np.isclose(val, en1, atol=1.e-4)
    assert np.isclose(val, an1, atol=1.e-4)


@pytest.mark.parametrize("simulator", simulators)
def test_l_division(simulator, value=numpy.random.uniform(0.1, 1.9*numpy.pi, 1)[0]):
    angle1 = Variable(name="angle1")
    variables = {angle1: value}
    qubit = 0
    control = 1
    H1 = paulis.X(qubit=qubit)
    U1 = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle1)
    e1 = ExpectationValue(U=U1, H=H1)
    added = e1 / 2
    val = simulate(added, variables=variables, backend=simulator)
    en1 = simulate(e1, variables=variables, backend=simulator) / 2
    an1 = np.sin(value) / 2.
    assert np.isclose(val, en1, atol=1.e-4)
    assert np.isclose(val, an1, atol=1.e-4)


@pytest.mark.parametrize("simulator", simulators)
def test_r_division(simulator, value=numpy.random.uniform(0.1, 1.9*numpy.pi, 1)[0]):
    angle1 = Variable(name="angle1")
    variables = {angle1: value}
    qubit = 0
    control = 1
    H1 = paulis.X(qubit=qubit)
    U1 = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle1)
    e1 = ExpectationValue(U=U1, H=H1)
    added = 2 / e1
    val = simulate(added, variables=variables, backend=simulator)
    en1 = 2 / simulate(e1, variables=variables, backend=simulator)
    an1 = 2 / np.sin(value)
    assert np.isclose(val, en1, atol=1.e-4)
    assert np.isclose(val, an1, atol=1.e-4)


@pytest.mark.parametrize("simulator", simulators)
def test_l_power(simulator, value=numpy.random.uniform(0.1, 1.9*numpy.pi, 1)[0]):
    angle1 = Variable(name="angle1")
    variables = {angle1: value}
    qubit = 0
    control = 1
    H1 = paulis.X(qubit=qubit)
    U1 = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle1)
    e1 = ExpectationValue(U=U1, H=H1)
    added = e1 ** 2
    val = simulate(added, variables=variables, backend=simulator)
    en1 = simulate(e1, variables=variables, backend=simulator) ** 2
    an1 = np.sin(angle1(variables=variables)) ** 2.
    assert np.isclose(val, en1, atol=1.e-4)
    assert np.isclose(val, an1, atol=1.e-4)


@pytest.mark.parametrize("simulator", simulators)
def test_r_power(simulator, value=numpy.random.uniform(0.1, 1.9*numpy.pi, 1)[0]):
    angle1 = Variable(name="angle1")
    variables = {angle1: value}
    qubit = 0
    control = 1
    H1 = paulis.X(qubit=qubit)
    U1 = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle1)
    e1 = ExpectationValue(U=U1, H=H1)
    added = 2 ** e1
    val = simulate(added, variables=variables, backend=simulator)
    en1 = 2 ** simulate(e1, variables=variables, backend=simulator)
    an1 = 2. ** np.sin(angle1(variables=variables))
    assert np.isclose(val, en1, atol=1.e-4)
    assert np.isclose(val, an1, atol=1.e-4)


### these four tests test mutual operations. We skip minus cuz it's not needed.

@pytest.mark.parametrize("simulator", simulators)
def test_ex_addition(simulator, value1=(numpy.random.randint(0, 1000) / 1000.0 * (numpy.pi / 2.0)),
                     value2=(numpy.random.randint(0, 1000) / 1000.0 * (numpy.pi / 2.0))):
    angle1 = Variable(name="angle1")
    angle2 = Variable(name="angle2")
    variables = {angle1: value1, angle2: value2}
    qubit = 0
    control = 1
    H1 = paulis.X(qubit=qubit)
    U1 = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle1)
    e1 = ExpectationValue(U=U1, H=H1)
    H2 = paulis.Y(qubit=qubit)
    U2 = gates.X(target=control) + gates.Rx(target=qubit, control=control, angle=angle2)
    e2 = ExpectationValue(U=U2, H=H2)
    added = e1 + e2
    val = simulate(added, variables=variables, backend=simulator)
    en1 = simulate(e1, variables=variables, backend=simulator)
    en2 = simulate(e2, variables=variables, backend=simulator)
    an1 = np.sin(angle1(variables=variables))
    an2 = -np.sin(angle2(variables=variables))
    assert np.isclose(val, en1 + en2, atol=1.e-4)
    assert np.isclose(val, an1 + an2, atol=1.e-4)


@pytest.mark.parametrize("simulator", simulators)
def test_ex_multiplication(simulator, value1=(numpy.random.randint(10, 1000) / 1000.0 * (numpy.pi / 2.0)),
                           value2=(numpy.random.randint(10, 1000) / 1000.0 * (numpy.pi / 2.0))):
    angle1 = Variable(name="angle1")
    angle2 = Variable(name="angle2")
    variables = {angle1: value1, angle2: value2}
    qubit = 0
    control = 1
    H1 = paulis.X(qubit=qubit)
    U1 = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle1)
    e1 = ExpectationValue(U=U1, H=H1)
    H2 = paulis.Y(qubit=qubit)
    U2 = gates.X(target=control) + gates.Rx(target=qubit, control=control, angle=angle2)
    e2 = ExpectationValue(U=U2, H=H2)
    added = e1 * e2
    val = simulate(added, variables=variables, backend=simulator)
    en1 = simulate(e1, variables=variables, backend=simulator)
    en2 = simulate(e2, variables=variables, backend=simulator)
    an1 = np.sin(angle1(variables=variables))
    an2 = -np.sin(angle2(variables=variables))
    assert np.isclose(val, en1 * en2, atol=1.e-4)
    assert np.isclose(val, an1 * an2, atol=1.e-4)


@pytest.mark.parametrize("simulator", simulators)
def test_ex_division(simulator, value1=(numpy.random.randint(10, 1000) / 1000.0 * (numpy.pi / 2.0)),
                     value2=(numpy.random.randint(10, 1000) / 1000.0 * (numpy.pi / 2.0))):
    angle1 = Variable(name="angle1")
    angle2 = Variable(name="angle2")
    variables = {angle1: value1, angle2: value2}
    qubit = 0
    control = 1
    H1 = paulis.X(qubit=qubit)
    U1 = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle1)
    e1 = ExpectationValue(U=U1, H=H1)
    H2 = paulis.Y(qubit=qubit)
    U2 = gates.X(target=control) + gates.Rx(target=qubit, control=control, angle=angle2)
    e2 = ExpectationValue(U=U2, H=H2)
    added = e1 / e2
    val = simulate(added, variables=variables, backend=simulator)
    en1 = simulate(e1, variables=variables, backend=simulator)
    en2 = simulate(e2, variables=variables, backend=simulator)
    an1 = np.sin(angle1(variables=variables))
    an2 = -np.sin(angle2(variables=variables))
    assert np.isclose(val, en1 / en2, atol=1.e-4)
    assert np.isclose(val, an1 / an2, atol=1.e-4)


@pytest.mark.parametrize("simulator", simulators)
def test_ex_power(simulator, value1=(numpy.random.randint(10, 1000) / 1000.0 * (numpy.pi / 2.0)),
                  value2=(numpy.random.randint(10, 1000) / 1000.0 * (numpy.pi / 2.0))):
    angle1 = Variable(name="angle1")
    angle2 = Variable(name="angle2")
    variables = {angle1: value1, angle2: value2}
    qubit = 0
    control = 1
    H1 = paulis.X(qubit=qubit)
    U1 = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle1)
    e1 = ExpectationValue(U=U1, H=H1)
    H2 = paulis.Y(qubit=qubit)
    U2 = gates.X(target=control) + gates.Rx(target=qubit, control=control, angle=angle2)
    e2 = ExpectationValue(U=U2, H=H2)
    added = (e1+1.e-4) ** e2 # avoid zero division
    val = simulate(added, variables=variables, backend=simulator)
    en1 = simulate(e1, variables=variables, backend=simulator)
    en2 = simulate(e2, variables=variables, backend=simulator)
    an1 = np.sin(angle1(variables=variables))
    an2 = -np.sin(angle2(variables=variables))
    assert np.isclose(val, (en1+1.e-4) ** en2, atol=1.e-4)
    assert np.isclose(val, (an1+1.e-4) ** an2, atol=1.e-4)


### these four tests test the mixed Objective,ExpectationValue operations to ensure propriety

@pytest.mark.parametrize("simulator", simulators)
def test_mixed_addition(simulator, value1=(numpy.random.randint(0, 1000) / 1000.0 * (numpy.pi / 2.0)),
                        value2=(numpy.random.randint(0, 1000) / 1000.0 * (numpy.pi / 2.0))):
    angle1 = Variable(name="angle1")
    angle2 = Variable(name="angle2")
    variables = {angle1: value1, angle2: value2}
    qubit = 0
    control = 1
    H1 = paulis.X(qubit=qubit)
    U1 = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle1)
    e1 = ExpectationValue(U=U1, H=H1)
    H2 = paulis.Y(qubit=qubit)
    U2 = gates.X(target=control) + gates.Rx(target=qubit, control=control, angle=angle2)
    e2 = ExpectationValue(U=U2, H=H2)
    added = e1 + e2
    val = simulate(added, variables=variables, backend=simulator)
    en1 = simulate(e1, variables=variables, backend=simulator)
    en2 = simulate(e2, variables=variables, backend=simulator)
    an1 = np.sin(angle1(variables=variables))
    an2 = -np.sin(angle2(variables=variables))
    assert np.isclose(val, en1 + en2, atol=1.e-4)
    assert np.isclose(val, float(an1 + an2), atol=1.e-4)


@pytest.mark.parametrize("simulator", simulators)
def test_mixed_multiplication(simulator, value1=(numpy.random.randint(10, 1000) / 1000.0 * (numpy.pi / 2.0)),
                              value2=(numpy.random.randint(10, 1000) / 1000.0 * (numpy.pi / 2.0))):
    angle1 = Variable(name="angle1")
    angle2 = Variable(name="angle2")
    variables = {angle1: value1, angle2: value2}
    qubit = 0
    control = 1
    H1 = paulis.X(qubit=qubit)
    U1 = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle1)
    e1 = ExpectationValue(U=U1, H=H1)
    H2 = paulis.Y(qubit=qubit)
    U2 = gates.X(target=control) + gates.Rx(target=qubit, control=control, angle=angle2)
    e2 = ExpectationValue(U=U2, H=H2)
    added = e1 * e2
    val = simulate(added, variables=variables, backend=simulator)
    en1 = simulate(e1, variables=variables, backend=simulator)
    en2 = simulate(e2, variables=variables, backend=simulator)
    an1 = np.sin(angle1(variables=variables))
    an2 = -np.sin(angle2(variables=variables))
    assert np.isclose(val, en1 * en2, atol=1.e-4)
    assert np.isclose(val, an1 * an2, atol=1.e-4)

@pytest.mark.parametrize("simulator", simulators)
def test_mixed_division(simulator, value1=(numpy.random.randint(10, 1000) / 1000.0 * (numpy.pi / 2.0)),
                        value2=(numpy.random.randint(10, 1000) / 1000.0 * (numpy.pi / 2.0))):
    angle1 = Variable(name="angle1")
    angle2 = Variable(name="angle2")
    variables = {angle1: value1, angle2: value2}
    qubit = 0
    control = 1
    H1 = paulis.X(qubit=qubit)
    U1 = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle1)
    e1 = ExpectationValue(U=U1, H=H1)
    H2 = paulis.Y(qubit=qubit)
    U2 = gates.X(target=control) + gates.Rx(target=qubit, control=control, angle=angle2)
    e2 = ExpectationValue(U=U2, H=H2)
    added = e1 / (e2+1.e-4) # avoid zero division
    val = simulate(added, variables=variables, backend=simulator)
    en1 = simulate(e1, variables=variables, backend=simulator)
    en2 = simulate(e2, variables=variables, backend=simulator)
    an1 = np.sin(angle1(variables=variables))
    an2 = -np.sin(angle2(variables=variables))
    assert np.isclose(val, en1 / (en2+1.e-4), atol=1.e-4)
    assert np.isclose(val, an1 / (an2+1.e-4), atol=1.e-4)


@pytest.mark.parametrize("simulator", simulators)
def test_mixed_power(simulator, value1=(numpy.random.randint(10, 1000) / 1000.0 * (numpy.pi / 2.0)),
                     value2=(numpy.random.randint(10, 1000) / 1000.0 * (numpy.pi / 2.0))):
    angle1 = Variable(name="angle1")
    angle2 = Variable(name="angle2")
    variables = {angle1: value1, angle2: value2}
    qubit = 0
    control = 1
    H1 = paulis.X(qubit=qubit)
    U1 = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle1)
    e1 = ExpectationValue(U=U1, H=H1)
    H2 = paulis.Y(qubit=qubit)
    U2 = gates.X(target=control) + gates.Rx(target=qubit, control=control, angle=angle2)
    e2 = ExpectationValue(U=U2, H=H2)
    added = (e1+1.e-4) ** e2
    val = simulate(added, variables=variables, backend=simulator)
    en1 = simulate(e1, variables=variables, backend=simulator)
    en2 = simulate(e2, variables=variables, backend=simulator)
    an1 = np.sin(angle1(variables=variables))
    an2 = -np.sin(angle2(variables=variables))
    assert np.isclose(val, (en1+1.e-4) ** en2, atol=1.e-4)
    assert np.isclose(val, (an1+1.e-4) ** an2, atol=1.e-4)


@pytest.mark.parametrize("simulator", simulators)
@pytest.mark.parametrize('op', [np.add, np.subtract, np.float_power, np.true_divide, np.multiply])
def test_heterogeneous_operations_l(simulator, op, value1=(numpy.random.randint(1, 1000) / 1000.0 * (numpy.pi / 2.0)),
                                    value2=(numpy.random.randint(1, 1000) / 1000.0 * (numpy.pi / 2.0))):
    angle1 = Variable(name="angle1")
    angle2 = Variable(name="angle2")
    variables = {angle1: value1, angle2: value2}
    qubit = 0
    control = 1
    H2 = paulis.X(qubit=qubit)
    U2 = gates.X(target=control) + gates.Ry(target=qubit, control=control, angle=angle2)
    e2 = ExpectationValue(U=U2, H=H2)
    added = Objective(args=[angle1, e2.args[0]], transformation=op)
    val = simulate(added, variables=variables, backend=simulator)
    en2 = simulate(e2, variables=variables, backend=simulator)
    an1 = angle1(variables=variables)
    an2 = np.sin(angle2(variables=variables))
    assert np.isclose(val, float(op(an1, en2)), atol=1.e-4)
    assert np.isclose(en2, an2, atol=1.e-4)


@pytest.mark.parametrize("simulator", simulators)
@pytest.mark.parametrize('op', [np.add, np.subtract, np.true_divide, np.multiply])
def test_heterogeneous_operations_r(simulator, op, value1=(numpy.random.randint(1, 999) / 1000.0 * (numpy.pi / 2.0)),
                                    value2=(numpy.random.randint(1, 999) / 1000.0 * (numpy.pi / 2.0))):
    angle1 = Variable(name="angle1")
    angle2 = Variable(name="angle2")
    variables = {angle1: value1, angle2: value2}
    qubit = 0
    control = 1
    H1 = paulis.Y(qubit=qubit)
    U1 = gates.X(target=control) + gates.Rx(target=qubit, control=control, angle=angle1)
    e1 = ExpectationValue(U=U1, H=H1)
    added = Objective(args=[e1.args[0], angle2], transformation=op)
    val = simulate(added, variables=variables, backend=simulator)
    en1 = simulate(e1, variables=variables, backend=simulator)
    an1 = -np.sin(angle1(variables=variables))
    an2 = angle2(variables=variables)
    assert np.isclose(val, float(op(en1, an2)), atol=1.e-4)
    assert np.isclose(en1, an1, atol=1.e-4)


@pytest.mark.parametrize("simulator", simulators)
def test_heterogeneous_gradient_r_add(simulator):
    ### the reason we don't test float power here is that it keeps coming up NAN, because the argument is too small
    angle1 = Variable(name="angle1")
    value = numpy.random.randint(100, 1000) / 1000.0 * (numpy.pi / 2.0)
    variables = {angle1: value}
    qubit = 0
    control = 1
    H1 = paulis.Y(qubit=qubit)
    U1 = gates.X(target=control) + gates.Rx(target=qubit, control=control, angle=angle1)
    e1 = ExpectationValue(U=U1, H=H1)
    added = Objective([e1.args[0], angle1], np.add)
    val = simulate(added, variables=variables, backend=simulator)
    en1 = simulate(e1, variables=variables, backend=simulator)
    an1 = -np.sin(angle1(variables=variables))
    anval = angle1(variables=variables)
    dO = grad(added, 'angle1')
    dE = grad(e1, 'angle1')
    deval = simulate(dE, variables=variables, backend=simulator)
    doval = simulate(dO, variables=variables, backend=simulator)
    dtrue = 1.0 + deval
    assert np.isclose(float(val), float(np.add(en1, anval)))
    assert np.isclose(en1, an1, atol=1.e-4)
    assert np.isclose(doval, dtrue, atol=1.e-4)


@pytest.mark.parametrize("simulator", simulators)
def test_heterogeneous_gradient_r_mul(simulator):
    ### the reason we don't test float power here is that it keeps coming up NAN, because the argument is too small
    angle1 = Variable(name="angle1")
    value = (numpy.random.randint(100, 1000) / 1000.0 * (numpy.pi / 2.0))
    variables = {angle1: value}
    qubit = 0
    control = 1
    H1 = paulis.Y(qubit=qubit)
    U1 = gates.X(target=control) + gates.Rx(target=qubit, control=control, angle=angle1)
    e1 = ExpectationValue(U=U1, H=H1)
    added = Objective([e1.args[0], angle1], transformation=np.multiply)
    val = simulate(added, variables=variables, backend=simulator)
    en1 = simulate(e1, variables=variables, backend=simulator)
    an1 = -np.sin(angle1(variables=variables))
    anval = angle1(variables=variables)
    dO = grad(added, 'angle1')
    dE = grad(e1, 'angle1')
    deval = simulate(dE, variables=variables, backend=simulator)
    doval = simulate(dO, variables=variables, backend=simulator)
    dtrue = deval * anval + en1
    assert np.isclose(float(val), float(np.multiply(en1, anval)))
    assert np.isclose(en1, an1, atol=1.e-4)
    assert np.isclose(doval, dtrue, atol=1.e-4)


@pytest.mark.parametrize("simulator", simulators)
def test_heterogeneous_gradient_r_div(simulator):
    ### the reason we don't test float power here is that it keeps coming up NAN, because the argument is too small
    angle1 = Variable(name="angle1")
    value = (numpy.random.randint(100, 1000) / 1000.0 * (numpy.pi / 2.0))
    variables = {angle1: value}
    qubit = 0
    control = 1
    H1 = paulis.Y(qubit=qubit)
    U1 = gates.X(target=control) + gates.Rx(target=qubit, control=control, angle=angle1)
    e1 = ExpectationValue(U=U1, H=H1)
    added = Objective(args=[e1.args[0], angle1], transformation=np.true_divide)
    val = simulate(added, variables=variables, backend=simulator)
    en1 = simulate(e1, variables=variables, backend=simulator)
    an1 = -np.sin(angle1(variables=variables))
    anval = angle1(variables=variables)
    dO = grad(added, 'angle1')
    dE = grad(e1, 'angle1')
    deval = simulate(dE, variables=variables, backend=simulator)
    doval = simulate(dO, variables=variables, backend=simulator)
    dtrue = deval / anval - en1 / (anval ** 2)
    assert np.isclose(float(val), float(np.true_divide(en1, anval)))
    assert np.isclose(en1, an1, atol=1.e-4)
    assert np.isclose(doval, dtrue, atol=1.e-4)


@pytest.mark.parametrize("simulator", simulators)
def test_inside(simulator, value1=(numpy.random.randint(10, 1000) / 1000.0 * (numpy.pi / 2.0)),
                value2=(numpy.random.randint(10, 1000) / 1000.0 * (numpy.pi / 2.0))):
    angle1 = Variable(name="angle1")
    angle2 = Variable(name="angle2")
    variables = {angle1: value1, angle2: value2}
    prod = angle1 * angle2
    qubit = 0
    control = None
    H = paulis.Y(qubit=qubit)
    U = gates.Rx(target=qubit, control=control, angle=prod)
    Up = gates.Rx(target=qubit, control=control, angle=prod + np.pi / 2)
    Down = gates.Rx(target=qubit, control=control, angle=prod - np.pi / 2)
    e1 = ExpectationValue(U=U, H=H)
    en1 = simulate(e1, variables=variables, backend=simulator)
    uen = simulate(0.5 * ExpectationValue(Up, H), variables=variables, backend=simulator)
    den = simulate(-0.5 * ExpectationValue(Down, H), variables=variables, backend=simulator)
    an1 = -np.sin(prod(variables=variables))
    anval = prod(variables=variables)
    an2 = angle2(variables=variables)
    dP = grad(prod, 'angle1')
    dE = grad(e1, 'angle1')
    deval = simulate(dE, variables=variables, backend=simulator)
    dpval = simulate(dP, variables=variables, backend=simulator)
    dtrue = an2 * (uen + den)
    assert np.isclose(en1, an1, atol=1.e-4)
    assert np.isclose(deval, dtrue, atol=1.e-4)


@pytest.mark.parametrize("simulator", simulators)
def test_akward_expression(simulator, value1=(numpy.random.randint(10, 1000) / 1000.0 * (numpy.pi / 2.0)),
                           value2=(numpy.random.randint(10, 1000) / 1000.0 * (numpy.pi / 2.0))):
    angle1 = Variable(name="angle1")
    angle2 = Variable(name="angle2")
    variables = {angle1: value1, angle2: value2}

    prod = angle1 * angle2
    qubit = 0
    control = None
    H = paulis.Y(qubit=qubit)
    U = gates.Rx(target=qubit, control=control, angle=prod)
    Up = gates.Rx(target=qubit, control=control, angle=prod + np.pi / 2)
    Down = gates.Rx(target=qubit, control=control, angle=prod - np.pi / 2)
    e1 = ExpectationValue(U=U, H=H)
    en1 = simulate(e1, variables=variables, backend=simulator)
    uen = simulate(0.5 * ExpectationValue(Up, H), variables=variables, backend=simulator)
    den = simulate(-0.5 * ExpectationValue(Down, H), variables=variables, backend=simulator)
    an1 = -np.sin(prod(variables=variables))
    anval = prod(variables=variables)
    an2 = angle2(variables=variables)
    added = angle1 * e1
    dO = grad(added, 'angle1')
    dE = grad(e1, 'angle1')
    deval = simulate(dE, variables=variables, backend=simulator)
    doval = simulate(dO, variables=variables, backend=simulator)
    dtrue = angle1(variables=variables) * deval + en1
    assert np.isclose(en1, an1)
    assert np.isclose(deval, an2 * (uen + den), atol=1.e-4)
    assert np.isclose(doval, dtrue, atol=1.e-4)


@pytest.mark.parametrize("simulator", simulators)
def test_really_awfull_thing(simulator, value1=(numpy.random.randint(10, 1000) / 1000.0 * (numpy.pi / 2.0)),
                             value2=(numpy.random.randint(10, 1000) / 1000.0 * (numpy.pi / 2.0))):
    angle1 = Variable(name="angle1")
    angle2 = Variable(name="angle2")
    variables = {angle1: value1, angle2: value2}

    prod = angle1 * angle2
    qubit = 0
    control = None
    H = paulis.Y(qubit=qubit)
    U = gates.Rx(target=qubit, control=control, angle=prod)
    Up = gates.Rx(target=qubit, control=control, angle=prod + np.pi / 2)
    Down = gates.Rx(target=qubit, control=control, angle=prod - np.pi / 2)
    e1 = ExpectationValue(U=U, H=H)
    en1 = simulate(e1, variables=variables, backend=simulator)
    uen = simulate(0.5 * ExpectationValue(Up, H), variables=variables, backend=simulator)
    den = simulate(-0.5 * ExpectationValue(Down, H), variables=variables, backend=simulator)
    an1 = -np.sin(prod(variables=variables))
    anval = prod(variables=variables)
    an2 = angle2(variables=variables)
    added = angle1 * e1
    raised = added.wrap(np.sin)
    dO = grad(raised, 'angle1')
    dE = grad(e1, 'angle1')
    dA = grad(added, 'angle1')
    val = simulate(added, variables=variables, backend=simulator)
    dave = simulate(dA, variables=variables, backend=simulator)
    deval = simulate(dE, variables=variables, backend=simulator)
    doval = simulate(dO, variables=variables, backend=simulator)
    dtrue = np.cos(val) * dave
    assert np.isclose(en1, an1, atol=1.e-4)
    assert np.isclose(deval, an2 * (uen + den), atol=1.e-4)
    assert np.isclose(doval, dtrue, atol=1.e-4)

# testing backward compatibility (VectorObjective is now QTensor)
def test_stacking():
    a=Variable('a')
    b=Variable('b')
    def f(x):
        return np.cos(x)**2. + np.sin(x)**2.
    vals = {Variable('a'):numpy.random.uniform(0,np.pi),Variable('b'):numpy.random.uniform(0,np.pi)}
    O = VectorObjective(argsets=[[a],[b],[a],[b]])
    O1 = O.apply(f)
    O2 = O1/4
    output = simulate(O2,variables=vals)
    assert np.isclose(1.,np.sum(output))

@pytest.mark.parametrize("simulator", [tequila.simulators.simulator_api.pick_backend("random"), tequila.simulators.simulator_api.pick_backend()])
def test_stacking_quantum(simulator, value1=(numpy.random.randint(10, 1000) / 1000.0 * (numpy.pi / 2.0)),
                value2=(numpy.random.randint(10, 1000) / 1000.0 * (numpy.pi / 2.0))):
    a = Variable('a')
    b = Variable('b')
    values = {a: value1, b: value2}
    H1 = tq.paulis.X(0)
    H2 = tq.paulis.Y(0)
    U1= tq.gates.Ry(angle=a,target=0)
    U2= tq.gates.Rx(angle=b,target=0)
    e1=ExpectationValue(U1,H1)
    e2=ExpectationValue(U2,H2)
    stacked= tq.objective.vectorize([e1, e2])
    out=simulate(stacked,variables=values,backend=simulator)
    v1=out[0]
    v2=out[1]
    an1= np.sin(a(values))
    an2= -np.sin(b(values))
    assert np.isclose(v1+v2,an1+an2,atol=1e-3)
    # not gonna contract, lets make gradient do some real work
    ga=grad(stacked,a)
    gb=grad(stacked,b)
    la=[tq.simulate(x,variables=values) for x in ga]
    print(la)
    lb=[tq.simulate(x,variables=values) for x in gb]
    print(lb)
    tota=np.sum(np.array(la))
    totb=np.sum(np.array(lb))
    gan1= np.cos(a(values))
    gan2= -np.cos(b(values))
    assert np.isclose(tota+totb,gan1+gan2,atol=1e-3)


@pytest.mark.parametrize("simulator", [tequila.simulators.simulator_api.pick_backend("random"), tequila.simulators.simulator_api.pick_backend()])
def test_total_type_jumble(simulator,value1=(numpy.random.randint(10, 1000) / 1000.0 * (numpy.pi / 2.0)),
                value2=(numpy.random.randint(10, 1000) / 1000.0 * (numpy.pi / 2.0))):
    a = Variable('a')
    b = Variable('b')
    values = {a: value1, b: value2}
    H1 = tq.paulis.X(0)
    H2 = tq.paulis.Y(0)
    U1= tq.gates.Ry(angle=a,target=0)
    U2= tq.gates.Rx(angle=b,target=0)
    e1=ExpectationValue(U1,H1)
    e2=ExpectationValue(U2,H2)
    stacked= tq.objective.vectorize([e1, e2])
    stacked = stacked*a*e2
    out=simulate(stacked,variables=values,backend=simulator)
    v1=out[0]
    v2=out[1]
    appendage =  a(values) * -np.sin(b(values))
    an1= np.sin(a(values)) *  appendage
    an2= -np.sin(b(values)) * appendage
    assert np.isclose(v1+v2,an1+an2)
    # not gonna contract, lets make gradient do some real work
    ga=grad(stacked,a)
    gb=grad(stacked,b)
    la=[tq.simulate(x,variables=values) for x in ga]
    print(la)
    lb=[tq.simulate(x,variables=values) for x in gb]
    print(lb)
    tota=np.sum(np.array(la))
    totb=np.sum(np.array(lb))
    gan1= np.cos(a(values)) * appendage + (np.sin(a(values)) * -np.sin(b(values))) - (np.sin(b(values)) * -np.sin(b(values)))
    gan2= np.sin(a(values)) * a(values) * -np.cos(b(values)) + 2 * (-np.cos(b(values)) * appendage)
    assert np.isclose(tota+totb,gan1+gan2)
