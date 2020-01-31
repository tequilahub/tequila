import pytest, numpy
import tequila as tq

# skip usage of symbolic simulator
simulators = []
for k in tq.simulators.INSTALLED_SIMULATORS.keys():
    if k != "symbolic":
        simulators.append(k)
samplers = []
for k in tq.simulators.INSTALLED_SAMPLERS.keys():
    if k != "symbolic":
        samplers.append(k)


@pytest.mark.parametrize("simulator", simulators)
def test_execution(simulator):
    U = tq.gates.Rz(angle="a", target=0) \
        + tq.gates.X(target=2) \
        + tq.gates.Ry(angle="b", target=1, control=2) \
        + tq.gates.Trotterized(angles=["c", "d"],
                               generators=[-0.25 * tq.paulis.Z(1), tq.paulis.X(0) + tq.paulis.Y(1)], steps=2) \
        + tq.gates.Trotterized(angles=[1.0, 2.0],
                               generators=[-0.25 * tq.paulis.Z(1), tq.paulis.X(0) + tq.paulis.Y(1)], steps=2) \
        + tq.gates.ExpPauli(angle="a", paulistring="X(0)Y(1)Z(2)")

    print(U.extract_variables())
    H = 1.0 * tq.paulis.X(0) + 2.0 * tq.paulis.Y(1) + 3.0 * tq.paulis.Z(2)
    O = tq.ExpectationValue(U=U, H=H)

    result = tq.optimizer_scipy.minimize(objective=O, maxiter=2, method="TNC", backend=simulator)


@pytest.mark.parametrize("simulator", samplers)
def test_execution_shot(simulator):
    U = tq.gates.Rz(angle="a", target=0) \
        + tq.gates.X(target=2) \
        + tq.gates.Ry(angle="b", target=1, control=2) \
        + tq.gates.Trotterized(angles=["c","d"],
                               generators=[-0.25 * tq.paulis.Z(1), tq.paulis.X(0) + tq.paulis.Y(1)], steps=2) \
        + tq.gates.Trotterized(angles=[1.0, 2.0],
                               generators=[-0.25 * tq.paulis.Z(1), tq.paulis.X(0) + tq.paulis.Y(1)], steps=2) \
        + tq.gates.ExpPauli(angle="a", paulistring="X(0)Y(1)Z(2)")
    H = 1.0 * tq.paulis.X(0) + 2.0 * tq.paulis.Y(1) + 3.0 * tq.paulis.Z(2)
    O = tq.ExpectationValue(U=U, H=H)

    result = tq.optimizer_scipy.minimize(objective=O, maxiter=2, method="TNC", backend=simulator, samples=3)
    assert (len(result.history.energies) <= 3)


@pytest.mark.parametrize("simulator", simulators)
def test_one_qubit_wfn(simulator):
    U = tq.gates.Trotterized(angles=["a"], steps=1, generators=[tq.paulis.Y(0)])
    H = tq.paulis.X(0)
    O = tq.ExpectationValue(U=U, H=H)
    result = tq.optimizer_scipy.minimize(objective=O, maxiter=15, backend=simulator)
    assert (numpy.isclose(result.energy, -1.0))

@pytest.mark.skip("shot based optimization is too bad to be tested")
@pytest.mark.parametrize("simulator", samplers)
def test_one_qubit_shot(simulator):
    U = tq.gates.Trotterized(angles=["a"], steps=1, generators=[tq.paulis.Y(0)])
    H = tq.paulis.X(0)
    O = tq.ExpectationValue(U=U, H=H)
    result = tq.optimizer_scipy.minimize(objective=O, maxiter=15, backend=simulator, samples=10000)
    assert (numpy.isclose(result.energy, -1.0, atol=1.e-2))
