import pytest, numpy
import tequila as tq
import multiprocessing as mp
try:
    from tequila.optimizers.optimizer_phoenics import minimize as minimize
    has_phoenics=True
except:
    has_phoenics=False
# skip usage of symbolic simulator
simulators = []
for k in tq.simulators.INSTALLED_SIMULATORS.keys():
    if k != "symbolic":
        simulators.append(k)
samplers = []
for k in tq.simulators.INSTALLED_SAMPLERS.keys():
    if k != "symbolic":
        samplers.append(k)


@pytest.mark.skipif(condition=not has_phoenics, reason="you don't have phoenics")
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

    H = 1.0 * tq.paulis.X(0) + 2.0 * tq.paulis.Y(1) + 3.0 * tq.paulis.Z(2)
    O = tq.ExpectationValue(U=U, H=H)
    result = minimize(objective=O, maxiter=1, backend=simulator)

@pytest.mark.skipif(condition=not has_phoenics, reason="you don't have phoenics")
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
    mi=2
    result = minimize(objective=O, maxiter=mi, backend=simulator)
    assert (len(result.history.energies) <= mi*mp.cpu_count())

@pytest.mark.skipif(condition=not has_phoenics, reason="you don't have phoenics")
@pytest.mark.parametrize("simulator", ['qulacs'])
def test_one_qubit_wfn(simulator):
    U = tq.gates.Trotterized(angles=["a"], steps=1, generators=[tq.paulis.Y(0)])
    H = tq.paulis.X(0)
    O = tq.ExpectationValue(U=U, H=H)
    result = tq.optimizers.optimizer_phoenics.minimize(objective=O, maxiter=8, backend=simulator)
    assert (numpy.isclose(result.energy, -1.0,atol=1.e-2))

@pytest.mark.skipif(condition=not has_phoenics, reason="you don't have phoenics")
@pytest.mark.parametrize("simulator", samplers)
def test_one_qubit_shot(simulator):
    U = tq.gates.Trotterized(angles=["a"], steps=1, generators=[tq.paulis.Y(0)])
    H = tq.paulis.X(0)
    O = tq.ExpectationValue(U=U, H=H)
    result = minimize(objective=O, maxiter=3, backend=simulator, samples=10000)
    assert (numpy.isclose(result.energy, -1.0, atol=1.e-2))



