import pytest, numpy
import tequila as tq
import multiprocessing as mp
from tequila.simulators.simulator_api import simulate
try:
    from tequila.optimizers.optimizer_gpyopt import minimize as minimize
    has_gpyopt=True
except:
    has_gpyopt=False

@pytest.mark.dependencies
def test_dependencies():
    assert(tq.has_gpyopt)


@pytest.mark.skipif(condition=not has_gpyopt, reason="you don't have GPyOpt")
@pytest.mark.parametrize("simulator", [tq.simulators.simulator_api.pick_backend("random")])
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

@pytest.mark.skipif(condition=not has_gpyopt, reason="you don't have GPyOpt")
@pytest.mark.parametrize("simulator", [tq.simulators.simulator_api.pick_backend("random", samples=1)])
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
    print(result.history.energies)
    assert (len(result.history.energies) <= mi*mp.cpu_count())

@pytest.mark.skipif(condition=not has_gpyopt, reason="you don't have GPyOpt")
@pytest.mark.parametrize("simulator", [tq.simulators.simulator_api.pick_backend("random")])
@pytest.mark.parametrize('method',['lbfgs','DIRECT','CMA'])
def test_one_qubit_wfn(simulator,method):
    U = tq.gates.Trotterized(angles=["a"], steps=1, generators=[tq.paulis.Y(0)])
    H = tq.paulis.X(0)
    O = tq.ExpectationValue(U=U, H=H)
    result = minimize(objective=O, maxiter=8, backend=simulator,acquisition=method)
    assert (numpy.isclose(result.energy, -1.0,atol=1.e-2))


@pytest.mark.skipif(condition=not has_gpyopt, reason="you don't have GPyOpt")
@pytest.mark.parametrize("simulator", [tq.simulators.simulator_api.pick_backend("random")])
def test_one_qubit_wfn_really_works(simulator):
    U = tq.gates.Trotterized(angles=["a"], steps=1, generators=[tq.paulis.Y(0)])
    H = tq.paulis.X(0)
    O = tq.ExpectationValue(U=U, H=H)
    result = minimize(objective=O, maxiter=8, backend=simulator)
    assert (numpy.isclose(result.energy, -1.0,atol=1.e-2))
    assert (numpy.isclose(result.energy,simulate(objective=O,variables=result.angles)))


@pytest.mark.skipif(condition=not has_gpyopt, reason="you don't have GPyOpt")
@pytest.mark.parametrize("simulator", [tq.simulators.simulator_api.pick_backend("random",samples=1)])
@pytest.mark.parametrize('method',['lbfgs'])
def test_if_sampling_broken(simulator,method):
    U = tq.gates.Trotterized(angles=["a"], steps=1, generators=[tq.paulis.Y(0)])
    H = tq.paulis.X(0)
    O = tq.ExpectationValue(U=U, H=H)
    result_1 = minimize(objective=O, maxiter=8, backend=simulator,acquisition=method)
    check_e=result_1.history.energies
    comparison=[]
    for params in result_1.history.angles:
        comparison.append(simulate(O,backend=simulator,samples=100000,variables=params))
    for i in range(len(comparison)):
        assert numpy.isclose(comparison[i],check_e[i],atol=0.1)



@pytest.mark.skip("skipped for now")
@pytest.mark.skipif(condition=not has_gpyopt, reason="you don't have GPyOpt")
@pytest.mark.parametrize("simulator", [tq.simulators.simulator_api.pick_backend("random", samples=1)])
def test_one_qubit_shot(simulator):
    U = tq.gates.Trotterized(angles=["a"], steps=1, generators=[tq.paulis.Y(0)])
    H = tq.paulis.X(0)
    O = tq.ExpectationValue(U=U, H=H)
    result = minimize(objective=O, maxiter=20, backend=simulator, samples=10000)
    assert (numpy.isclose(result.energy, -1.0, atol=1.e-2))