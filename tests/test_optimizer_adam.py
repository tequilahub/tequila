import pytest, numpy
import tequila as tq
import multiprocessing as mp
from tequila.simulators.simulator_api import simulate
from tequila.optimizers.optimizer_adam import minimize

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
    result = minimize(objective=O, maxiter=mi, backend=simulator,samples=1024)
    print(result.history.energies)
    assert (len(result.history.energies) <= mi*mp.cpu_count())

@pytest.mark.parametrize("simulator", [tq.simulators.simulator_api.pick_backend("random")])
def test_one_qubit_wfn(simulator,lr):
    U = tq.gates.Trotterized(angles=["a"], steps=1, generators=[tq.paulis.Y(0)])
    H = tq.paulis.X(0)
    O = tq.ExpectationValue(U=U, H=H)
    result = minimize(objective=O,lr=lr, maxiter=100, backend=simulator)
    assert (numpy.isclose(result.energy, -1.0,atol=1.e-2))


@pytest.mark.parametrize("simulator", [tq.simulators.simulator_api.pick_backend("random")])
@pytest.mark.parametrize('lr',numpy.random.uniform(.01,.1,1))
def test_one_qubit_sampling(simulator,lr):
    U = tq.gates.Trotterized(angles=["a"], steps=1, generators=[tq.paulis.Y(0)])
    H = tq.paulis.X(0)
    O = tq.ExpectationValue(U=U, H=H)
    result = minimize(objective=O,lr=lr,samples=1000, maxiter=100, backend=simulator)
    assert (numpy.isclose(result.energy, -1.0,atol=1.e-2))

