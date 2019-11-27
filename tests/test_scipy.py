import pytest, numpy
import tequila as tq

@pytest.mark.parametrize("simulator", tq.simulators.get_all_wfn_simulators())
def test_execution(simulator):
    U = tq.gates.Rz(angle="a", target=0) \
        + tq.gates.X(target=2) \
        + tq.gates.Ry(angle=tq.Variable(name="b", value=2.0), target=1, control=2) \
        + tq.gates.Trotterized(angles=[tq.Variable(name="c", value=1.23), tq.Variable(name="d", value=1.23)],
                               generators=[-0.25 * tq.paulis.Z(1), tq.paulis.X(0) + tq.paulis.Y(1)], steps=2) \
        + tq.gates.ExpPauli(angle="a", paulistring="X(0)Y(1)Z(2)")
    H = 1.0 * tq.paulis.X(0) + 2.0 * tq.paulis.Y(1) + 3.0 * tq.paulis.Z(2)
    O = tq.Objective(unitaries=U, observable=H)

    result = tq.optimizer_scipy.minimize(objective=O, maxiter=2, method="TNC", simulator=simulator)
    assert (len(result.history.energies) == 2+1)


@pytest.mark.parametrize("simulator", tq.simulators.get_all_samplers())
def test_execution_shot(simulator):
    U = tq.gates.Rz(angle="a", target=0) \
        + tq.gates.X(target=2) \
        + tq.gates.Ry(angle=tq.Variable(name="b", value=2.0), target=1, control=2) \
        + tq.gates.Trotterized(angles=[tq.Variable(name="c", value=1.23), tq.Variable(name="d", value=1.23)],
                               generators=[-0.25 * tq.paulis.Z(1), tq.paulis.X(0) + tq.paulis.Y(1)], steps=2) \
        + tq.gates.ExpPauli(angle="a", paulistring="X(0)Y(1)Z(2)")
    H = 1.0 * tq.paulis.X(0) + 2.0 * tq.paulis.Y(1) + 3.0 * tq.paulis.Z(2)
    O = tq.Objective(unitaries=U, observable=H)

    result = tq.optimizer_scipy.minimize(objective=O, maxiter=2, method="TNC", simulator=simulator, samples=3)
    assert (len(result.history.energies) == 2+1)

@pytest.mark.parametrize("simulator", tq.simulators.get_all_wfn_simulators())
def test_one_qubit_wfn(simulator):
    U = tq.gates.Trotterized(angles=[tq.Variable(name="a", value=0.0)], steps=1, generators=[tq.paulis.Y(0)])
    H = tq.paulis.X(0)
    O = tq.Objective(unitaries=U, observable=H)
    result = tq.optimizer_scipy.minimize(objective=O, maxiter=15, simulator=simulator)
    assert (numpy.isclose(result.energy, -1.0))


@pytest.mark.parametrize("simulator", tq.simulators.get_all_samplers())
def test_one_qubit_shot(simulator):
    U = tq.gates.Trotterized(angles=[tq.Variable(name="a", value=0.0)], steps=1, generators=[tq.paulis.Y(0)])
    H = tq.paulis.X(0)
    O = tq.Objective(unitaries=U, observable=H)
    result = tq.optimizer_scipy.minimize(objective=O, maxiter=15, simulator=simulator, samples=10000)
    assert (numpy.isclose(result.energy, -1.0))
