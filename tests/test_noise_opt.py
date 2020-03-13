from tequila.circuit import gates
from tequila.objective import ExpectationValue
from tequila.hamiltonian import paulis
from tequila.circuit.noise import BitFlip,PhaseDamp,PhaseFlip,AmplitudeDamp,PhaseAmplitudeDamp,DepolarizingError
import numpy
import pytest
from tequila.simulators.simulator_api import INSTALLED_SAMPLERS
import tequila.simulators.simulator_api
import tequila as tq


@pytest.mark.parametrize("simulator", ['qiskit','pyquil','cirq'])
@pytest.mark.parametrize("p", numpy.random.uniform(0.1,.4,1))
@pytest.mark.parametrize('method',tq.optimizer_scipy.OptimizerSciPy.gradient_free_methods)
def test_bit_flip_scipy_gradient_free(simulator, p,method):

    qubit = 0
    H = paulis.Qm(qubit)
    U = gates.Rx(target=qubit,angle=tq.Variable('a'))
    O = ExpectationValue(U=U, H=H)
    NM=BitFlip(p,['rx'])
    result = tq.optimizer_scipy.minimize(objective=O,samples=10000,backend=simulator, method=method,noise=NM, tol=1.e-4,silent=False)
    assert(numpy.isclose(result.energy, p, atol=1.e-2))

@pytest.mark.parametrize("simulator", ['qiskit','cirq'])
@pytest.mark.parametrize("p", numpy.random.uniform(0.1,.4,1))
@pytest.mark.parametrize('method',tq.optimizer_scipy.OptimizerSciPy.gradient_based_methods)
def test_bit_flip_scipy_gradient(simulator, p,method):

    qubit = 0
    H = paulis.Qm(qubit)
    U = gates.Rx(target=qubit,angle=tq.Variable('a'))
    O = ExpectationValue(U=U, H=H)
    NM=BitFlip(p,['rx'])
    result = tq.optimizer_scipy.minimize(objective=O,samples=10000,backend=simulator, method=method,noise=NM, tol=1.e-4,silent=False)
    assert(numpy.isclose(result.energy, p, atol=1.e-2))

@pytest.mark.parametrize("simulator", ['qiskit','pyquil','cirq'])
@pytest.mark.parametrize("p", numpy.random.uniform(0.1,.4,1))
@pytest.mark.parametrize('method',tq.optimizer_scipy.OptimizerSciPy.hessian_based_methods)
def test_bit_flip_scipy_hessian(simulator, p,method):

    qubit = 0
    H = paulis.Qm(qubit)
    U = gates.Rx(target=qubit,angle=tq.Variable('a'))
    O = ExpectationValue(U=U, H=H)
    NM=BitFlip(p,['rx'])
    result = tq.optimizer_scipy.minimize(objective=O,samples=10000,backend=simulator, method=method,noise=NM, tol=1.e-4,silent=False)
    assert(numpy.isclose(result.energy, p, atol=1.e-2))

@pytest.mark.parametrize("simulator", ['qiskit'])
@pytest.mark.parametrize("p", numpy.random.uniform(0.1,.4,1))
def test_bit_flip_phoenics(simulator, p):

    qubit = 0
    H = paulis.Qm(qubit)
    U = gates.Rx(target=qubit,angle=tq.Variable('a'))
    O = ExpectationValue(U=U, H=H)
    NM=BitFlip(p,['rx'])
    result = tq.optimizer_phoenics.minimize(objective=O,maxiter=3,samples=1000,backend=simulator,noise=NM)
    assert(numpy.isclose(result.energy, p, atol=1.e-2))


@pytest.mark.parametrize("simulator", ['cirq'])
@pytest.mark.parametrize("p", numpy.random.uniform(0.1,.4,1))
@pytest.mark.parametrize('method',['lbfgs','DIRECT','CMA'])
def test_bit_flip_gpyopt(simulator, p,method):

    qubit = 0
    H = paulis.Qm(qubit)
    U = gates.Rx(target=qubit,angle=tq.Variable('a'))
    O = ExpectationValue(U=U, H=H)
    NM=BitFlip(p,['rx'])
    result = tq.optimizer_gpyopt.minimize(objective=O,maxiter=10,samples=10000,backend=simulator, acquisition=method,noise=NM)
    assert(numpy.isclose(result.energy, p, atol=1.e-2))