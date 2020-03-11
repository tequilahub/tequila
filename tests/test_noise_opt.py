from tequila.circuit import gates
from tequila.objective import ExpectationValue
from tequila.objective.objective import Variable
from tequila.hamiltonian import paulis
from tequila import simulate
import tequila
from tequila.circuit.noise import BitFlip,PhaseDamp,PhaseFlip,AmplitudeDamp,PhaseAmplitudeDamp,DepolarizingError
import numpy
import pytest
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

@pytest.mark.parametrize("simulator", ['qiskit','pyquil','cirq'])
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