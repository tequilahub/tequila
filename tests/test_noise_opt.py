from tequila.circuit import gates
from tequila.objective import ExpectationValue
from tequila.hamiltonian import paulis
from tequila.circuit.noise import BitFlip
import numpy
import pytest
import tequila as tq

from tequila.simulators.simulator_api import SUPPORTED_NOISE_BACKENDS, INSTALLED_SAMPLERS
samplers = [k for k in INSTALLED_SAMPLERS.keys() if k in SUPPORTED_NOISE_BACKENDS]


@pytest.mark.dependencies
def test_dependencies():
    for k in SUPPORTED_NOISE_BACKENDS:
        assert k in samplers


@pytest.mark.skipif(len(samplers) == 0, reason="Missing necessary backends")
@pytest.mark.parametrize("simulator", [numpy.random.choice(samplers)])
@pytest.mark.parametrize("p", numpy.random.uniform(0.1, .4, 1))
@pytest.mark.parametrize('method', numpy.random.choice(['NELDER-MEAD', 'COBYLA'],1))
def test_bit_flip_scipy_gradient_free(simulator, p, method):
    qubit = 0
    H = paulis.Qm(qubit)
    U = gates.Rx(target=qubit, angle=tq.Variable('a'))
    O = ExpectationValue(U=U, H=H)
    NM = BitFlip(p, 1)
    result = tq.optimizer_scipy.minimize(objective=O, samples=1, backend=simulator, method=method, noise=NM, tol=1.e-4,
                                         silent=False)


@pytest.mark.skipif(len(samplers) == 0, reason="Missing necessary backends")
@pytest.mark.parametrize("simulator", [numpy.random.choice(samplers)])
@pytest.mark.parametrize("p", numpy.random.uniform(0.1, .4, 1))
@pytest.mark.parametrize('method',
                         [tq.optimizer_scipy.OptimizerSciPy.gradient_based_methods[numpy.random.randint(0, 4, 1)[0]]])
def test_bit_flip_scipy_gradient(simulator, p, method):
    qubit = 0
    H = paulis.Qm(qubit)
    U = gates.Rx(target=qubit, angle=tq.Variable('a'))
    O = ExpectationValue(U=U, H=H)
    NM = BitFlip(p, 1)
    result = tq.optimizer_scipy.minimize(objective=O, samples=1, backend=simulator, method=method, noise=NM, tol=1.e-4,
                                         silent=False)


@pytest.mark.skipif(len(samplers) == 0, reason="Missing necessary backends")
@pytest.mark.parametrize("simulator", [numpy.random.choice(samplers)])
@pytest.mark.parametrize("p", numpy.random.uniform(0.1, .4, 1))
@pytest.mark.parametrize('method',
                         [["TRUST-KRYLOV", "NEWTON-CG", "TRUST-NCG", "TRUST-CONSTR"][numpy.random.randint(0, 4, 1)[0]]])
def test_bit_flip_scipy_hessian(simulator, p, method):
    qubit = 0
    H = paulis.Qm(qubit)
    U = gates.Rx(target=qubit, angle=tq.Variable('a'))
    O = ExpectationValue(U=U, H=H)
    NM = BitFlip(p, 1)
    result = tq.optimizer_scipy.minimize(objective=O, samples=1, backend=simulator, method=method, noise=NM, tol=1.e-4,
                                         silent=False)


@pytest.mark.skipif(len(samplers) == 0, reason="Missing necessary backends")
@pytest.mark.skipif(not tq.optimizers.has_phoenics, reason="Missing phoenics installation")
@pytest.mark.parametrize("simulator", [numpy.random.choice(samplers)])
@pytest.mark.parametrize("p", numpy.random.uniform(0.1, .4, 1))
def test_bit_flip_phoenics(simulator, p):
    qubit = 0
    H = paulis.Qm(qubit)
    U = gates.Rx(target=qubit, angle=tq.Variable('a'))
    O = ExpectationValue(U=U, H=H)
    NM = BitFlip(p, 1)
    result = tq.optimizers.optimizer_phoenics.minimize(objective=O, maxiter=3, samples=1, backend=simulator, noise=NM)


@pytest.mark.skipif(len(samplers) == 0, reason="Missing necessary backends")
@pytest.mark.skipif(not tq.optimizers.has_gpyopt, reason="Missing gpyopt installation")
@pytest.mark.parametrize("simulator", [numpy.random.choice(samplers)])
@pytest.mark.parametrize("p", numpy.random.uniform(0.1, .4, 1))
@pytest.mark.parametrize('method', ['lbfgs', 'DIRECT', 'CMA'])
def test_bit_flip_gpyopt(simulator, p, method):
    qubit = 0
    H = paulis.Qm(qubit)
    U = gates.Rx(target=qubit, angle=tq.Variable('a'))
    O = ExpectationValue(U=U, H=H)
    NM = BitFlip(p, 1)
    result = tq.optimizers.optimizer_gpyopt.minimize(objective=O, maxiter=10, samples=1, backend=simulator,
                                                     method=method, noise=NM)
