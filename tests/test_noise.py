from tequila.circuit import gates
from tequila.objective import ExpectationValue
from tequila.objective.objective import Variable
from tequila.hamiltonian import paulis
from tequila import simulate
import tequila
from tequila.circuit.noise import BitFlip, PhaseDamp, PhaseFlip, AmplitudeDamp, PhaseAmplitudeDamp, DepolarizingError
import numpy
import pytest
from tequila.simulators.simulator_api import SUPPORTED_NOISE_BACKENDS
samplers = [k for k in tequila.INSTALLED_SAMPLERS.keys() if k in SUPPORTED_NOISE_BACKENDS]


@pytest.mark.dependencies
def test_dependencies():
    for k in SUPPORTED_NOISE_BACKENDS:
        assert k in samplers

@pytest.mark.skipif(len(samplers) == 0, reason="Missing necessary backends")
@pytest.mark.parametrize("simulator", samplers)
@pytest.mark.parametrize("p", numpy.random.uniform(0., 1., 1))
@pytest.mark.parametrize('controlled', [False, True])
def test_bit_flip(simulator, p, controlled):
    qubit = 0

    if controlled:
        U = gates.X(target=1) + gates.CX(1, 0)
        H = paulis.Qm(0)
        NM = BitFlip(p, 2)
    else:
        U = gates.X(target=0)
        NM = BitFlip(p, 1)
        H = paulis.Qm(qubit)
    O = ExpectationValue(U=U, H=H)

    E = simulate(O, backend=simulator, samples=1000, noise=NM)
    assert (numpy.isclose(E, 1.0 - p, atol=1.e-1))


@pytest.mark.skipif(len(samplers) == 0, reason="Missing necessary backends")
@pytest.mark.parametrize("simulator", samplers)
@pytest.mark.parametrize("p", numpy.random.uniform(0., 1., 1))
@pytest.mark.parametrize("angle", numpy.random.uniform(0., 2 * numpy.pi, 1))
def test_rx_bit_flip_0(simulator, p, angle):
    U = gates.Rx(target=0, angle=Variable('a'))
    H = paulis.Z(0)
    NM = BitFlip(p, 1)

    O = ExpectationValue(U=U, H=H)

    E = simulate(O, backend=simulator, samples=1000, variables={'a': angle}, noise=NM)
    assert (numpy.isclose(E, (1 - 2 * p) * numpy.cos(angle), atol=1.e-1))


@pytest.mark.skipif(len(samplers) == 0, reason="Missing necessary backends")
@pytest.mark.parametrize("simulator", samplers)
@pytest.mark.parametrize("p", numpy.random.uniform(0., 1., 1))
@pytest.mark.parametrize("angle", numpy.random.uniform(0., 2 * numpy.pi, 1))
def test_rx_bit_flip_1(simulator, p, angle):
    U = gates.X(target=0) + gates.CRx(control=0, target=1, angle="a")
    H = paulis.Z(1) * paulis.I(0)
    NM = BitFlip(p, 2)
    O = ExpectationValue(U=U, H=H)

    E = simulate(O, backend=simulator, samples=1000, variables={'a': angle}, noise=NM)
    print(E)
    print(p + numpy.cos(angle) - p * numpy.cos(angle))
    assert (numpy.isclose(E, p + numpy.cos(angle) - p * numpy.cos(angle), atol=1.e-1))


@pytest.mark.skipif(len(samplers) == 0, reason="Missing necessary backends")
@pytest.mark.parametrize("simulator", samplers)
@pytest.mark.parametrize("p", numpy.random.uniform(0., 1., 1))
def test_double_cnot_bit_flip(simulator, p):
    qubit = 1
    U = gates.X(0) + gates.X(2) + gates.CX(0, 1) + gates.CX(2, 1)
    H = paulis.Qm(qubit)
    O = ExpectationValue(U=U, H=H)
    NM = BitFlip(p, 2)

    E = simulate(O, backend=simulator, samples=1000, noise=NM)
    assert (numpy.isclose(E, 2 * (p - p * p), atol=1.e-1))


@pytest.mark.skipif(len(samplers) == 0, reason="Missing necessary backends")
@pytest.mark.parametrize("simulator", samplers)
@pytest.mark.parametrize("p", numpy.random.uniform(0., 1., 1))
def test_phase_flip(simulator, p):
    qubit = 0
    H = paulis.X(qubit)
    U = gates.H(target=qubit)
    O = ExpectationValue(U=U, H=H)
    NM = PhaseFlip(p, 1)
    E = simulate(O, backend=simulator, samples=1000, noise=NM)
    assert (numpy.isclose(E, 1.0 - 2 * p, atol=1.e-1))


@pytest.mark.skipif(len(samplers) == 0, reason="Missing necessary backends")
@pytest.mark.parametrize("simulator", samplers)
@pytest.mark.parametrize("p", numpy.random.uniform(0., 1., 1))
@pytest.mark.parametrize("angle", numpy.random.uniform(0., 2 * numpy.pi, 1))
def test_rz_phase_flip_0(simulator, p, angle):
    qubit = 0
    H = paulis.Y(qubit)
    U = gates.H(target=qubit) + gates.Rz(angle=Variable('a'), target=qubit) + gates.H(target=qubit)
    O = ExpectationValue(U=U, H=H)
    NM = PhaseFlip(p, 1)
    E = simulate(O, backend=simulator, variables={'a': angle}, samples=1000, noise=NM)
    assert (numpy.isclose(E, ((-1. + 2 * p) ** 3) * numpy.sin(angle), atol=1.e-1))


@pytest.mark.skipif(len(samplers) == 0, reason="Missing necessary backends")
@pytest.mark.parametrize("simulator", samplers)
@pytest.mark.parametrize("p", numpy.random.uniform(0., 1., 1))
@pytest.mark.parametrize("angle", numpy.random.uniform(0., 2 * numpy.pi, 1))
def test_rz_phase_flip_1(simulator, p, angle):
    U = gates.X(target=0) + gates.H(1) + gates.CRz(control=0, target=1, angle=Variable('a')) + gates.H(1)
    H = paulis.Z(1) * paulis.I(0)
    O = ExpectationValue(U, H)
    NM = PhaseFlip(p, 2)
    E = simulate(O, backend=simulator, variables={'a': angle}, samples=1000, noise=NM)
    print(E)
    assert (numpy.isclose(E, ((1.0 - 2 * p) ** 2) * numpy.cos(angle), atol=1.e-1))


@pytest.mark.skipif(len(samplers) == 0, reason="Missing necessary backends")
@pytest.mark.parametrize("simulator", samplers)
@pytest.mark.parametrize("p", numpy.random.uniform(0., 1., 1))
def test_phase_damp(simulator, p):
    qubit = 0
    H = paulis.X(qubit)
    U = gates.H(target=qubit)
    O = ExpectationValue(U=U, H=H)
    NM = PhaseDamp(p, 1)
    E = simulate(O, backend=simulator, samples=1000, noise=NM)
    assert (numpy.isclose(E, numpy.sqrt(1 - p), atol=1.e-1))


@pytest.mark.skipif(len(samplers) == 0, reason="Missing necessary backends")
@pytest.mark.parametrize("simulator", samplers)
@pytest.mark.parametrize("p", numpy.random.uniform(0., 1., 1))
def test_amp_damp(simulator, p):
    qubit = 0
    H = (0.5) * (paulis.I(0) - paulis.Z(0))
    U = gates.X(target=qubit)
    O = ExpectationValue(U=U, H=H)
    NM = AmplitudeDamp(p, 1)
    E = simulate(O, backend=simulator, samples=1, noise=NM)
    # assert (numpy.isclose(E, 1-p, atol=1.e-1))


@pytest.mark.skipif(len(samplers) == 0, reason="Missing necessary backends")
@pytest.mark.parametrize("simulator", samplers)
@pytest.mark.parametrize("p", numpy.random.uniform(0., 1., 1))
def test_phase_amp_damp(simulator, p):
    qubit = 0
    H = paulis.Z(0)
    U = gates.X(target=qubit)
    O = ExpectationValue(U=U, H=H)
    NM = PhaseAmplitudeDamp(p, 1 - p, 1)
    E = simulate(O, backend=simulator, samples=1, noise=NM)
    # assert (numpy.isclose(E, -1+2*p, atol=1.e-1))


@pytest.mark.skipif(len(samplers) == 0, reason="Missing necessary backends")
@pytest.mark.parametrize("simulator", samplers)
@pytest.mark.parametrize("p", numpy.random.uniform(0., 1., 1))
def test_phase_amp_damp_is_both(simulator, p):
    qubit = 0
    H = paulis.Z(0)
    U = gates.X(target=qubit)
    O = ExpectationValue(U=U, H=H)
    NM1 = PhaseDamp(1 - p, 1) + AmplitudeDamp(p, 1)
    E1 = simulate(O, backend=simulator, samples=1, noise=NM1)
    NM2 = PhaseAmplitudeDamp(p, 1 - p, 1)
    E2 = simulate(O, backend=simulator, samples=1, noise=NM2)
    # assert (numpy.isclose(E1,E2, atol=1.e-1))


@pytest.mark.skipif(len(samplers) == 0, reason="Missing necessary backends")
@pytest.mark.parametrize("simulator", samplers)
@pytest.mark.parametrize("p", numpy.random.uniform(0., 1., 1))
@pytest.mark.parametrize('controlled', [False, True])
def test_depolarizing_error(simulator, p, controlled):
    cq = 1
    qubit = 0
    H = paulis.Z(0)
    if controlled:
        U = gates.X(target=cq) + gates.X(target=qubit, control=cq)
        NM = DepolarizingError(p, 2)
    else:
        U = gates.X(target=qubit)
        NM = DepolarizingError(p, 1)
    O = ExpectationValue(U=U, H=H)

    E = simulate(O, backend=simulator, samples=1, noise=NM)
    # assert (numpy.isclose(E, -1+p, atol=1.e-1))


@pytest.mark.skipif(len(samplers) == 0, reason="Missing necessary backends")
@pytest.mark.parametrize("simulator", samplers)
@pytest.mark.parametrize("p", numpy.random.uniform(0., 1., 1))
def test_repetition_works(simulator, p):
    qubit = 0
    H = paulis.Qm(qubit)
    U = gates.X(target=qubit) + gates.X(target=qubit)
    O = ExpectationValue(U=U, H=H)
    NM = BitFlip(p, 1)
    E = simulate(O, backend=simulator, samples=1, noise=NM)
    # assert (numpy.isclose(E, 2*(p-p*p), atol=1.e-1))
