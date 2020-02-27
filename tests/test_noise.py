from tequila.circuit import gates
from tequila.objective import ExpectationValue
from tequila.objective.objective import Variable
from tequila.hamiltonian import paulis
from tequila import simulate
from tequila import simulators
from tequila.circuit.noise import NoiseModel,BitFlip,PhaseDamp,PhaseFlip,AmplitudeDamp
import numpy
import pytest

@pytest.mark.parametrize("simulator", [simulators.pick_backend('qiskit')])
@pytest.mark.parametrize("p", numpy.random.uniform(0.,1.,1))
def test_bit_flip(simulator, p):


    qubit = 0
    H = paulis.Qm(qubit)
    U = gates.X(target=qubit)
    O = ExpectationValue(U=U, H=H)
    NM=BitFlip(p,['x'])
    E = simulate(O,backend=simulator,samples=100000,noise_model=NM)
    assert (numpy.isclose(E, 1.0-p, atol=1.e-2))



@pytest.mark.parametrize("simulator", [simulators.pick_backend('qiskit')])
@pytest.mark.parametrize("p", numpy.random.uniform(0.,1.,1))
def test_phase_flip(simulator, p):


    qubit = 0
    H = paulis.Qm(qubit)
    U = gates.Y(target=qubit)
    O = ExpectationValue(U=U, H=H)
    NM=PhaseFlip(p,['y'])
    E = simulate(O,backend=simulator,samples=100000,noise_model=NM)
    assert (numpy.isclose(E, 1.0-p, atol=1.e-2))


@pytest.mark.parametrize("simulator", [simulators.pick_backend('qiskit')])
@pytest.mark.parametrize("p", numpy.random.uniform(0.,1.,1))
def test_phase_damp(simulator, p):


    qubit = 0
    H = paulis.Qm(qubit)
    U = gates.H(target=qubit)
    O = ExpectationValue(U=U, H=H)
    NM=PhaseDamp(p,['h'])
    E = simulate(O,backend=simulator,samples=100000,noise_model=NM)
    assert (numpy.isclose(E, 0.5, atol=1.e-2))


@pytest.mark.parametrize("simulator", [simulators.pick_backend('qiskit')])
@pytest.mark.parametrize("p", numpy.random.uniform(0.,1.,1))
def test_amp_damp(simulator, p):


    qubit = 0
    H = (0.5)*(paulis.I(0)-paulis.Z(0))
    U = gates.X(target=qubit)
    O = ExpectationValue(U=U, H=H)
    NM=AmplitudeDamp(p,['x'])
    E = simulate(O,backend=simulator,samples=100000,noise_model=NM)
    assert (numpy.isclose(E, 1-p, atol=1.e-2))