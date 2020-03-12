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


@pytest.mark.parametrize("simulator", ['qiskit','pyquil','cirq'])
@pytest.mark.parametrize("p", numpy.random.uniform(0.,1.,1))
@pytest.mark.parametrize('controlled',[False,True])
def test_bit_flip(simulator, p,controlled):


    qubit = 0
    H = paulis.Qm(qubit)
    if controlled:
        U = gates.X(target=1)+gates.CX(1,0)
        NM = BitFlip(p, ['cx'])
    else:
        U = gates.X(target=0)
        NM = BitFlip(p, ['x'])
    O = ExpectationValue(U=U, H=H)

    E = simulate(O,backend=simulator,samples=100000,noise_model=NM)
    assert (numpy.isclose(E, 1.0-p, atol=1.e-2))



@pytest.mark.parametrize("simulator", ['qiskit','pyquil','cirq'])
@pytest.mark.parametrize("p", numpy.random.uniform(0.,1.,1))
def test_phase_flip(simulator, p):


    qubit = 0
    H = paulis.Qm(qubit)
    U = gates.H(target=qubit)+gates.Z(target=qubit)+gates.H(target=qubit)
    O = ExpectationValue(U=U, H=H)
    NM=PhaseFlip(p,['Z'])
    E = simulate(O,backend=simulator,samples=100000,noise_model=NM)
    assert (numpy.isclose(E, 1.0-p, atol=1.e-2))


@pytest.mark.parametrize("simulator", ['qiskit','pyquil','cirq'])
@pytest.mark.parametrize("p", numpy.random.uniform(0.,1.,1))
def test_phase_damp(simulator, p):


    qubit = 0
    H = paulis.Qm(qubit)
    U = gates.H(target=qubit)
    O = ExpectationValue(U=U, H=H)
    NM=PhaseDamp(p,['h'])
    E = simulate(O,backend=simulator,samples=100000,noise_model=NM)
    assert (numpy.isclose(E, 0.5, atol=1.e-2))


@pytest.mark.parametrize("simulator", ['qiskit','pyquil','cirq'])
@pytest.mark.parametrize("p", numpy.random.uniform(0.,1.,1))
def test_amp_damp(simulator, p):


    qubit = 0
    H = (0.5)*(paulis.I(0)-paulis.Z(0))
    U = gates.X(target=qubit)
    O = ExpectationValue(U=U, H=H)
    NM=AmplitudeDamp(p,['x'])
    E = simulate(O,backend=simulator,samples=100000,noise_model=NM)
    assert (numpy.isclose(E, 1-p, atol=1.e-2))


@pytest.mark.parametrize("simulator", ['qiskit','pyquil','cirq'])
@pytest.mark.parametrize("p", numpy.random.uniform(0.,1.,1))
def test_phase_amp_damp(simulator, p):


    qubit = 0
    H = paulis.Z(0)
    U = gates.X(target=qubit)
    O = ExpectationValue(U=U, H=H)
    NM=PhaseAmplitudeDamp(p,1-p,['x'])
    E = simulate(O,backend=simulator,samples=100000,noise_model=NM)
    assert (numpy.isclose(E, -1+2*p, atol=1.e-2))


@pytest.mark.parametrize("simulator", ['qiskit','pyquil','cirq'])
@pytest.mark.parametrize("p", numpy.random.uniform(0.,1.,1))
def test_phase_amp_damp_is_both(simulator, p):


    qubit = 0
    H = paulis.Z(0)
    U = gates.X(target=qubit)
    O = ExpectationValue(U=U, H=H)
    NM1=PhaseDamp(1-p,['x'])+AmplitudeDamp(p,['x'])
    E1 = simulate(O,backend=simulator,samples=100000,noise_model=NM1)
    NM2 = PhaseAmplitudeDamp(p,1-p, ['x'])
    E2 =simulate(O,backend=simulator,samples=100000,noise_model=NM2)
    assert (numpy.isclose(E1,E2, atol=1.e-2))

@pytest.mark.parametrize("simulator", ['qiskit','pyquil','cirq'])
@pytest.mark.parametrize("p", numpy.random.uniform(0.,1.,1))
@pytest.mark.parametrize('controlled',[False,True])
def test_depolarizing_error(simulator, p,controlled):

    cq=1
    qubit = 0
    H = paulis.Z(0)
    if controlled:
        U = gates.X(target=cq)+gates.X(target=qubit,control=cq)
        NM = DepolarizingError(p, ['cx'])
    else:
        U= gates.X(target=qubit)
        NM = DepolarizingError(p, ['x'])
    O = ExpectationValue(U=U, H=H)

    E = simulate(O,backend=simulator,samples=100000,noise_model=NM)
    assert (numpy.isclose(E, -1+p, atol=1.e-2))

@pytest.mark.parametrize("simulator", ['qiskit','pyquil','cirq'])
@pytest.mark.parametrize("p", numpy.random.uniform(0.,1.,1))
def test_repetition_works(simulator, p):
    qubit = 0
    H = paulis.Qm(qubit)
    U = gates.X(target=qubit)+gates.X(target=qubit)
    O = ExpectationValue(U=U, H=H)
    NM=BitFlip(p,['x'])
    E = simulate(O,backend=simulator,samples=100000,noise_model=NM)
    assert (numpy.isclose(E, 2*(p-p*p), atol=1.e-2))

