from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila.circuit import gates
from tequila.hamiltonian import paulis
from tequila.simulators.simulator_api import simulate

import numpy
import pytest

@pytest.mark.parametrize("paulis", [(gates.X,paulis.X), (gates.Y, paulis.Y), (gates.Z,paulis.Z)])
@pytest.mark.parametrize("qubit", [0,1,2])
@pytest.mark.parametrize("init", [0,1])
def test_pauli_gates(paulis, qubit, init):
    iwfn = QubitWaveFunction.from_int(i=init, n_qubits=qubit+1)
    wfn = simulate(paulis[0](qubit), initial_state=init)
    iwfn=iwfn.apply_qubitoperator(paulis[1](qubit))
    assert(iwfn.isclose(wfn))

@pytest.mark.parametrize("rot", [(gates.Rx,paulis.X), (gates.Ry, paulis.Y), (gates.Rz,paulis.Z)])
@pytest.mark.parametrize("angle", numpy.random.uniform(0.0, 2*numpy.pi, 5))
@pytest.mark.parametrize("qubit", [0,1,2])
@pytest.mark.parametrize("init", [0,1])
def test_rotations(rot, qubit, angle, init):
    pauli = rot[1](qubit)
    gate = rot[0](target=qubit, angle=angle)
    iwfn = QubitWaveFunction.from_int(i=init, n_qubits=qubit+1)
    wfn = simulate(gate, initial_state=init)
    test= numpy.cos(-angle/2.0)*iwfn + 1.0j*numpy.sin(-angle/2.0)* iwfn.apply_qubitoperator(pauli)
    assert(wfn.isclose(test))

@pytest.mark.parametrize("qubit", [0,2])
@pytest.mark.parametrize("init", [0,1])
def test_hadamard(qubit, init):
    gate = gates.H(target=qubit)
    iwfn = QubitWaveFunction.from_int(i=init, n_qubits=qubit+1)
    wfn = simulate(gate, initial_state=init)
    test= 1.0/numpy.sqrt(2)*(iwfn.apply_qubitoperator(paulis.Z(qubit)) + iwfn.apply_qubitoperator(paulis.X(qubit)))
    assert(wfn.isclose(test))

@pytest.mark.parametrize("target", [0,2])
@pytest.mark.parametrize("control", [1,3])
@pytest.mark.parametrize("gate", [gates.X, gates.Y, gates.Z, gates.H])
def test_controls(target, control, gate):
    c0 = gates.X(target=control) + gate(target=target, control=None)
    c1 = gates.X(target=control) + gate(target=target, control=control)
    wfn0 = simulate(c0, initial_state=0, backend="symbolic")
    wfn1 = simulate(c1, initial_state=0, backend="symbolic")
    assert(wfn0.isclose(wfn1))

    c0 = gates.QCircuit()
    c1 = gate(target=target, control=control)
    wfn0 = simulate(c0, initial_state=0)
    wfn1 = simulate(c1, initial_state=0)
    assert(wfn0.isclose(wfn1))




