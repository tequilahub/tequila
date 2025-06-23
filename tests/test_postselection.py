from math import sqrt
import numpy as np
import tequila as tq
from tequila import QubitWaveFunction
from tequila.circuit.postselection import PostselectionCircuit, Postselection


def test_without_postselection():
    U = tq.gates.H(0) + tq.gates.CNOT(0, 1)
    postselection_circuit = PostselectionCircuit(U)

    wfn, norm = postselection_circuit.simulate()
    ref_wfn = tq.simulate(U)

    assert np.isclose(norm, 1.0)
    assert wfn.isclose(ref_wfn)


def test_with_postselection():
    U = tq.gates.H(0) + tq.gates.CNOT(0, 1)
    postselection_circuit = PostselectionCircuit(U)
    postselection_circuit += Postselection([0])

    wfn, norm = postselection_circuit.simulate()
    ref_wfn = QubitWaveFunction.from_array(np.array([1.0, 0.0, 0.0, 0.0]))

    assert np.isclose(norm, 1 / sqrt(2))
    assert wfn.isclose(ref_wfn)


def test_multiple_fragments():
    U = tq.gates.H(0) + tq.gates.CNOT(0, 1) + tq.gates.X(0)
    U.n_qubits = 3
    postselection_circuit = PostselectionCircuit(U)
    postselection_circuit += Postselection([0])
    V = tq.gates.H(1) + tq.gates.CNOT(1, 2) + tq.gates.X(1)
    postselection_circuit += PostselectionCircuit(V)
    postselection_circuit += Postselection([1])

    wfn, norm = postselection_circuit.simulate()
    ref_wfn = QubitWaveFunction.from_array(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    assert np.isclose(norm, 1 / 2)
    assert wfn.isclose(ref_wfn)


def test_impossible_postselection():
    U = tq.gates.H(0) + tq.gates.CNOT(0, 1) + tq.gates.X(1)
    postselection_circuit = PostselectionCircuit(U)
    postselection_circuit += Postselection([0, 1])

    wfn, norm = postselection_circuit.simulate()
    assert np.isclose(norm, 0.0)
    assert np.allclose(wfn.to_array(), np.array([0.0, 0.0, 0.0, 0.0]))
