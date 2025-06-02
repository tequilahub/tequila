import numpy as np
from numpy.testing import assert_almost_equal
import tequila as tq


test_case = np.load("data/circuit_3qubit_to_matrix_test.npy")

def test_circuit_to_matrix():
    """
    Test the conversion of a 3-qubit circuit to a unitary matrix.
    """
    circuit = tq.gates.H(target=0) + tq.gates.CNOT(target=1,control=0) + tq.gates.CNOT(target=2,control=1)

    # Convert the circuit to a unitary matrix
    unitary_matrix = circuit.to_matrix()

    # Compare with the expected result
    assert_almost_equal(unitary_matrix, test_case, decimal=8)

def test_circuit_to_matrix_with_params():
    P = tq.paulis.X(0) + tq.paulis.Y(1)
    U = tq.gates.ExpPauli(paulistring="X(0)Y(1)", angle="a")

    PM = P.to_matrix()
    N = 2**P.n_qubits

    for a in [1.0, 2.0, -1.0]:
        UM1 = U.to_matrix({"a":a})
        UM2 = np.cos(a/2) * np.eye(N) + 1.0j * np.sin(a/2) * PM
        assert_almost_equal(UM1, UM2)