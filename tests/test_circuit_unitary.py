import numpy as np
from numpy.testing import assert_almost_equal
from tequila import QCircuit


test_case = np.load("tests/data/circuit_3qubit_to_matrix_test.npy")

def test_circuit_to_matrix():
    """
    Test the conversion of a 3-qubit circuit to a unitary matrix.
    """
    circuit = QCircuit(3)

    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)

    # Convert the circuit to a unitary matrix
    unitary_matrix = circuit.to_matrix()

    # Compare with the expected result
    assert_almost_equal(unitary_matrix, test_case, decimal=8)