import importlib.util
import numpy as np
from numpy.testing import assert_almost_equal
import tequila as tq
import pytest
import importlib


test_case = np.array(
    [
        [
            0.70710678 + 0.0j,
            0.70710678 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
        ],
        [
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.70710678 + 0.0j,
            -0.70710678 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
        ],
        [
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.70710678 + 0.0j,
            0.70710678 + 0.0j,
        ],
        [
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.70710678 + 0.0j,
            -0.70710678 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
        ],
        [
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.70710678 + 0.0j,
            0.70710678 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
        ],
        [
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.70710678 + 0.0j,
            -0.70710678 + 0.0j,
        ],
        [
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.70710678 + 0.0j,
            0.70710678 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
        ],
        [
            0.70710678 + 0.0j,
            -0.70710678 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
        ],
    ]
)

# Check if quimb is installed
HAS_QUIMB = importlib.util.find_spec("quimb") is not None


@pytest.mark.skipif(condition=not HAS_QUIMB, reason="quimb not installed")
def test_circuit_to_matrix():
    """
    Test the conversion of a 3-qubit circuit to a unitary matrix.
    """
    circuit = tq.gates.H(target=0) + tq.gates.CNOT(target=1, control=0) + tq.gates.CNOT(target=2, control=1)

    # Convert the circuit to a unitary matrix
    unitary_matrix = circuit.to_matrix()

    # Compare with the expected result
    assert_almost_equal(unitary_matrix, test_case, decimal=8)


@pytest.mark.skipif(condition=not HAS_QUIMB, reason="quimb not installed")
def test_circuit_to_matrix_with_params():
    PM = np.kron(tq.paulis.Y(0).to_matrix(), tq.paulis.X(0).to_matrix())
    U = tq.gates.ExpPauli(paulistring="X(0)Y(1)", angle="a")

    N = 2**2

    for a in [1.0, 2.0, -1.0]:
        UM1 = U.to_matrix({"a": a})
        UM2 = np.cos(-a / 2) * np.eye(N) + 1.0j * np.sin(-a / 2) * PM
        assert_almost_equal(UM1, UM2)
