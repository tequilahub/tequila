import importlib.util
import numpy as np
from numpy.testing import assert_almost_equal
import tequila as tq
import pytest
import importlib

# Check if quimb is installed
HAS_QUIMB = importlib.util.find_spec("quimb") is not None
from tequila import BitNumbering

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


def test_circuit_to_matrix():
    """
    Test the conversion of a 3-qubit circuit to a unitary matrix.
    """
    circuit = tq.gates.H(target=0) + tq.gates.CNOT(target=1, control=0) + tq.gates.CNOT(target=2, control=1)

    # Convert the circuit to a unitary matrix
    unitary_matrix = circuit.to_matrix(numbering=BitNumbering.LSB)

    # internal test case
    HH = 1.0 / np.sqrt(2) * (tq.paulis.X(0) + tq.paulis.Z(0))
    CNOT1 = tq.paulis.X(1) * (1 - tq.paulis.Z(0)) * 0.5 + 0.5 * (1 + tq.paulis.Z(0))
    CNOT2 = tq.paulis.X(2) * (1 - tq.paulis.Z(1)) * 0.5 + 0.5 * (1 + tq.paulis.Z(1))

    M = (CNOT2 * CNOT1 * HH).to_matrix()

    # Compare with the expected result
    assert_almost_equal(unitary_matrix, M, decimal=8)


def test_circuit_to_matrix_with_params():
    # X(0)Y(1) is X on qubit 0 and Y on qubit 1. Tequila's matrix convention is big endian,
    # so this is kron(X, Y) and not kron(Y, X).
    PM = (tq.paulis.X(0) * tq.paulis.Y(1)).to_matrix()
    U = tq.gates.ExpPauli(paulistring="X(0)Y(1)", angle="a")

    N = 2**2

    for a in [1.0, 2.0, -1.0]:
        UM1 = U.to_matrix({"a": a})
        UM2 = np.cos(a / 2) * np.eye(N) - 1.0j * np.sin(a / 2) * PM
        UM1 = U.to_matrix({"a": a}, numbering=BitNumbering.LSB)
        UM2 = np.cos(-a / 2) * np.eye(N) + 1.0j * np.sin(-a / 2) * PM
        assert_almost_equal(UM1, UM2)
