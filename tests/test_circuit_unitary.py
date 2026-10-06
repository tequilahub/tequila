import importlib.util
import numpy as np
from numpy.testing import assert_almost_equal
import tequila as tq
import pytest
import importlib

# Check if quimb is installed
HAS_QUIMB = importlib.util.find_spec("quimb") is not None


def controlled_x_matrix(control, target, n_qubits):
    """Build a CNOT matrix using tequila's MSB-first basis convention."""
    dim = 2**n_qubits
    matrix = np.zeros((dim, dim), dtype=complex)
    for column in range(dim):
        control_bit = (column >> (n_qubits - 1 - control)) & 1
        row = column
        if control_bit:
            row ^= 1 << (n_qubits - 1 - target)
        matrix[row, column] = 1
    return matrix


@pytest.mark.skipif(condition=not HAS_QUIMB, reason="quimb not installed")
def test_circuit_to_matrix():
    """Test the conversion of a 3-qubit circuit to a unitary matrix."""
    circuit = tq.gates.H(target=0) + tq.gates.CNOT(target=1, control=0) + tq.gates.CNOT(target=2, control=1)
    unitary_matrix = circuit.to_matrix()

    h = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    identity = np.eye(2, dtype=complex)
    expected = (
        controlled_x_matrix(control=1, target=2, n_qubits=3)
        @ controlled_x_matrix(control=0, target=1, n_qubits=3)
        @ np.kron(np.kron(h, identity), identity)
    )

    assert_almost_equal(unitary_matrix, expected, decimal=8)


@pytest.mark.skipif(condition=not HAS_QUIMB, reason="quimb not installed")
@pytest.mark.parametrize(
    ("target", "n_qubits", "expected"),
    [
        (1, 2, np.kron(np.eye(2), np.array([[0, 1], [1, 0]]))),
        (2, 3, np.kron(np.eye(4), np.array([[0, 1], [1, 0]]))),
    ],
)
def test_circuit_to_matrix_with_idle_qubits(target, n_qubits, expected):
    circuit = tq.gates.X(target=target)
    circuit.n_qubits = n_qubits

    assert_almost_equal(circuit.to_matrix(), expected)


@pytest.mark.skipif(condition=not HAS_QUIMB, reason="quimb not installed")
def test_circuit_to_matrix_with_params():
    # X(0)Y(1) is X on qubit 0 and Y on qubit 1; qubit 0 is the MSB.
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1.0j], [1.0j, 0]], dtype=complex)
    pauli_matrix = np.kron(x, y)
    unitary = tq.gates.ExpPauli(paulistring="X(0)Y(1)", angle="a")
    n = 2**2

    for angle in [1.0, 2.0, -1.0]:
        actual = unitary.to_matrix({"a": angle})
        expected = np.cos(angle / 2) * np.eye(n) - 1.0j * np.sin(angle / 2) * pauli_matrix
        assert_almost_equal(actual, expected)
