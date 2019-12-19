from tequila.hamiltonian import QubitHamiltonian, PauliString, paulis
from tequila.grouping.binary_rep import BinaryPauliString, BinaryHamiltonian
from collections import namedtuple
import numpy as np

BinaryPauli = namedtuple("BinaryPauli", "coeff, binary")


def prepare_test_hamiltonian():
    '''
    Return a test hamiltonian and its solution
    '''
    H = -1.0 * paulis.Z(0) -0.5 * paulis.Z(1) + 0.1 * paulis.X(0)*paulis.X(1) 
    coeff_sol = np.array([-1.0, -0.5, 0.1])
    binary_sol = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0]])

    return H, H.n_qubits, binary_sol, coeff_sol

def test_binarypauli_conversion():
    '''
    Testing PauliString's built-in binary form conversion
    '''
    H, n_qubits, binary_sol, coeff_sol = prepare_test_hamiltonian()

    word1 = H.paulistrings[0].binary(n_qubits)
    word2 = H.paulistrings[1].binary(n_qubits)
    word3 = H.paulistrings[2].binary(n_qubits)

    assert (word1.coeff == coeff_sol[0])
    assert (all(word1.binary == binary_sol[0, :]))
    assert (word2.coeff == coeff_sol[1])
    assert (all(word2.binary == binary_sol[1, :]))
    assert (word3.coeff == coeff_sol[2])
    assert (all(word3.binary == binary_sol[2, :]))

def test_binary_pauli():
    '''
    Testing binary form of the pauli strings 
    '''

    x1 = BinaryPauliString([1, 0], 1)
    x1_other = BinaryPauliString(np.array([1, 0]), 2.1)
    assert (x1.same_pauli(x1_other))
    assert (x1.commute(x1_other))

    y1 = BinaryPauliString([1, 1], 1)
    assert (not x1.commute(y1))

    xx = BinaryPauliString([1, 1, 0, 0], 2)
    yy = BinaryPauliString([1, 1, 1, 1], 2.1 + 2j)
    assert (xx.commute(yy))

def test_binary_hamiltonian_initialization():
    '''
    Testing binary form of the hamiltonian
    '''
    H, n_qubits, binary_sol, coeff_sol = prepare_test_hamiltonian()

    H_binary = BinaryHamiltonian(H)

    binary_equal_matrix = H_binary.get_binary_matrix() == binary_sol

    assert (binary_equal_matrix.all())
    assert (all(H_binary.get_coeff() == coeff_sol))