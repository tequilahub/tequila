from tequila.hamiltonian import QubitHamiltonian, PauliString, paulis
from tequila.grouping.parser import hamiltonian_to_binary
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

def test_binarypauli_group_conversion():
    '''
    Testing binary form conversion for entire Hamiltonian
    '''
    H, n_qubits, binary_sol, coeff_sol = prepare_test_hamiltonian()

    H_binary, H_coeff = hamiltonian_to_binary(H)

    binary_equal_matrix = H_binary == binary_sol

    assert (binary_equal_matrix.all())
    assert (all(H_coeff == coeff_sol))

test_binarypauli_group_conversion()