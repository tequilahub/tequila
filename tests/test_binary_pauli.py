from tequila.hamiltonian import QubitHamiltonian, PauliString, paulis
from tequila.grouping.binary_rep import BinaryPauliString, BinaryHamiltonian
from collections import namedtuple
import numpy as np

BinaryPauli = namedtuple("BinaryPauli", "coeff, binary")


def prepare_test_hamiltonian():
    '''
    Return a test hamiltonian and its solution
    '''
    H = -1.0 * paulis.Z(0) * paulis.Z(1) - 0.5 * paulis.Y(0) * paulis.Y(
        1) + 0.1 * paulis.X(0) * paulis.X(1) + 0.2 * paulis.Z(2)
    coeff_sol = np.array([-1.0, -0.5, 0.1, 0.2])
    binary_sol = np.array([[0, 0, 0, 1, 1, 0], [1, 1, 0, 1, 1, 0],
                           [1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]])

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

    H_binary = BinaryHamiltonian.init_from_qubit_hamiltonian(H)

    binary_equal_matrix = np.array(H_binary.get_binary()) == binary_sol

    assert (binary_equal_matrix.all())
    assert (all(np.array(H_binary.get_coeff()) == coeff_sol))


def test_to_qubit_hamiltonian():
    '''
    Testing transformation to qubit hamiltonian 
    '''
    H, n_qubits, binary_sol, coeff_sol = prepare_test_hamiltonian()

    binary_hamiltonian = BinaryHamiltonian.init_from_qubit_hamiltonian(H)

    assert (equal_qubit_hamiltonian(H,
                                    binary_hamiltonian.to_qubit_hamiltonian()))


def test_single_qubit_basis_transfrom():
    '''
    Testing whether transformations using the binary form 
    and the transformation through direct computation agree
    '''
    H, n_qubits, binary_sol, coeff_sol = prepare_test_hamiltonian()

    single_qub_H, old_basis, new_basis = BinaryHamiltonian.init_from_qubit_hamiltonian(
        H).get_qubit_wise()

    H_brute_force = brute_force_transformation(H, old_basis, new_basis)

    assert (equal_qubit_hamiltonian(single_qub_H.to_qubit_hamiltonian(),
                                    H_brute_force))

    H = -1.0 * paulis.X(0) * paulis.X(1) * paulis.X(2) + 2.0 * paulis.Y(
        0) * paulis.Y(1)

    single_qub_H, old_basis, new_basis = BinaryHamiltonian.init_from_qubit_hamiltonian(
        H).get_qubit_wise()

    H_brute_force = brute_force_transformation(H, old_basis, new_basis)

    assert (equal_qubit_hamiltonian(single_qub_H.to_qubit_hamiltonian(),
                                    H_brute_force))


def brute_force_transformation(H, old_basis, new_basis):
    def pair_unitary(a, b):
        '''
        Accepts a BinaryPauliString. 
        Return the paired unitary 1/sqrt(2) (a + b) in qubit hamiltonian. 
        '''
        a = QubitHamiltonian.init_from_paulistring(a.to_pauli_strings())
        b = QubitHamiltonian.init_from_paulistring(b.to_pauli_strings())
        return (1 / 2)**(1 / 2) * (a + b)

    U = QubitHamiltonian.init_unit()
    for i, i_basis in enumerate(old_basis):
        U *= pair_unitary(i_basis, new_basis[i])

    return U * H * U


def equal_qubit_hamiltonian(a, b):
    tiny = 1e-6
    for key, value in a.items():
        if key in b.keys():
            if not (abs(value - b[key]) < tiny):
                return False
        else:
            if not (abs(value) < tiny):
                return False
    return True
