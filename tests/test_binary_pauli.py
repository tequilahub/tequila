import tequila as tq
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
    word4 = paulis.I().paulistrings[0].binary(n_qubits)

    assert (word1.coeff == coeff_sol[0])
    assert (all(word1.binary == binary_sol[0, :]))
    assert (word2.coeff == coeff_sol[1])
    assert (all(word2.binary == binary_sol[1, :]))
    assert (word3.coeff == coeff_sol[2])
    assert (all(word3.binary == binary_sol[2, :]))
    assert (all(word4.binary == np.zeros(2 * n_qubits)))


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
        H).single_qubit_form()

    H_brute_force = brute_force_transformation(H, old_basis, new_basis)

    assert (equal_qubit_hamiltonian(single_qub_H.to_qubit_hamiltonian(),
                                    H_brute_force))

    H = -1.0 * paulis.X(0) * paulis.X(1) * paulis.X(2) + 2.0 * paulis.Y(
        0) * paulis.Y(1)

    single_qub_H, old_basis, new_basis = BinaryHamiltonian.init_from_qubit_hamiltonian(
        H).single_qubit_form()

    H_brute_force = brute_force_transformation(H, old_basis, new_basis)

    assert (equal_qubit_hamiltonian(single_qub_H.to_qubit_hamiltonian(),
                                    H_brute_force))


def brute_force_transformation(H, old_basis, new_basis):
    def pair_unitary(a, b):
        '''
        Accepts a BinaryPauliString. 
        Return the paired unitary 1/sqrt(2) (a + b) in qubit hamiltonian. 
        '''
        a = QubitHamiltonian.from_paulistrings(a.to_pauli_strings())
        b = QubitHamiltonian.from_paulistrings(b.to_pauli_strings())
        return (1 / 2) ** (1 / 2) * (a + b)

    U = QubitHamiltonian(1)
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


def test_commuting_groups():
    '''
    Testing whether the partitioning gives commuting parts
    '''
    H, _, _, _ = prepare_test_hamiltonian()
    H = H + paulis.X(0) + paulis.Y(0)
    H = BinaryHamiltonian.init_from_qubit_hamiltonian(H)

    commuting_parts = H.commuting_groups()

    for part in commuting_parts:
        assert part.is_commuting()


def test_qubit_wise_commuting():
    '''
    Testing whether method is_qubit_wise_commuting correctly 
    recognizes qubit wise commuting parts.
    '''
    not_qwc = -1.0 * paulis.Z(0) * paulis.Z(1) - 0.5 * paulis.Y(0) * paulis.Y(1)
    not_qwc = BinaryHamiltonian.init_from_qubit_hamiltonian(not_qwc)
    qwc = paulis.Z(0) * paulis.Z(1) + paulis.Z(1) * paulis.Y(2)
    qwc = BinaryHamiltonian.init_from_qubit_hamiltonian(qwc)

    assert not not_qwc.is_qubit_wise_commuting()
    assert qwc.is_qubit_wise_commuting()


def test_get_qubit_wise():
    '''
    Testing whether the get_qubit_wise methods correctly gives the all-Z form of the hamiltonian
    '''
    H, _, _, _ = prepare_test_hamiltonian()
    H = BinaryHamiltonian.init_from_qubit_hamiltonian(H)
    qwc, qwc_U = H.get_qubit_wise()

    # Check qwc has all z
    for term, val in qwc.items():
        for qub in term:
            assert qub[1] == 'Z'

    # Checking the expectation values are the same
    U = tq.gates.ExpPauli(angle="a", paulistring=tq.PauliString.from_string('X(0)Y(1)'))
    variables = {"a": np.random.rand(1) * 2 * np.pi}

    e_ori = tq.ExpectationValue(H=H.to_qubit_hamiltonian(), U=U)
    e_qwc = tq.ExpectationValue(H=qwc, U=U + qwc_U)
    e_integrated = tq.ExpectationValue(H=H.to_qubit_hamiltonian(), U=U, optimize_measurements=True)
    result_ori = tq.simulate(e_ori, variables)
    result_qwc = tq.simulate(e_qwc, variables)
    result_integrated = tq.simulate(e_qwc, variables)

    assert np.isclose(result_ori, result_qwc)
    assert np.isclose(result_ori, result_integrated)

    # Checking the optimized expectation values are the same
    initial_values = {k: np.random.uniform(0.0, 6.0, 1) for k in e_ori.extract_variables()}
    sol1 = tq.minimize(method='bfgs', objective=e_ori, initial_values=initial_values)
    sol2 = tq.minimize(method='bfgs', objective=e_qwc, initial_values=initial_values)
    sol3 = tq.minimize(method='bfgs', objective=e_integrated, initial_values=initial_values)

    assert np.isclose(sol1.energy, sol2.energy)
    assert np.isclose(sol1.energy, sol3.energy)
