import tequila as tq
from tequila.hamiltonian import QubitHamiltonian, PauliString, paulis 
from tequila.grouping.binary_rep import BinaryPauliString, BinaryHamiltonian
from tequila.grouping.overlapping_methods import OverlappingGroups, OverlappingAuxiliary
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


def test_binary_pauli_equality():
    '''
    Testing __eq__ implemented in binary form of the pauli strings.
    '''
    x1 = BinaryPauliString([1, 0], 1.)
    x1_other_same_coeff = BinaryPauliString(np.array([1, 0]), 1.)
    x1_other_diff_coeff = BinaryPauliString(np.array([1, 0]), 1. + 1e-9)
    assert x1 == x1_other_same_coeff
    assert x1 != x1_other_diff_coeff

    y1 = BinaryPauliString([1, 1], 1)
    assert x1 != y1
    assert y1 == y1

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

    commuting_parts, suggested_sample_size = H.commuting_groups()

    for part in commuting_parts:
        assert part.is_commuting()

def prepare_cov_dict(H):
    eigenValues, eigenVectors = np.linalg.eigh(H.to_matrix())
    wfn0 = tq.QubitWaveFunction(eigenVectors[:,0])
    terms = BinaryHamiltonian.init_from_qubit_hamiltonian(H).binary_terms
    cov_dict = {}
    for term1 in terms:
        for term2 in terms:
            pw1 = BinaryPauliString(term1.get_binary(), 1.0)
            pw2 = BinaryPauliString(term2.get_binary(), 1.0)
            op1 = QubitHamiltonian.from_paulistrings(pw1.to_pauli_strings())
            op2 = QubitHamiltonian.from_paulistrings(pw2.to_pauli_strings())
            prod_op = op1 * op2
            cov_dict[(term1.binary_tuple(), term2.binary_tuple())] = wfn0.inner(prod_op(wfn0)) - wfn0.inner(op1(wfn0)) * wfn0.inner(op2(wfn0))
    return cov_dict

def test_sorted_insertion():
    '''
    Testing whether sorted insertion works.
    '''
    H, _, _, _ = prepare_test_hamiltonian()
    H = H + paulis.X(0) + paulis.Y(0) + 1.05 * paulis.Z(0) * paulis.Z(1) * paulis.Z(2)
    Hbin = BinaryHamiltonian.init_from_qubit_hamiltonian(H)

    options = {"method":"si", "condition": "qwc", "cov_dict":prepare_cov_dict(H)}
    commuting_parts, suggested_sample_size = Hbin.commuting_groups(options=options)
    for part in commuting_parts:
        assert part.is_qubit_wise_commuting()
    assert np.isclose(np.sum(suggested_sample_size), 1)

    options["condition"] = "fc"
    commuting_parts, suggested_sample_size = Hbin.commuting_groups(options=options)
    for part in commuting_parts:
        assert part.is_commuting()
    assert np.isclose(np.sum(suggested_sample_size), 1)

def prep_o_groups():
    H, _, _, _ = prepare_test_hamiltonian()
    H = H + paulis.X(0) + paulis.Y(0) + 1.05 * paulis.Z(0) * paulis.Z(1) * paulis.Z(2)
    terms = (BinaryHamiltonian.init_from_qubit_hamiltonian(H)).binary_terms
    return OverlappingGroups.init_from_binary_terms(terms)

def test_term_exists_in():
    '''
    Verifies if term_exists_in correctly locates the different positions of Pauli terms.
    '''
    o_groups = prep_o_groups() 
    for term_idx, term in enumerate(o_groups.o_terms):
        for pos in o_groups.term_exists_in[term_idx]:
            assert term in o_groups.o_groups[pos]

def test_variable_coeff_num():
    '''
    Verifies if the number of variable coefficients are correct.
    '''
    o_groups = prep_o_groups() 
    n_coeff_num = np.sum(o_groups.wo_fixed.n_coeff_grp)
    n_coeff_num_2 = 0
    for list_of_term_pos in o_groups.wo_fixed.term_exists_in:
        n_coeff_num_2 += len(list_of_term_pos)
    assert n_coeff_num == n_coeff_num_2

def test_fixed_coeff_num():
    '''
    Verifies if the correct number of variables are fixed.
    '''
    o_groups = prep_o_groups() 
    n_coeff_num = np.sum(o_groups.wo_fixed.n_coeff_grp)
    n_coeff_w_fixed = 0
    for list_of_term_pos in o_groups.term_exists_in:
        n_coeff_w_fixed += len(list_of_term_pos)
    assert n_coeff_num + len(o_groups.o_terms) == n_coeff_w_fixed

def test_overlapping_sorted_insertion():
    '''
    Testing whether overlapping sorted insertion works.
    '''
    H, _, _, _ = prepare_test_hamiltonian()
    H = H + paulis.X(0) + paulis.Y(0) + 1.05 * paulis.Z(0) * paulis.Z(1) * paulis.Z(2)
    Hbin = BinaryHamiltonian.init_from_qubit_hamiltonian(H)
    options = {"method":"ics", "condition": "qwc", "cov_dict":prepare_cov_dict(H)}
    commuting_parts, suggested_sample_size = Hbin.commuting_groups(options=options)
    H_tmp = QubitHamiltonian.zero()
    for part in commuting_parts:
        assert part.is_qubit_wise_commuting()
        H_tmp += part.to_qubit_hamiltonian()
    assert np.isclose(np.sum(suggested_sample_size), 1)
    assert equal_qubit_hamiltonian(H_tmp, H)

    options["condition"] = "fc"
    commuting_parts, suggested_sample_size = Hbin.commuting_groups(options=options)
    H_tmp = QubitHamiltonian.zero()
    for part in commuting_parts:
        assert part.is_commuting()
        H_tmp += part.to_qubit_hamiltonian()
    assert np.isclose(np.sum(suggested_sample_size), 1)
    assert equal_qubit_hamiltonian(H_tmp, H)

    U = tq.gates.ExpPauli(angle="a", paulistring=tq.PauliString.from_string('X(0)Y(1)'))
    variables = {"a": np.random.rand(1) * 2 * np.pi}
    e_ori = tq.ExpectationValue(H=Hbin.to_qubit_hamiltonian(), U=U)
    e_integrated_ics = tq.ExpectationValue(H=Hbin.to_qubit_hamiltonian(), U=U, optimize_measurements=options)
    result_ori = tq.simulate(e_ori, variables)
    compiled = tq.compile(e_integrated_ics)
    evaluated = compiled(variables)
    assert np.isclose(result_ori, evaluated)

test_overlapping_sorted_insertion()

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
    options = {"method":"si", "condition": "fc"}
    e_integrated_si = tq.ExpectationValue(H=H.to_qubit_hamiltonian(), U=U, optimize_measurements=options)
    result_ori = tq.simulate(e_ori, variables)
    result_qwc = tq.simulate(e_qwc, variables)
    result_integrated = tq.simulate(e_integrated, variables)
    result_integrated_si = tq.simulate(e_integrated_si, variables)

    assert np.isclose(result_ori, result_qwc)
    assert np.isclose(result_ori, result_integrated)
    assert np.isclose(result_ori, result_integrated_si)

    # Checking the optimized expectation values are the same
    initial_values = {k: np.random.uniform(0.0, 6.0) for k in e_ori.extract_variables()}
    sol1 = tq.minimize(method='bfgs', objective=e_ori, initial_values=initial_values)
    sol2 = tq.minimize(method='bfgs', objective=e_qwc, initial_values=initial_values)
    sol3 = tq.minimize(method='bfgs', objective=e_integrated, initial_values=initial_values)

    assert np.isclose(sol1.energy, sol2.energy)
    assert np.isclose(sol1.energy, sol3.energy)
