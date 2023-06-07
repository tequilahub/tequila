from openfermion import QubitOperator
import numpy as np
from scipy.sparse import csc_matrix
from itertools import combinations
from tequila import TequilaException

def get_pauli_word_tuple(P: QubitOperator):
    """Given a single pauli word P, extract the tuple representing the word.
    """
    words = list(P.terms.keys())
    if len(words) != 1:
        raise TequilaException("P given is not a single pauli word")
    return words[0]


def get_pauli_word(P: QubitOperator):
    """Given a single pauli word P, extract the same word with coefficient 1.
    """
    words = list(P.terms.keys())
    if len(words) != 1:
        raise TequilaException("P given is not a single pauli word")
    return QubitOperator(words[0])


def get_pauli_word_coefficient(P: QubitOperator):
    """Given a single pauli word P, extract its coefficient.
    """
    coeffs = list(P.terms.values())
    return coeffs[0]

def get_occ_no(mol, n_qubits):
    """
    Given some molecule, find the reference occupation number state.
    Args:
        mol (str): H2, LiH, BeH2, H2O, NH3
    Returns:
        occ_no (str): Occupation no. vector.
    """
    n_electrons = {'h2': 2, 'lih': 4, 'beh2': 6, 'h2o': 10, 'nh3': 10, 'n2': 14, 'hf':10, 'ch4':10, 'co':14, 'h4':4, 'ch2':8, 'heh':2, 'h6':6, 'nh':8, 'h3':2}
    occ_no = '1'*n_electrons[mol.lower()] + '0'*(n_qubits - n_electrons[mol.lower()])

    return occ_no

def partial_order(x, y):
    """
    As described in arXiv:quant-ph/0003137 pg.10, computes the if x <= y where <= is a partial order and x and y are binary strings (but inputted as integers).
    Args:
        x, y (int): Integers that will be converted to binary to then check x <= y.

    Returns:
        partial_order(bool): Whether x <= y

    """
    if x > y:
        return False

    else:
        x_b, y_b = format(x, 'b'), format(y, 'b')

        if len(x_b) != len(y_b):
            while len(x_b) != len(y_b):
                x_b = '0' + x_b

        length = len(x_b)

        partial_order = False
        for l0 in range(length):
            if x_b[0:l0] == y_b[0:l0] and y_b[l0:length] == (length - l0)*'1':
                partial_order = True
                break

        return partial_order

def allz(pw: QubitOperator):
    """
    Checks if a Pauli word is all-z.
    Args:
        pw (QubitOperator): Single Pauli word as Qubit Operator
    Returns:
        True/False (Bool): Whether PW is all-z
    """
    for pwt in get_pauli_word_tuple(pw):
        if pwt[1] != 'Z':
            return False

    return True

def isz(pw, qubit_no):
    """
    Checks if a Pauli word is z on the qubit_no'th qubit numner.
    Args:
        pw (QubitOperator): Single Pauli word as Qubit Operator
    Returns:
        True/False (Bool): Whether PW is z on said qubit_no
    """
    for pwt in get_pauli_word_tuple(pw):
        if pwt[0] == qubit_no and pwt[1] == 'Z':
            return True
    return False

def pw_ev_single_basis_state(pw: QubitOperator, index):
    """
    Returns the EV of a single PW w.r.t. a computational basis state
    Args:
        pw (QubitOperator): Single Pauli word as Qubit Operator
        index (ndarray): Computational basis state in {0,1}^n_qubits
        n_qubits (int): No. of qubits
    Returns:
        ev (float): Expectation value of PW
    """

    if allz(pw):
        ev = get_pauli_word_coefficient(pw)
        for idx, i in enumerate(index):
            if i == 1 and isz(pw, idx):
                ev *= -1
        return ev
    else:
        return 0

def pw_matrix_element(pw: QubitOperator, l_index, r_index, n_qubits = None):
    """
    Returns the matrix element of a single PW = <l_index|pw|r_index>
    Args:
        pw (QubitOperator): Single Pauli word as Qubit Operator
        l_index (ndarray): Computational basis state in {0,1}^n_qubits
        r_index (ndarray): Computational basis state in {0,1}^n_qubits
        n_qubits (int): No. of qubits
    Returns:
        me (float): Matrix element of PW
    """

    def find_pw_basis(pw, n_qubits):
        basis = ['I'] * n_qubits
        for pwt in get_pauli_word_tuple(pw):
            basis[pwt[0]] = pwt[1]
        return basis

    if allz(pw):
        return 0
    else:
        me = get_pauli_word_coefficient(pw)
        basis = find_pw_basis(pw, n_qubits)

        for i in range(n_qubits):
            if l_index[i] == r_index[i]:
                if basis[i] == 'Z':
                    if l_index[i] == 1: me *= -1
                elif basis[i] != 'I':
                    return 0.
            else:
                if basis[i] == 'Y':
                    if l_index[i] == 0:
                        me *= -1j
                    else:
                        me *= 1j
                elif basis[i] != 'X':
                    return 0.
        return me

def op_matrix_element(op: QubitOperator, l_index, r_index, n_qubits = None):
    """
    Returns the matrix element of a single PW = <l_index|pw|r_index>
    Args:
        pw (QubitOperator): Single Pauli word as Qubit Operator
        index (ndarray): Computational basis state in {0,1}^n_qubits
        n_qubits (int): No. of qubits
    Returns:
        me (float): Matrix element of PW
    """
    return np.sum(list(map(lambda x: pw_matrix_element(QubitOperator(term=x[0], coefficient=x[1]), l_index, r_index, n_qubits), op.terms.items())))

def truncate_wavefunction(wfs, perc = None, n_qubits = None, tol = 1e-5):
    """
    Makes config_dict from wfs. Include the most significant slater determinants (up to perc % coeffs).
    If perc is None, return the entire wavefunction.
    Returns a config_dict {Computational basis state: coefficient}
    """

    configs = []
    coeffs = []
    if perc is None:
        for idx, coeff in enumerate(wfs):
            if np.abs(coeff) > tol:
                config = format(idx, 'b')
                if len(config) < n_qubits:
                    config = '0'*(n_qubits - len(config)) + config
                configs.append(np.array(list(config),dtype=int))
                coeffs.append(coeff)

    elif perc > 0 and perc < 100:
        wfs_rd = np.copy(wfs)
        wfs_rd_nz_idx = np.nonzero(wfs_rd)[0]
        wfs_rd_nz = wfs_rd[wfs_rd_nz_idx]
        abs_wfs = np.abs(wfs_rd_nz)
        srt_idx = np.argsort(-abs_wfs)
        weight = 0.
        for i, idx in enumerate(srt_idx):
            weight = np.sqrt(weight ** 2 + abs_wfs[idx] ** 2)
            if weight >= (perc * 0.01):
                end = i
                break

        min_idx = srt_idx[:end+1]
        wfs_out = wfs_rd_nz[min_idx]
        wfs_idx_out = wfs_rd_nz_idx[min_idx]

        wfs_out = wfs_out/np.linalg.norm(wfs_out) #Renormalisation

        for idx, coeff in enumerate(wfs_out):
            if np.abs(coeff) > tol:
                config = format(wfs_idx_out[idx], 'b')
                if len(config) < n_qubits:
                    config = '0'*(n_qubits - len(config)) + config
                configs.append(np.array(list(config),dtype=int))
                coeffs.append(coeff)
    else:
        raise TequilaException("The number of truncated slater determinant terms must be between 1 and 2^N_q")
    print("Using {} slater determinants: {}% of the approximate WF".format(len(configs), weight))
    return configs, np.array(coeffs)


def build_multiple_bases_mat(op, configs, n_qubits):
    """
    Returns the sparse matrix representation of op.
    """
    data = np.array(list(map(lambda x: qubit_op_ev_single_basis_state(op, x), configs)))
    nz_loc = np.where(np.abs(data) > 1e-8)[0]
    col = row = nz_loc
    data = data[nz_loc]
    if len(configs) > 1:
        data_pair = np.array(list(map(lambda x: op_matrix_element(op, x[0], x[1], n_qubits), combinations(configs,2))))
        idx_pairs = np.array(list(combinations(range(len(configs)), 2)))
        nz_loc = np.where(np.abs(data_pair) > 1e-8)[0]
        col = np.concatenate( (col, idx_pairs[nz_loc, 0], idx_pairs[nz_loc, 1]) )
        row = np.concatenate( (row, idx_pairs[nz_loc, 1], idx_pairs[nz_loc, 0]) )
        data = np.concatenate( (data, data_pair[nz_loc], np.conjugate(data_pair[nz_loc])) )
    return csc_matrix((data, (row, col)), shape=(len(configs), len(configs)))

def op_ev_multiple_bases(op: QubitOperator, configs, coeffs, n_qubits):
    """
    Returns the EV of operator w.r.t. config_dict.
    """
    mat = build_multiple_bases_mat(op, configs, n_qubits)
    ev = np.dot( np.conjugate(coeffs), mat * coeffs )
    return ev

def qubit_op_ev_single_basis_state(op: QubitOperator, index):
    """
    Returns the EV of an arbitrary QubitOperator w.r.t. a computational basis state
    Args:
        op (QubitOperator): Qubit Operator
        index (ndarray): Computational basis state in {0,1}^n_qubits
        n_qubits (int): No. of qubits
    Returns:
        ev (float): Expectation value of Operator
    """
    return np.sum(list(map(lambda x: pw_ev_single_basis_state(QubitOperator(term=x[0], coefficient=x[1]), index), op.terms.items())))
