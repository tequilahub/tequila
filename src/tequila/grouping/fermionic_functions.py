import numpy as np
import tequila as tq
import openfermion as of
from openfermion import FermionOperator, QubitOperator, expectation, get_sparse_operator, jordan_wigner, reverse_jordan_wigner, normal_ordered, count_qubits, variance
from itertools import product
import scipy as sp
import tequila.grouping.ev_utils as evu
from functools import partial
import multiprocessing as mp
from tequila import TequilaException

def get_obt_tbt(h_ferm, spin_orb=True):
    '''
    Parameters
     ----------
    mol_name -
    Name of the molecule to obtain one- and two-body integrals, Fermionic Hamiltonian, and number of electrons
    basis -
    Basis set used to compute the integrals.
    spin_orb -
    If true, spin-orbitals is used, if false, spin symmetry is used to reduce the number of orbitals by half (assumes that spin-up and spin-down are identical).
    geometry -
    Geometry of the molecular system.

    Returns
    -------
    (obt, tbt) Tuple of one- and two-body integrals.
    h_ferm Fermionic Hamiltonian of the molecular system.
    num_elecs Number of electrons in the molecular system.
    '''
    no_h_ferm = normal_ordered(h_ferm)
    tbt = get_tbt(no_h_ferm, spin_orb = spin_orb)
    h1b = no_h_ferm - tbt_to_ferm(tbt, spin_orb)
    h1b = normal_ordered(of_simplify(h1b))
    obt = get_obt(h1b, spin_orb=spin_orb)
    return (obt, tbt)

def of_simplify(op):
    '''
    Simplifies fermionic operator by converting to Qubit and back again.
    '''
    return reverse_jordan_wigner(jordan_wigner(op))

def get_spin_orbitals(H : FermionOperator):
    '''
    Obtain the number of spin orbitals of H
    '''
    n = -1
    for term, val in H.terms.items():
        if len(term) == 4:
            n = max([
                n, term[0][0], term[1][0],
                term[2][0], term[3][0]
            ])
        elif len(term) == 2:
            n = max([
                n, term[0][0], term[1][0]])
    n += 1
    return n

def get_tbt(H : FermionOperator, n = None, spin_orb=False):
    '''
    Obtain the 4-rank tensor that represents two body interaction in H.
    In chemist ordering a^ a a^ a.
    In addition, simplify tensor assuming symmetry between alpha/beta coefficients
    '''
    if n is None:
        n = get_spin_orbitals(H)

    phy_tbt = np.zeros((n, n, n, n))
    for term, val in H.terms.items():
        if len(term) == 4:
            phy_tbt[
                term[0][0], term[1][0],
                term[2][0], term[3][0]
            ] = np.real_if_close(val)

    chem_tbt = np.transpose(phy_tbt, [0, 3, 1, 2])
    chem_tbt_sym = (chem_tbt - np.transpose(chem_tbt, [0,3,2,1]) + np.transpose(chem_tbt, [2,3,0,1]) - np.transpose(chem_tbt, [2,1,0,3]) ) / 4.0

    # Spin-orbital to orbital
    n_orb = phy_tbt.shape[0]
    n_orb = n_orb // 2
    alpha_indices = list(range(0, n_orb * 2, 2))
    beta_indices = list(range(1, n_orb * 2, 2))

    chem_tbt_orb = (chem_tbt_sym[np.ix_(alpha_indices, alpha_indices, beta_indices, beta_indices)]
                    - np.transpose(chem_tbt_sym[np.ix_(alpha_indices, beta_indices, beta_indices, alpha_indices)], [0,3,2,1]))
    if spin_orb:
        chem_tbt = np.zeros_like(chem_tbt_sym)
        n = chem_tbt_orb.shape[0]
        for i, j, k, l in product(range(n), repeat=4):
            for a, b in product(range(2), repeat=2):
                chem_tbt[(2*i+a, 1), (2*j+a, 0), (2*k+b, 1), (2*l+b, 0)] = chem_tbt_orb[i,j,k,l]
        return chem_tbt
    else:
        return chem_tbt_orb



def get_obt(H : FermionOperator, n = None, spin_orb=False, tiny=1e-12):
    '''
    Obtain the 2-rank tensor that represents one body interaction in H.
    In addition, simplify tensor assuming symmetry between alpha/beta coefficients
    '''
    # getting N^2 phy_tbt and then (N/2)^2 chem_tbt
    if n is None:
        n = get_spin_orbitals(H)

    obt = np.zeros((n,n))
    for term, val in H.terms.items():
        if len(term) == 2:
            if term[0][1] == 1 and term[1][1] == 0:
                obt[term[0][0], term[1][0]] = np.real_if_close(val)
            elif term[1][1] == 1 and term[0][1] == 0:
                obt[term[1][0], term[0][0]] = -np.real_if_close(val)
            else:
                print("Warning, one-body operator has double creation/annihilation operators!")
                quit()

    if spin_orb:
        return obt

    # Spin-orbital to orbital
    n_orb = obt.shape[0]
    n_orb = n_orb // 2

    obt_red_uu = np.zeros((n_orb, n_orb))
    obt_red_dd = np.zeros((n_orb, n_orb))
    obt_red_ud = np.zeros((n_orb, n_orb))
    obt_red_du = np.zeros((n_orb, n_orb))
    for i in range(n_orb):
        for j in range(n_orb):
            obt_red_uu[i,j] = obt[2*i, 2*j]
            obt_red_dd[i,j] = obt[2*i+1, 2*j+1]
            obt_red_ud = obt[2*i, 2*j+1]
            obt_red_du = obt[2*i+1, 2*j]

    if np.sum(np.abs(obt_red_du)) + np.sum(np.abs(obt_red_ud)) != 0:
        print("Warning, operator to one-body transformation ran with spin_orb=false, but spin-orbit couplings are not 0!")
    if np.sum(np.abs(obt_red_uu - obt_red_dd)) > tiny:
        print("Warning, operator to one-body transformation ran with spin_orb=false, but isn't symmetric to spin-flips")
        print("obt_uu - obt_dd = {}".format(obt_red_uu - obt_red_dd))

    obt = (obt_red_uu + obt_red_dd) / 2

    return obt

def get_cartan_ferm_op(tsr, spin_orb=False):
    '''
    Return the corresponding fermionic operators in Cartan subalgebra
    based on the tensor. This tensor can index over spin-orbtals or orbitals
    '''
    if len(tsr.shape) == 4:
        n = tsr.shape[0]
        op = FermionOperator.zero()
        for i, j in product(range(n), repeat=2):
            if not spin_orb:
                for a, b in product(range(2), repeat=2):
                    op += FermionOperator(
                        term = (
                            (2*i+a, 1), (2*i+a, 0),
                            (2*j+b, 1), (2*j+b, 0)
                        ), coefficient=tsr[i, i, j, j]
                    )
            else:
                op += FermionOperator(
                    term=(
                        (i, 1), (i, 0),
                        (j, 1), (j, 0)
                    ), coefficient=tsr[i, i, j, j]
                )
        return op
    elif len(tsr.shape) == 2:
        n = tsr.shape[0]
        op = FermionOperator.zero()
        for i in range(n):
            if not spin_orb:
                for a in range(2):
                    op += FermionOperator(
                        term = (
                            (2*i+a, 1), (2*i+a, 0)
                        ), coefficient=tsr[i, i]
                    )
            else:
                op += FermionOperator(
                    term = (
                        (i, 1), (i, 0)
                    ), coefficient=tsr[i, i]
                )
        return op

def get_ferm_op(tsr, spin_orb=False):
    '''
    Return the corresponding fermionic operators based on the tensor
    This tensor can index over spin-orbtals or orbitals
    '''
    if len(tsr.shape) == 4:
        n = tsr.shape[0]
        op = FermionOperator.zero()
        for i, j, k, l in product(range(n), repeat=4):
            if not spin_orb:
                for a, b in product(range(2), repeat=2):
                    op += FermionOperator(
                        term = (
                            (2*i+a, 1), (2*j+a, 0),
                            (2*k+b, 1), (2*l+b, 0)
                        ), coefficient=tsr[i, j, k, l]
                    )
            else:
                op += FermionOperator(
                    term=(
                        (i, 1), (j, 0),
                        (k, 1), (l, 0)
                    ), coefficient=tsr[i, j, k, l]
                )
        return op
    elif len(tsr.shape) == 2:
        n = tsr.shape[0]
        op = FermionOperator.zero()
        for i in range(n):
            for j in range(n):
                if not spin_orb:
                    for a in range(2):
                        op += FermionOperator(
                            term = (
                                (2*i+a, 1), (2*j+a, 0)
                            ), coefficient=tsr[i, j]
                        )
                else:
                    op += FermionOperator(
                        term = (
                            (i, 1), (j, 0)
                        ), coefficient=tsr[i, j]
                    )
        return op

def tbt_to_ferm(tbt : np.array, spin_orb, norm_ord = False):
    '''
    Converts two body tensor into Fermion Operator.
    '''
    if norm_ord == True:
        return normal_ordered(get_ferm_op(tbt, spin_orb))
    else:
        return get_ferm_op(tbt, spin_orb)

def cartan_tbt_to_ferm(ctbt : np.array, spin_orb, norm_ord = False):
    '''
    Converts two body tensor into Fermion Operator.
    '''
    if norm_ord == True:
        return normal_ordered(get_cartan_ferm_op(ctbt, spin_orb))
    else:
        return get_cartan_ferm_op(ctbt, spin_orb)

def obt_to_ferm(obt : np.array, spin_orb = False, norm_ord = False):
    '''
    Converts one body tensor into Fermion Operator.
    '''
    if norm_ord == True:
        return of.normal_ordered(get_ferm_op(obt, spin_orb))
    else:
        return get_ferm_op(obt, spin_orb)

def tbt_orb_to_so(tbt):
    '''
    Converts two body term to spin orbitals.
    '''
    n = tbt.shape[0]

    tbt_so = np.zeros([2*n,2*n,2*n,2*n])
    for i1, i2, i3, i4 in product(range(n), repeat=4):
        for a in [0,1]:
            for b in [0,1]:
                tbt_so[2*i1+a,2*i2+a,2*i3+b,2*i4+b] = tbt[i1,i2,i3,i4]
    return tbt_so

def obt_orb_to_so(obt):
    '''
    Converts one body term to spin orbitals.
    '''
    n = obt.shape[0]

    obt_so = np.zeros([2*n,2*n])
    for i1, i2 in product(range(n), repeat=2):
        for a in [0,1]:
            obt_so[2*i1+a,2*i2+a] = obt[i1,i2]

    return obt_so

def convert_u_to_so(U):
    '''
    Converts unitary matrix to spin orbitals
    '''
    nf = U.shape[0]
    n = 2 * U.shape[1]
    U_so = np.zeros([nf, n, n])
    for i in range(nf):
        U_so[i,:,:] = obt_orb_to_so(U[i,:,:])
    return U_so

def convert_tbts_to_frags(tbts, spin_orb = False):
    '''
    Converts two body terms to fermionic fragments.
    '''
    ops = []
    nf = tbts.shape[0]
    for i in range(nf):
        ops.append(tbt_to_ferm(tbts[i,:,:,:,:],spin_orb))
    return ops

def symmetric(M, tol = 1e-8, ret_op = False):
    '''
    if ret_op = False, checks whether a given matrix is symmetric.
    if ret_op = True, returns the symmetrc form of the given matrix.
    '''
    M_res = np.tril(M) + np.triu(M.T, 1)
    if ret_op == False:
        if np.sum(np.abs(M - M_res)) > tol:
            return False
        return True
    else:
        return M_res

def check_tbt_symmetry(tbt):
    '''
    Returns symmetric form of tbt.
    '''
    N = tbt.shape[0] ** 2
    tbt_res = np.reshape(tbt, (N,N))
    if not symmetric(tbt_res):
        print("Non-symmetric two-body tensor as input for SVD routine, calculations might have errors...")
    else:
        tbt_res = symmetric(tbt_res, ret_op = True)
    return tbt_res

def diagonalizing_tbt(tbt_res):
    '''
    Returns diagonal form and unitary rotation to diagonalize tbt_res
    '''
    print("Diagonalizing two-body tensor")
    lamda, U = np.linalg.eig(tbt_res)
    ind = np.argsort(np.abs(lamda))[::-1]
    lamda = lamda[ind]
    U = U[:,ind]
    return lamda, U

def fragmentization(lamda, U, n, tol = 1e-8):
    '''
    Returns fragments of tbt
    '''
    N = n**2
    L_mats = []
    flags = np.zeros(N)

    for i in range(N):
        if abs(lamda[i]) < tol:
            print("Truncating SVD for coefficients with magnitude smaller or equal to {}, using {} fragments".format(abs(lamda[i]), (i)))
            break
        cur_l = np.reshape(U[:, i], (n,n))
        if not symmetric(cur_l):
            flags[i] = 1
        else:
            cur_l = symmetric(cur_l, ret_op = True)
        L_mats.append(cur_l)
    return L_mats, flags


def lr_decomp(tbt : np.array, spin_orb=True, tiny=1e-8):
    '''
    Singular Value Decomposition of the two-body tensor term
    '''
    print("Starting SVD routine")
    n = tbt.shape[0]

    tbt_res = check_tbt_symmetry(tbt)
    lamda, U = diagonalizing_tbt(tbt_res)
    L_mats, flags = fragmentization(lamda, U, n)
    num_ops = len(L_mats)

    TBTS = np.zeros((num_ops, n, n, n, n))
    CARTAN_TBTS = np.zeros(( num_ops, n, n, n, n))
    U_ARR = np.zeros((num_ops, n,n))

    for i in range(num_ops):
        if flags[i] == 0:
            omega_l, U_l = np.linalg.eigh(L_mats[i])

            tbt_svd_CSA = np.zeros((n,n,n,n))

            for j in range(n):
                for k in range(j,n):
                    tbt_svd_CSA[j,j,k,k] = omega_l[j]*omega_l[k]
                    tbt_svd_CSA[k,k,j,j] = tbt_svd_CSA[j,j,k,k]


            tbt_svd_CSA = lamda[i] * tbt_svd_CSA
            tbt_svd = unitary_cartan_rotation_from_matrix(U_l, tbt_svd_CSA)
            TBTS[i,:,:,:,:] = tbt_svd
            CARTAN_TBTS[i,:,:,:,:] = tbt_svd_CSA
        else:
            if np.sum(np.abs(L_mats[i] + L_mats[i].T)) > tiny:
                raise TequilaException("SVD operator {} if neither Hermitian or anti-Hermitian, cannot do double factorization into Hermitian fragment!".format(i))

            temp_l = 1j * L_mats[i]
            cur_l = (temp_l + temp_l.conj().T)/2
            omega_l, U_l = np.linalg.eigh(cur_l)

            tbt_svd_CSA = np.zeros((n,n,n,n))

            for j in range(n):
                for k in range(j,n):
                    tbt_svd_CSA[j,j,k,k] = -omega_l[j]*omega_l[k]
                    tbt_svd_CSA[k,k,j,j] = tbt_svd_CSA[j,j,k,k]

            tbt_svd_CSA = lamda[i] * tbt_svd_CSA
            tbt_svd = unitary_cartan_rotation_from_matrix(U_l, tbt_svd_CSA)
            TBTS[i,:,:,:,:] = tbt_svd
            CARTAN_TBTS[i,:,:,:,:] = tbt_svd_CSA

        U_ARR[i,:,:] = U_l
    print("Finished SVD routine")

    L_ops = []

    for i in range(num_ops):
        op_1d = obt_to_ferm(L_mats[i], spin_orb)
        L_ops.append(lamda[i] * op_1d * op_1d)
    return CARTAN_TBTS, TBTS, L_ops, U_ARR

def unitary_cartan_rotation_from_matrix(Umat, tbt : np.array):
    '''
    Rotates Cartan tbt using orbital rotation matrix U_mat
    '''
    rotated_tbt = np.einsum('al,bl,cm,dm,llmm',Umat,Umat,Umat,Umat,tbt)
    return rotated_tbt

def qubit_number(op):
    '''
    Returns number of qubits in the operator
    '''
    return count_qubits(op)

def get_occ_no(n_elec, n_qubits):
    """
    Given some molecule, find the reference occupation number state.
    Args:
        n_elec (int): Number of electrons in the system.
    Returns:
        occ_no (str): Occupation no. vector.
    """
    occ_no = '1'*n_elec + '0'*(n_qubits - n_elec)

    return occ_no

def n_elec(mol):
    '''
    Given some molecule, find the reference occupation number state.
    Args:
        mol (str): H2, LiH, BeH2, H2O, NH3
    Returns:
        Number of electrons (int)
    '''
    n_electrons = {'h2': 2, 'lih': 4, 'beh2': 6, 'h2o': 10, 'nh3': 10, 'n2': 14, 'hf':10, 'ch4':10, 'co':14, 'h4':4, 'ch2':8, 'heh':2, 'h6':6, 'nh':8, 'h3':2, 'h4sq':4, 'h2ost':10, 'beh2st':6, 'h2ost2':10, 'beh2st2':6, 'li2frco':2, 'beh2frco':4}
    return n_electrons[mol.lower()]

def create_hamiltonian_in_subspace(indices, Hq, n_qubits):
    """
    Given some basis states, create the Hamiltonian within the span of those basis states.
    Args:
        qubit_basis_states(List[array] or List[str]): List of basis vectors to create hamiltonian within
        Hq (QubitOperator): Qubit hamiltonian
        n_qubits (int): Number of qubits.
    Returns:
        H_mat_sub (sp.sparse.coo_matrix): Hamiltonian matrix defined in subspace.
        indices (List[int]): Gives the index in the 2**n dimensional space of the ith qubit_basis_state.
    """
    subspace_dim = len(indices)
    row_idx = []
    col_idx = []
    H_mat_elements = []
    elements_sum = np.zeros((len(indices),len(indices)))
    op_sum = QubitOperator.zero()
    for prog, op in enumerate(Hq):
        op_sum += op
        if (prog + 1)%350 == 0 or prog == len(Hq.terms) - 1:
            opspar = of.get_sparse_operator(op_sum, n_qubits)
            op_sum = of.QubitOperator.zero()
            for iidx, iindx in enumerate(indices):
                for jidx, jindx in enumerate(indices):
                    elements_sum[iidx, jidx] += opspar[iindx, jindx]

    for iidx, iindx in enumerate(indices):
        for jidx, jindx in enumerate(indices):
            row_idx.append(iidx)
            col_idx.append(jidx)
            H_mat_elements.append(elements_sum[iidx, jidx])
    H_mat_sub = sp.sparse.coo_matrix((H_mat_elements, (row_idx, col_idx)), shape = (subspace_dim, subspace_dim))
    return H_mat_sub

def get_jw_cisd_basis_states(n_elec, n_qubits):
    """
    Given some molecule, find the all BK basis vectors that correspond to occupation numbers that are achieved by single and double excitations.
    Args:
        mol (str): H2, LiH, BeH2, H2O, NH3
        n_qubits (int): No. of qubits
    Returns:
        jw_basis_states (List[array]): List of all JW basis states corresponding to occupation numbers achieved by singles and doubles from reference occupation number.
    """
    ref_occ_nos = get_occ_no(n_elec, n_qubits)
    indices = get_jw_cisd_basis_states_wrap(ref_occ_nos, n_qubits)
    return indices

def get_jw_cisd_basis_states_wrap(ref_occ_nos, n_qubits):
    """
    Given some occupation number, find the all other occupation numbers that are achieved by single and double excitations.
    Args:
        ref_occ_nos (str): Reference (likely HF) occupation number ordered from left to right going from 0 -> n-1 in terms of orbitals.
    Returns:
        cisd_basis_states (List[str]): List of all occupation number achieved by singles and doubles from reference occupation number.
    """

    indices = [find_index(get_jw_basis_states(ref_occ_nos))]
    for occidx, occ_orbitals in enumerate(ref_occ_nos):
        if occ_orbitals == '1':
            annihilated_state = list(ref_occ_nos)
            annihilated_state[occidx] = '0'
            #Singles
            for virtidx, virtual_orbs in enumerate(ref_occ_nos):
                if virtual_orbs == '0':
                    new_state = annihilated_state[:]
                    new_state[virtidx] = '1'
                    indices.append(find_index(get_jw_basis_states(''.join(new_state))))
                    #Doubles
                    for occ2idx in range(occidx +1, n_qubits):
                        if ref_occ_nos[occ2idx] == '1':
                            annihilated_state_double = new_state[:]
                            annihilated_state_double[occ2idx] = '0'
                            for virt2idx in range(virtidx +1, n_qubits):
                                if ref_occ_nos[virt2idx] == '0':
                                    new_state_double = annihilated_state_double[:]
                                    new_state_double[virt2idx] = '1'
                                    indices.append(find_index(get_jw_basis_states(''.join(new_state_double))))
    return indices

def find_index(basis_state):
    """
    Given some qubit/fermionic basis state, find the index of the a wavefunction that corresponds to that array.
    Args:
        basis_state (str or list/np.array): Occupation number vector/ Qubit basis state. If str, ordered from left to right going from 0 -> n-1 in terms of orbitals/qubits.
    Returns:
        index (int): Index of the basis in total Qubit space.
    """
    index = 0
    n_qubits = len(basis_state)
    for j in range(n_qubits):
        index += int(basis_state[j])*2**(n_qubits - j - 1)

    return index

def get_jw_basis_states(occ_no_list):
    """
    Implementation from arXiv:quant-ph/0003137 and https://doi.org/10.1021/acs.jctc.8b00450. Given some reference occupation no's in the fermionic space, find the corresponding BK basis state in the qubit space.
    Args:
        occ_no_list (List[str]): List of occupation number vectors. Occ no. vectors ordered from left to right going from 0 -> n-1 in terms of orbitals.
    Returns:
        basis_state (np.array): Basis vector in (JW transformed) qubit space corresponding to occ_no_state.
    """
    jw_list = []
    for occ_no in occ_no_list:
        qubit_state = np.array(list(occ_no), dtype = int)
        jw_list.append(qubit_state)
    return jw_list

def expectation_value(op, psi, n, trunc=False):
    '''
    Calculates expectation of op over psi.
    '''
    opq = jordan_wigner(op)
    if trunc is False:
        e_val = expectation(get_sparse_operator(opq, n_qubits=n), psi)
    else:
        e_val = evu.op_ev_multiple_bases(opq, psi[0], psi[1], n)
    return real_round(e_val)

def variance_value(op, psi, n, trunc=False, neg_tol = 1e-7):
    '''
    Calculates variance of op over psi.
    '''
    opq = jordan_wigner(op)
    if trunc is False:
        var_val = variance(get_sparse_operator(opq, n_qubits=n), psi)
    else:
        var_val = evu.op_ev_multiple_bases(opq * opq, psi[0], psi[1], n) - evu.op_ev_multiple_bases(opq, psi[0], psi[1], n) ** 2
    var_val = real_round(var_val)
    if -neg_tol <= var_val < 0:
        var_val = 0
    if var_val < -neg_tol:
        raise TequilaException('Variance of an observable should not be negative')
    return var_val

def real_round(x, tol=1e-10):
    '''
    Returns real part of complex numbers if the imaginary part is negligible.
    '''
    if abs(x.imag)/abs(x) >= tol:
        print("Warning, rounding x={} to abs(x). Complex component is above tolerance (relative magnitude: {:.2f})".format(x, abs(x.imag)/abs(x)))
        return abs(x)
    return x.real

def get_E(psi, n, n_qubits, trunc=False):
    '''
    Returns the dictionary of one electron excitations over psi
    '''
    gEd = partial(get_E_dummy, psi, n, n_qubits, trunc)
    with mp.Pool(mp.cpu_count()) as pool:
        sq = pool.map(gEd, range(n ** 2))
        pool.close()
        pool.join()
    return np.array(sq)

def get_E_dummy(psi, n, n_qubits, trunc, ind):
    '''
    Returns the expectation value of one electron excitation over psi
    '''
    j = ind % n
    i = ind // n
    op = FermionOperator(((i, 1), (j, 0)), coefficient=1.0)
    return expectation_value(op, psi, n_qubits, trunc)

def get_EE(psi, n, n_qubits, trunc=False):
    '''
    Returns the dictionary value of two electron excitations over psi
    '''
    gEEd = partial(get_EE_dummy, psi, n, n_qubits, trunc)
    with mp.Pool(mp.cpu_count()) as pool:
        sq = pool.map(gEEd, range(n ** 4))
        pool.close()
        pool.join()
    return np.array(sq)

def get_EE_dummy(psi, n, n_qubits, trunc, ind):
    '''
    Returns the expectation value of two electron excitation over psi
    '''
    l = ind % n
    k = ind % n ** 2 // n
    j = ind % n ** 3 // n ** 2
    i = ind // n ** 3
    op = FermionOperator(((i, 1), (j, 0), (k, 1), (l, 0)), coefficient=1.0)
    return expectation_value(op, psi, n_qubits, trunc=trunc)

def reorganize(n, ev_dict_E, ev_dict_EE):
    '''
    -----------
    '''
    ev_dict_ob_ob = np.zeros([n,n,n,n])
    for i, j, k, l in product(range(n), repeat=4):
        ev_dict_ob_ob[i, j, k, l] = ev_dict_EE[(n**3) * (i) + (n**2) * (j) + n * (k) + l] - ev_dict_E[n * (i) + j] * ev_dict_E[n * (k) + l]
    return ev_dict_ob_ob

def compute_covk(op1, op2, psi, n_qubits, trunc=False):
    '''
    ---------------------
    '''
    cko = partial(covk_one, op1, op2, psi, n_qubits, trunc)
    length = len(op1)
    with mp.Pool(mp.cpu_count()) as pool:
        covs = pool.map(cko, range(length))
        pool.close()
        pool.join()
    return np.array(covs)

def covk_one(op1, op2, psi, n_qubits, trunc, ind):
    '''
    ----------------------
    '''
    cov = covariance(op1[ind], op2, psi, n_qubits, trunc) + covariance(op2, op1[ind], psi, n_qubits, trunc)
    return cov

def covariance(op1, op2, psi, n_qubits, trunc):
    '''
    Returns covariance between op1 and op2
    '''
    op1q = of.jordan_wigner(op1)
    op2q = of.jordan_wigner(op2)

    sp_op1 = get_sparse_operator(op1q, n_qubits)
    sp_op2 = get_sparse_operator(op2q, n_qubits)
    if trunc is False:
        psi_tmp = sp_op2.dot(psi)
        cross = np.dot(np.conjugate(psi.T), sp_op1.dot(psi_tmp))
        exp_op1 = np.dot(np.conjugate(psi.T), sp_op1.dot(psi))
        exp_op2 = np.dot(np.conjugate(psi.T), sp_op2.dot(psi))
    else:
        cross = evu.op_ev_multiple_bases(op1q * op2q, psi[0], psi[1], n_qubits)
        exp_op1 = evu.op_ev_multiple_bases(op1q, psi[0], psi[1], n_qubits)
        exp_op2 = evu.op_ev_multiple_bases(op2q, psi[0], psi[1], n_qubits)
    return cross - exp_op1 * exp_op2

def covariance_ob_ob(obt1, obt2, ev_dict_ob_ob):
    '''

    '''
    my_cov = 0.0 + 0.0j
    my_cov = np.einsum('ij,kl,ijkl',obt1, obt2, ev_dict_ob_ob)
    return my_cov

def fff_1_iter(obt, tbts, var, c0, ck, fff_var):
    '''
    Computes a single iteration of FFF optimization (arXiv:2208.14490v3 - Section 2.2)
    '''
    m0 = compute_meas_alloc(var, obt, tbts, fff_var.nq, fff_var.mix)
    lambda_opt = compute_lambda_optimum(fff_var.coo, c0, ck, m0, fff_var.nf, fff_var.n)
    new_obt, new_tbts = modify_ops(obt, tbts, lambda_opt, fff_var.o_t, fff_var.nf, fff_var.n, fff_var.uops)
    new_var = modify_var(var, fff_var.coo, c0, ck, lambda_opt, fff_var.n)
    new_c0, new_ck = modify_c(fff_var.coo, c0, ck, lambda_opt, fff_var.nf, fff_var.n)
    return new_obt, new_tbts, new_var, new_c0, new_ck

def compute_lambda_optimum(Coo, C0, Ck, m0, nf, n):
    '''
    Computes the optimum value of lamda
    '''
    symCoo = np.zeros_like(Coo)
    diagCoo = np.zeros_like(Coo)
    mat = np.zeros_like(Coo)
    vec = np.zeros_like(C0)
    nall = Coo.shape[0]
    for i in range(nall):
        for j in range(i, nall):
            symCoo[i,j] = Coo[i,j] + Coo[j,i]
            symCoo[j,i] = symCoo[i,j]
    for k in range(nf):
        for p in range(n):
            for q in range(n):
                ind1 = n * k + p
                ind2 = n * k + q
                diagCoo[ind1, ind2] = symCoo[ind1, ind2]/m0[k + 1]
    mat = (1/m0[0]) * symCoo + diagCoo
    for k in range(nf):
        for p in range(n):
            ind = n * k + p
            vec[ind] = Ck[ind]/m0[k + 1]
    vec = vec - (1/m0[0]) * C0
    lam = solve_Axb(mat, vec)
    return lam

def var_avg(n_qubits):
    """ Haar average variance of a Pauli product.
    """
    return 1. - 1. / ( 2 ** n_qubits  + 1 )

def get_avg_variances(ops, n_qubits):
    """ Compute Haar average variances of ops.
    """
    return np.array(list(map(lambda x: get_avg_variance(x, n_qubits), ops)))

def get_avg_variance(op, n_qubits):
    """ Compute Haar average variance of op.
    """
    pws = get_pauliword_list(jordan_wigner(op))
    c_sq = np.sum(list(map(lambda x: np.abs(evu.get_pauli_word_coefficient(x)) ** 2, pws)))
    return c_sq * var_avg(n_qubits)

def get_pauliword_list(H: QubitOperator):
    """Obtain a list of pauli words in H.
    """
    pws = []
    for pw, val in H.terms.items():
        if len(pw) != 0: pws.append(QubitOperator(term=pw, coefficient=val))
    return pws


def solve_Axb(A, b):
    '''
    Provides asolution frox x, Ax = b.
    '''
    b_tmp = np.zeros((len(b), 1))
    b_tmp[:,0] = b
    sol = np.linalg.lstsq(A, b_tmp, rcond=None)
    x0 = sol[0]
    return x0.T[0]

def modify_ops(obt, tbts, lambda_opt, O_t, nf, n, uops):
    '''
    Computes new values of obt and tbt after a single FFF iteration
    '''
    ntmp = obt.shape[0]
    new_tbts = np.zeros([nf, ntmp, ntmp, ntmp, ntmp])
    sig_o = np.zeros([ntmp, ntmp])
    for i in range(nf):
        obt_tmp = np.zeros([ntmp, ntmp])
        for j in range(n):
            ind = n * i + j
            obt_tmp += lambda_opt[ind] * O_t[i, j, :, :]
        sig_o += obt_tmp
        new_tbts[i,:,:,:,:] = tbts[i,:,:,:,:] - my_obt_to_tbt(obt_tmp,uops[i,:,:])
    new_obt = obt + sig_o
    return new_obt, new_tbts

def my_obt_to_tbt(obt, uop):
    '''
    Converts the repartioned one body term into two body terms.
    '''
    nsize = obt.shape[0]
    rot_obt = np.zeros([nsize, nsize])
    tbt_from_obt = np.zeros([nsize, nsize, nsize, nsize])
    rot_obt = np.einsum('pa,kb,pk',np.conjugate(uop),uop,obt)
    if np.sum(np.abs(np.diag(np.diag(rot_obt)) - rot_obt))  > 1e-5:
        print("Warning:", np.sum(np.abs(np.diag(np.diag(rot_obt)) - rot_obt)))
    tbt_from_obt = np.einsum('al,bl,cl,dl,ll',uop, np.conjugate(uop), uop, np.conjugate(uop), rot_obt)
    return tbt_from_obt

def modify_var(var, coo, c0, ck, lam, n, tol=1e-10):
    '''
    Computes the variances of the repartitioned fragments
    '''
    nvar = len(var)
    new_var = np.zeros(len(var))
    delta_var1 = 0.0
    for l1 in range(len(lam)):
        for l2 in range(len(lam)):
            delta_var1 += lam[l1] * lam[l2] * coo[l1,l2]
        delta_var1 += lam[l1] * c0[l1]
    new_var[0] = var[0] + delta_var1
    if new_var[0] < tol: new_var[0] = 0.
    for k in range(nvar-1):
        delta_var_k = 0.0
        for p in range(n):
            ind1 = n * k + p
            for q in range(n):
                ind2 = n * k + q
                delta_var_k += lam[ind1] * lam[ind2] * coo[ind1, ind2]
            delta_var_k -= lam[ind1] * ck[ind1]
        new_var[k+1] = var[k+1] + delta_var_k
        if new_var[k+1] < tol: new_var[k+1] = 0.
    return new_var

def modify_c(coo, c0, ck, lam, nf, n):
    '''
    Computes the covariances of the repartitioned fragments
    '''
    new_c0 = np.copy(c0)
    new_ck = np.copy(ck)
    for ind1 in range(len(lam)):
        for ind2 in range(len(lam)):
            new_c0[ind1] += coo[ind1, ind2] * lam[ind2] + coo[ind2, ind1] * lam[ind2]
    for k in range(nf):
        for p in range(n):
            ind1 = n * k + p
            for q in range(n):
                ind2 = n * k + q
                new_ck[ind1] -= coo[ind1, ind2] * lam[ind2] + coo[ind2, ind1] * lam[ind2]
    return new_c0, new_ck

def compute_meas_alloc(varbs, obt=None, tbts=None, n_qubits=None, mix=0.0):
    '''
    Computes the measurement allocations based on the variances of repartitioned fragments.
    '''
    if mix > 1e-6:
        all_ops = [obt_to_ferm(obt, True)]
        ops = convert_tbts_to_frags(tbts, True)
        for i in range(len(ops)):
            all_ops.append(ops[i])
        avg_vars = get_avg_variances(all_ops, n_qubits)
        vtmp = mix * avg_vars + (1-mix) * varbs
    else:
        vtmp = varbs
    sqrt_vars = np.sqrt(vtmp)
    meas_alloc = sqrt_vars/np.sum(sqrt_vars)
    for i in range(len(meas_alloc)):
        if meas_alloc[i] < 1e-6:
            meas_alloc[i] = 1e-6
    return np.real( meas_alloc/np.sum(meas_alloc))

def depth_eff_order_mf(N):
    '''
    Returns index ordering for linear depth circuit

    For example N = 6 gives elimination order
    [ 0.  0.  0.  0.  0.  0.]
    [ 7.  0.  0.  0.  0.  0.]
    [ 5. 10.  0.  0.  0.  0.]
    [ 3.  8. 12.  0.  0.  0.]
    [ 2.  6. 11. 14.  0.  0.]
    [ 1.  4.  9. 13. 15.  0.]
    '''
    l = []
    for c in range(0, N-1):
        for r in range(1, N):
            if r - c > 0:
                l.append([r, c, 2*c - r + N])
    l.sort(key=lambda x: x[2])
    return [(a[0], a[1]) for a in l]

def get_orb_rot(U, qubit_list = [], method = 'short', tol = 1e-12):
    '''
    Construct sequence of orbital rotations that implement mean-field unitary given by NxN unitary U
    Currently supported only for real U
    '''
    
    N = len(U)
    C = tq.QCircuit()
    
    if qubit_list == []:
        qubit_list = list(range(N))
    
    assert len(qubit_list) >= len(U), 'Insufficient qubits for orbital rotation' #check if sufficient qubits
    
    U[abs(U) < tol] = 0

    if method == 'naive':
        theta_list, phi_list = given_rotation(U, tol)
    elif method == 'short':
        ordering = depth_eff_order_mf(N)
        theta_list, phi_list = given_rotation(U, tol, ordering)
    
    #filter
    theta_list_new = []
    for i, theta in enumerate(theta_list):
        if abs(theta[0] % (2*np.pi)) > tol:
            theta_list_new.append(theta)
    
    phi_list_new = []
    for i, phi in enumerate(phi_list):
        if abs(phi[0]) > tol:
            phi_list_new.append(phi)
    
    for phi in phi_list_new:
        C += n_rotation(qubit_list[phi[1]], phi[0])
    
    gates = []
    for theta in theta_list_new:
        gates.append(orbital_rotation(qubit_list[theta[1]], qubit_list[theta[2]], -theta[0]))
    gates.reverse()

    for gate in gates:
        C += gate
    return C

def orbital_rotation(i, j, theta):
    '''
    Implements exp(theta(a^_i a_j - a^_j a_i))
    Right now restricted to |i-j| <= 1 and jordan wigner transform.
    '''
    if abs(i-j) <= 1:
        return tq.gates.CNOT(control=i, target=j) + tq.gates.Ry(angle=2*theta, target=i, control=j) + tq.gates.CNOT(control=i, target=j)

def n_rotation(i, phi):
    return tq.gates.Rz(angle = phi, target=i)

def given_rotation(U, tol = 1e-12, ordering = None):
    '''
    Decomposes the Unitary into a set of Rz by angle phi and Givens Rotations by angle theta.
    Input:
    U (np.array): Rotation matrix
    tol: tolerance for U elements
    '''
    
    U[abs(U) < tol] = 0
    n = U.shape[0]

    theta = []
    phi = []
    if ordering is None:
        for c in range(n):
            for r in range(n-1, c, -1):
                t = np.arctan2(-U[r,c], U[r-1,c])
                theta.append((t, r, r-1))
                
                g = givens_matrix(n,r,r-1,t)
                U = np.dot(g, U)
    else:
        for r, c in ordering:
            t = np.arctan2(-U[r,c], U[r-1,c])
            theta.append((t, r, r-1))
            
            g = givens_matrix(n,r,r-1,t)
            U = np.dot(g, U)
    
    for i in range(n):
        ph = np.angle(U[i,i])
        phi.append((ph, i))
        
    return theta, phi

def givens_matrix(n, p, q, theta): #verified
    '''
    Returns the n dimension givens rotation matrix by theta between rows p and q.
    '''
    g = np.eye(n)
    g[p,p] = np.cos(theta)
    g[q,q] = np.cos(theta)
    g[p,q] = np.sin(theta)
    g[q,p] = - np.sin(theta)
    return g
