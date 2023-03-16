import numpy as np
import math
import cmath
import os
import gc
from os.path import exists
import tequila as tq
import openfermion as of
from openfermion import FermionOperator, QubitOperator, MolecularData, expectation, get_sparse_operator
from openfermionpyscf import run_pyscf
from openfermion.transforms import get_fermion_operator
from scipy.sparse import save_npz, load_npz, csc_matrix
import scipy as sp
import itertools
import multiprocessing as mp
import pickle
from functools import partial


def obtain_SD(mol_name, basis="sto3g", fermion=True, spin_orb=True, geometry=1):
    '''
    Gets the one body and two body terms of the hamilatonian
    '''
    h_ferm, num_elecs = get_system(mol_name, ferm=fermion, basis=basis, geometry=geometry, n_elec=True)
    tbt = get_tbt(h_ferm, spin_orb = spin_orb)
    h1b = h_ferm - tbt_to_ferm(tbt, spin_orb)
    h1b = of_simplify(h1b)
    obt = get_obt(h1b, spin_orb=spin_orb)
    return (obt, tbt), h_ferm, num_elecs

def get_system(mol_name, ferm = False, basis='sto3g', geometry=1):
    '''
    Obtain system from specified parameters
    '''
    g, c = chooseType(mol_name, geometry)
    mol = MolecularData(g, basis, 1, c)
    mol = run_pyscf(mol)
    ham = mol.get_molecular_hamiltonian()
    if ferm:
        return get_fermion_operator(ham), mol.n_electrons
    else:
        return ham, mol.n_electrons
    
def chooseType(typeHam, geometries):
    '''
    Genreate the molecular data of specified type of Hamiltonian
    '''
    charge = 0
    if typeHam == 'h2':
        molData = [
            ['H', [0, 0, 0]],
            ['H', [0, 0, geometries]]
        ]
    elif typeHam == 'h3':
        molData = [
            ['H', [0, 0, 0]],
            ['H', [0, 0, geometries]],
            ['H', [0, 0, 2*geometries]]
        ]
        charge = 1
    elif typeHam == 'n2':
        molData = [
            ['N', [0, 0, 0]],
            ['N', [0, 0, geometries]]
        ]
    elif typeHam == 'lih':
        molData = [
            ['Li', [0, 0, 0]],
            ['H', [0, 0, geometries]]
        ]
    # Giving symmetrically stretch H2O. ∠HOH = 107.6°
    elif typeHam == 'h2o':
        angle = 107.6 / 2
        angle = math.radians(angle)
        xDistance = geometries * math.sin(angle)
        yDistance = geometries * math.cos(angle)
        molData = [
            ['O', [0, 0, 0]],
            ['H', [-xDistance, yDistance, 0]],
            ['H', [xDistance, yDistance, 0]]
        ]
    elif typeHam == 'hf':
        molData = [
            ['H', [0, 0, 0]],
            ['F', [0, 0, geometries]]
        ]
    elif typeHam == 'co':
        molData = [
            ['C', [0, 0, 0]],
            ['O', [0, 0, geometries]]
        ]
    elif typeHam == 'beh2':
        molData = [
            ['Be', [0, 0, 0]],
            ['H', [0, 0, -geometries]],
            ['H', [0, 0, geometries]]
        ]
    elif typeHam == 'h4':
        molData = [
            ['H', [0, 0, 0]],
            ['H', [0, 0, geometries]],
            ['H', [0, 0, 2*geometries]],
            ['H', [0, 0, 3*geometries]]
        ]
    elif typeHam == 'h6':
        molData = [
            ['H', [0, 0, 0]],
            ['H', [0, 0, geometries]],
            ['H', [0, 0, 2*geometries]],
            ['H', [0, 0, 3*geometries]],
            ['H', [0, 0, 4*geometries]],
            ['H', [0, 0, 5*geometries]]
        ]
    elif typeHam == 'heh':
        molData = [
            ['He', [0, 0, 0]],
            ['H', [0, 0, geometries]]
        ]
        charge = 1
    elif typeHam == 'ch2':
        angle = 101.89 / 2
        angle = math.radians(angle)
        xDistance = 1.0 * math.sin(angle)
        yDistance = 1.0 * math.cos(angle)
        molData = [
            ['C', [0, 0, 0]],
            ['H', [-xDistance, yDistance, 0]],
            ['H', [xDistance, yDistance, 0]]
        ]
    elif typeHam == 'nh3':
    # Is there a more direct way of making three vectors with specific mutual angle?
        bondAngle = 107
        bondAngle = math.radians(bondAngle)
        cos = math.cos(bondAngle)
        sin = math.sin(bondAngle)

        # The idea is second and third vecctor dot product is cos(angle) * geometry^2. 
        thirdyRatio = (cos - cos**2) / sin
        thirdxRatio = (1 - cos**2 - thirdyRatio**2) ** (1/2)
        molData = [
            ['H', [0, 0, geometries]],
            ['H', [0, sin * geometries, cos * geometries]], 
            ['H', [thirdxRatio * geometries, thirdyRatio * geometries, cos * geometries]], 
            ['N', [0, 0, 0]], 
        ]
    elif typeHam == 'ch4':
        l_edge = (2./3.) * math.sqrt(6.) * geometries #length of the edge of the tetrahedron.
        h_tet = (4./3.) * geometries
        zdisp = -(1./3.) * geometries
        molData = [
            ['H', [0, 0, geometries]],
            ['H', [- (math.sqrt(3.)/3.) * l_edge, 0, zdisp]],
            ['H', [(math.sqrt(3.) / 6.) * l_edge, l_edge / 2.0, zdisp]], 
            ['H', [(math.sqrt(3.) / 6.) * l_edge, -l_edge / 2.0, zdisp]], 
            ['C', [0, 0, 0]], 
        ]
        
    else:
        raise(ValueError(typeHam, 'Unknown type of hamiltonian given'))

    return molData, charge

def of_simplify(op):
    '''
    Simplifies fermionic operator by converting to Qubit and back again.
    '''
    return of.reverse_jordan_wigner(of.jordan_wigner(op))

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
            ] = val

    chem_tbt = np.transpose(phy_tbt, [0, 3, 1, 2])

    if spin_orb:
        return chem_tbt

    # Spin-orbital to orbital 
    n_orb = phy_tbt.shape[0]
    n_orb = n_orb // 2
    alpha_indices = list(range(0, n_orb * 2, 2))
    beta_indices = list(range(1, n_orb * 2, 2))
    chem_tbt = chem_tbt[
        np.ix_(alpha_indices, alpha_indices,
                    beta_indices, beta_indices)]

    return chem_tbt

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

def get_ferm_op(tsr, spin_orb=False):
    '''
    Return the corresponding fermionic operators based on the tensor
    This tensor can index over spin-orbtals or orbitals
    '''
    if len(tsr.shape) == 4:
        n = tsr.shape[0]
        op = FermionOperator.zero()
        for i, j, k, l in itertools.product(range(n), repeat=4):
            if not spin_orb:
                for a, b in itertools.product(range(2), repeat=2):
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
        return of.normal_ordered(get_ferm_op(tbt, spin_orb))
    else:
        return get_ferm_op(tbt, spin_orb)
    
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
    n_qubit = 2*n

    tbt_so = np.zeros([2*n,2*n,2*n,2*n])
    for i1 in range(n):
        for i2 in range(n):
            for i3 in range(n):
                for i4 in range(n):
                    for a in [0,1]:
                        for b in [0,1]:
                            tbt_so[2*i1+a,2*i2+a,2*i3+b,2*i4+b] = tbt[i1,i2,i3,i4]
    return tbt_so

def obt_orb_to_so(obt):
    '''
    Converts one body term to spin orbitals.
    '''
    n = obt.shape[0]
    n_qubit = 2*n

    obt_so = np.zeros([2*n,2*n])
    for i1 in range(n):
        for i2 in range(n):
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
                                        
                                          
def lr_decomp(tbt : np.array, tol=1e-6, spin_orb=True, tiny=1e-8):
    '''
     Singular Value Decomposition of the two-body tensor term
    '''
    print("Starting SVD routine")
    n = tbt.shape[0]
    N = n**2
    
    tbt_res = np.reshape(tbt, (N,N))
    if not symmetric(tbt_res):
        print("Non-symmetric two-body tensor as input for SVD routine, calculations might have errors...")
    else:
        tbt_res = symmetric(tbt_res, ret_op = True)

    print("Diagonalizing two-body tensor")
    lamda, U = np.linalg.eig(tbt_res)
    ind = np.argsort(np.abs(lamda))[::-1]
    lamda = lamda[ind]
    U = U[:,ind]

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
                error("SVD operator {} if neither Hermitian or anti-Hermitian, cannot do double factorization into Hermitian fragment!".format(i))

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
    n = tbt.shape[0]
    rotated_tbt = np.einsum('al,bl,cm,dm,llmm',Umat,Umat,Umat,Umat,tbt)
    return rotated_tbt

def qubit_number(op):
    '''
    Returns number of qubits in the operator
    '''
    return of.count_qubits(op)



