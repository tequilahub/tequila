import numpy as np
from pathlib import Path
import os
import gc
from os.path import exists
import tequila as tq
import openfermion as of
from openfermion import FermionOperator, QubitOperator
from scipy.sparse import save_npz, load_npz, csc_matrix
import itertools
import multiprocessing as mp
import pickle
import tequila.grouping.fermionic_functions as ferm
import h5py

def get_init_ops(mol_name, calc_type, spin_orb):
    '''
    Gets the fermionic fragments of the given molecular hamiltonian.
    '''
    path = "SAVE/" + mol_name + "/"
    Path(path).mkdir(exist_ok=True)
    if os.path.isfile(path + "tensor_terms.pkl"):
        with open(path + "tensor_terms.pkl", 'rb') as file:
            INIT = pickle.load(file)
        obt = INIT[0]
        tbt_ham_opt = INIT[1]
        with open(path + "ham.pkl", 'rb') as file:
            h_ferm = pickle.load(file)
    else:
        obtb, h_ferm, num_elecs = ferm.obtain_SD(mol_name, n_elec=True, spin_orb = spin_orb)
        obt = obtb[0]
        tbt_ham_opt = obtb[1]
        with open(path + "tensor_terms.pkl", 'wb') as file:
            pickle.dump([obt, tbt_ham_opt], file)
        with open(path + "ham.pkl", 'wb') as file:
            pickle.dump(h_ferm, file)
    n_qubit = ferm.qubit_number(h_ferm)
    if spin_orb != True:
        obt = ferm.obt_orb_to_so(obt)
        tbt = ferm.tbt_orb_to_so(tbt_ham_opt)
    else:
        tbt = tbt_ham_opt
    n = obt.shape[0]
    
    if calc_type == "lr":
        if os.path.isfile(path + "lr.pkl"):
            with open(path + "lr.pkl", 'rb') as f:
                INIT = pickle.load(file)
            CARTAN_TBTS_tmp = INIT[0]
            TBTS_tmp = INIT[1]
            U_OPS_tmp = INIT[2]                
        else:
            CARTAN_TBTS_tmp, TBTS_tmp, OPS_tmp, U_OPS_tmp = ferm.lr_decomp(tbt_ham_opt, tol=1e-8, spin_orb=spin_orb)
            with open(path + "lr.pkl", 'wb') as file:
                pickle.dump([CARTAN_TBTS_tmp, TBTS_tmp, U_OPS_tmp], file)
        OPS = ferm.convert_tbts_to_frags(TBTS_tmp, spin_orb)
    
    if spin_orb != True:
        U_OPS = ferm.convert_u_to_so(U_OPS_tmp)
        tbts = np.zeros([len(OPS), n, n, n, n])
        cartan_tbts = np.zeros([len(OPS), n, n, n, n])
        for i in range(len(OPS)):
            tbts[i,:,:,:,:] = ferm.tbt_orb_to_so(TBTS_tmp[i,:,:,:,:])
            cartan_tbts[i, :, :, :, :] = ferm.tbt_orb_to_so(CARTAN_TBTS_tmp[i, :, :, :, :])

    else:
        U_OPS = U_OPS_tmp
        tbts = TBTS_tmp
        cartan_tbts = CARTAN_TBTS_tmp

    all_OPS = [ferm.obt_to_ferm(obt, True)]
    for i in range(len(OPS)):
        all_OPS.append(OPS[i])
    return h_ferm, obt, tbt, n_qubit, all_OPS, U_OPS, tbts, cartan_tbts
