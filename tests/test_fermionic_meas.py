from tequila.grouping.fermionic_functions import obt_orb_to_so, tbt_orb_to_so, lr_decomp, compute_meas_alloc, convert_tbts_to_frags, obt_to_ferm, given_rotation, get_obt_tbt, n_elec, of_simplify
import tequila.grouping.fermionic_methods as fm
import tequila as tq
from openfermion import count_qubits, jordan_wigner, normal_ordered, FermionOperator
from numpy.testing import assert_allclose
import numpy as np
import os
from shutil import rmtree
import pickle
from pathlib import Path

TEST_DATA_DIR = Path(__file__).resolve().parent / 'data'

mol_name = "h3"
with open( TEST_DATA_DIR / "h3_sto3g.pkl", "rb") as f:
    h_ferm = pickle.load(f)
h_ferm = normal_ordered(of_simplify(h_ferm))
(obt_orb,tbt_orb) = get_obt_tbt(h_ferm, spin_orb=False)
obt = obt_orb_to_so(obt_orb)
tbt = tbt_orb_to_so(tbt_orb)
ctbts, rtbts, _, uops_orb = lr_decomp(tbt_orb, spin_orb=False)

def test_to_so():
    #orb_to_so doubles the dimensions.
    assert obt.shape[0] == 2 * obt_orb.shape[0]
    assert tbt.shape[0] == 2 * tbt_orb.shape[0]
    #orb_to_so correctly assignes the tensor in spin orbitals.
    for p in range(obt_orb.shape[0]):
        for q in range(obt_orb.shape[1]):
            assert abs(obt_orb[p,q]-obt[2*p+1,2*q+1]) < 1e-6
            assert abs(obt[2*p,2*q+1]) < 1e-6
            assert abs(tbt_orb[p,q,0,0]-tbt[2*p,2*q,1,1]) < 1e-6
            assert abs(tbt_orb[p,q,0,0]-tbt[2*p+1,2*q+1,1,1]) < 1e-6
            assert abs(tbt[2*p,2*q+1,0,0]) <= 1e-6

def test_cartan_diag():
    #Cartan two-body tensors are "diagonal".
    ctbts_diag = np.zeros_like(ctbts)
    for p in range(ctbts.shape[1]):
        for q in range(ctbts.shape[3]):
            ctbts_diag[:,p,p,q,q] = ctbts[:,p,p,q,q]
    assert np.sum(np.abs(ctbts - ctbts_diag)) < 1e-6

def test_lr_rot():
    #The two-body fragments are rotated into cartan fragments.
    rotated_rtbts = np.einsum('ipa, iqb, irc, isd, ipqrs -> iabcd', uops_orb, uops_orb, uops_orb, uops_orb, rtbts)
    assert np.sum(np.abs(rotated_rtbts - ctbts)) < 1e-6

def test_lr_sum():
    #The two-body fragments sum to tbt_orb.
    assert np.sum(np.abs( np.sum(rtbts, axis=0) - tbt_orb )) < 1e-6
            
h_ferm, obt, tbt, n_qubit, all_ops, uops, tbts, cartan_tbts = fm.get_init_ops(h_ferm, mol_name, "lr", spin_orb = False, save=False)
psis_energy, psis_fci = fm.get_wavefunction(jordan_wigner(h_ferm), "fci", mol_name, n_elec(mol_name), save=False)
_, psis_appr = fm.get_wavefunction(jordan_wigner(h_ferm), "cisd", mol_name, n_elec(mol_name), save=False)
psi_appr = fm.truncate_wavefunction(psis_appr[0], perc=50., n_qubits=n_qubit)

def test_truncation():
    #Truncated wavefunction is a single determinant.
    assert len(psi_appr[0]) == 1
    assert np.abs(psi_appr[1]) > 0.5

def test_fff():
    fff_method = "Full"
    rmtree("SAVE/" + mol_name.lower() + "/", ignore_errors=True)
    
    exps_appr, vars_appr = fm.compute_ev_var_all_ops(psi_appr, n_qubit, all_ops, trunc=True)
    ev_dict_all = fm.init_ev_dict(mol_name, psi_appr, n_qubit, trunc=True, save=False)

    O_t = fm.compute_O_t(uops, fff_method, cartan_tbts, mol_name, save=False)
    CovOO = fm.compute_cov_OO(O_t, [ev_dict_all], mol_name, fff_method, save=False)
    Cov0 = fm.compute_cov_O(O_t, obt, [ev_dict_all], mol_name, fff_method, save=False)
    all_covs = fm.compute_all_covs(all_ops, O_t, psi_appr, n_qubit, mol_name, fff_method, trunc=True, save=False)

    Covk = np.zeros(O_t.shape[0] * O_t.shape[1])
    for k in range(O_t.shape[0]):
        for p in range(O_t.shape[1]):
            ind = O_t.shape[1] * k + p
            Covk[ind] = all_covs[k][p]

    fff_var = fm.fff_aux(3, n_qubit, O_t.shape[0], O_t.shape[1], O_t, CovOO, Cov0, Covk, uops, 1e-3)
    new_obt, new_tbts, meas_alloc, var_new = fm.fff_multi_iter(obt, tbts, psi_appr, vars_appr, fff_var, fff_method)
    meas_alloc = compute_meas_alloc(var_new, new_obt, new_tbts, n_qubit, 1e-3)
    exps_new, vars_new = fm.compute_ev_var_all_ops(psis_fci[0], n_qubit, [obt_to_ferm(new_obt,True)] + convert_tbts_to_frags(new_tbts, True))
    #FFF lowers variances.
    assert np.sum(np.divide(vars_new, meas_alloc)) < np.sum(np.sqrt(vars_appr)) ** 2
    print(np.sum(np.sqrt(vars_appr)) ** 2)
    print(np.sum(np.divide(vars_new, meas_alloc)))
    #FFF fragments add up to the original expectation value.
    print(np.sum(exps_new) + h_ferm.constant)
    assert np.abs( np.sum(exps_new) + h_ferm.constant - psis_energy[0] ) < 1e-6

    #FFF fragments are still orbital rotations of HF solvable parts.
    new_c_tbts = np.einsum('ipa, iqb, irc, isd, ipqrs -> iabcd', uops, uops, uops, uops, new_tbts)
    new_c_tbts_diag = np.zeros_like(new_c_tbts)
    for p in range(new_c_tbts.shape[1]):
        for q in range(new_c_tbts.shape[3]):
            new_c_tbts_diag[:,p,p,q,q] = new_c_tbts[:,p,p,q,q]
    assert np.sum(np.abs(new_c_tbts - new_c_tbts_diag)) < 1e-6
   

def test_given_rotation():
    U = np.array([[0, 1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    expected_theta = [[0.0, 2, 1], [-np.pi/2, 1, 0], [-np.pi, 2, 1]]
    expected_phi = [[0.0, 0], [0.0, 1], [np.pi, 2]]

    theta, phi = given_rotation(U)

    assert_allclose(theta, expected_theta)
    assert_allclose(phi, expected_phi)


