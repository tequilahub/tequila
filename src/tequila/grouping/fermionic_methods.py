import numpy as np
from pathlib import Path
import os
from os.path import exists
from openfermion import FermionOperator, jordan_wigner, get_sparse_operator, expectation, count_qubits
import pickle
import tequila.grouping.fermionic_functions as ferm
from itertools import product
from tequila.hamiltonian import QubitHamiltonian
import scipy as sp
from tequila.grouping.ev_utils import truncate_wavefunction
from tequila import TequilaException
from shutil import rmtree
import logging

def get_psi(h_ferm, mol_name, n_elec, n_qubit, trunc, trunc_perc, get_fci):
    if get_fci:
        _, psis_fci = get_wavefunction(jordan_wigner(h_ferm), "fci", mol_name, n_elec)
    _, psis_appr = get_wavefunction(jordan_wigner(h_ferm), "cisd", mol_name, n_elec)
    psi_appr = psis_appr[0]
    if trunc:
        psi_appr = truncate_wavefunction(psi_appr, perc=trunc_perc, n_qubits=n_qubit)
    if get_fci:
        return psis_fci[0], psi_appr
    else:
        return psi_appr

def do_fff(h_ferm, n_elec, options=None, restart=False, metric_estim=True):
    '''
    Main function for Fluid Fermionic Fragments methods.
    Parameters
    ---------
    options: dictionary: Dictionary containing user-defined parameters:
        mol_name -
        Name of the molecule to perform FFF on: Used for saving restart files.
        fff_method -
        Method for F3 optimization.
        One from Full, R1, or R2 [see Quantum 7, 889 (2023)].
        n_iter -
        Number of FFF iteration.
        calc_type -
        Method used for decomposing the molecular electronic Hamiltonian into Hartree-Fock solvable fragments.
        One from lr or fr. Only lr is implemented for now [see Quantum 7, 889 (2023)].
        trunc_perc -
        Percentage to which CISD WF is truncated. Lower the trunc_perc, the more efficient the optimization, but
        higher the final variances.
        mix -
        Amount of Haar average variance to mix for measurement allocation. [see JCTC 18 (12), 7394-7402].
        fff_thresh -
    F3 optimization is only applied to fragments with variances larger than fff_thresh.
    restart -
    To restart F3 optimization or not.

    Returns
    -------
    Optimized fragment operators in FermionOperator type.
    Unitary operators (orbtial rotations) diagonalizing each fragment.
    '''
    def process_options(options):
        mol_name, fff_method, n_iter, calc_type, trunc_perc, mix, fff_thresh = 'null', "r2", 5, 'lr', 100., 0.0, 1e-4
        if options is not None:
            if "mol_name" in options: mol_name=options["mol_name"]
            if "fff_method" in options: fff_method=options["fff_method"]
            if "n_iter" in options: n_iter=options["n_iter"]
            if "calc_type" in options: calc_type=options["calc_type"]
            if "trunc_perc" in options: trunc_perc=options["trunc_perc"]
            if "mix" in options: mix=options["mix"]
            if "fff_thresh" in options: fff_thresh=options["fff_thresh"]
        return mol_name, fff_method, n_iter, calc_type, trunc_perc, mix, fff_thresh

    def check_restart(restart, mol_name):
        if mol_name == "null":
            logging.warning("Saving restart files to SAVE/null/ specify mol_name=\'desired_path_name\' to save in SAVE/desired_path_name.")
            logging.warning("To use these restart files, run mv SAVE/null SAVE/desired_path_name, then run with options={mol_name:\'desired_path_name\'}.")
            print("Warning: Saving restart files to SAVE/null/ specify mol_name=\'desired_path_name\' to save in SAVE/desired_path_name.")
            print("Warning: To use these restart files, run mv SAVE/null SAVE/desired_path_name, then run with options={mol_name:\'desired_path_name\'}.")
            rmtree("SAVE/" + mol_name.lower() + "/", ignore_errors=True)
        else:
            if not(restart):
                try:
                    os.remove("SAVE/" + mol_name.lower() + "/ev_dict.pkl")
                except OSError:
                    pass
                rmtree("SAVE/" + mol_name.lower() + "/" + fff_method.lower(), ignore_errors=True)

    def get_fff_obj(all_OPS, U_OPS, cartan_tbts, tbts, vars_appr, apply_fff_to):
        all_ops_fff = [all_OPS[0]] + [all_OPS[i + 1] for i in apply_fff_to]
        uops_fff = U_OPS[apply_fff_to]
        cartan_tbts_fff = cartan_tbts[apply_fff_to]
        tbts_fff = tbts[apply_fff_to]
        vars_appr_fff = [vars_appr[0]] + [vars_appr[i + 1] for i in apply_fff_to]
        return all_ops_fff, uops_fff, cartan_tbts_fff, tbts_fff, vars_appr_fff

    def reorganize_fff_obj(new_obt, new_tbts_fff, var_new_fff, all_OPS, tbts, vars_appr):
        new_ops_fff = ferm.convert_tbts_to_frags(new_tbts_fff, True)
        new_tbts = []
        new_all_ops = [ferm.obt_to_ferm(new_obt, True)]
        var_new = [var_new_fff[0]]

        for i in range(len(all_OPS) - 1):
            if i in apply_fff_to:
                new_all_ops.append(None)
                var_new.append(None)
                new_tbts.append(None)
            else:
                new_all_ops.append(all_OPS[i+1])
                var_new.append(vars_appr[i+1])
                new_tbts.append(tbts[i])

        for idx, i in enumerate(apply_fff_to):
            new_all_ops[i + 1] = new_ops_fff[idx]
            new_tbts[i] = new_tbts_fff[idx]
            var_new[i + 1] = var_new_fff[idx+1]

        return new_all_ops, np.array(new_tbts), np.array(var_new)

    mol_name, fff_method, n_iter, calc_type, trunc_perc, mix, fff_thresh = process_options(options)
    check_restart(restart, mol_name)

    if trunc_perc < 100.:
        trunc = True
        if mix < 1e-8: mix = 1e-3
    else:
        trunc = False
    h_ferm, obt, tbt, n_qubit, all_OPS, U_OPS, tbts, cartan_tbts = get_init_ops(h_ferm, mol_name, calc_type, spin_orb = False)
    if metric_estim:
        psi_fci, psi_appr = get_psi(h_ferm, mol_name, n_elec, n_qubit, trunc, trunc_perc, metric_estim)
        print("===================================================")
        print("FCI info before optimization.")
        compute_and_print_ev_var(psi_fci, h_ferm, all_OPS)
        print("===================================================")
    else:
        psi_appr = get_psi(h_ferm, mol_name, n_elec, n_qubit, trunc, trunc_perc, False)


    print("===================================================")
    print("Getting approximate variances")
    _, vars_appr = compute_ev_var_all_ops(psi_appr, n_qubit, all_OPS, trunc=trunc)
    apply_fff_to = np.where(vars_appr[1:] > fff_thresh)[0]
    all_ops_fff, uops_fff, cartan_tbts_fff, tbts_fff, vars_appr_fff = get_fff_obj(all_OPS, U_OPS, cartan_tbts, tbts, vars_appr, apply_fff_to)
    print("Applying F3 to {} fragments out of {}".format(len(vars_appr_fff), len(vars_appr)))
    print("===================================================")

    print("Computing necessary covariance estimates")
    ev_dict_all = init_ev_dict(mol_name, psi_appr, n_qubit, trunc=trunc)
    O_t = compute_O_t(uops_fff, fff_method, cartan_tbts_fff, mol_name)

    CovOO = compute_cov_OO(O_t, [ev_dict_all], mol_name, fff_method)
    Cov0 = compute_cov_O(O_t, obt, [ev_dict_all], mol_name, fff_method)
    all_covs = compute_all_covs(all_ops_fff, O_t, psi_appr, n_qubit, mol_name, fff_method, trunc=trunc)

    Covk = np.zeros(O_t.shape[0] * O_t.shape[1])
    for k in range(O_t.shape[0]):
        for p in range(O_t.shape[1]):
            ind = O_t.shape[1] * k + p
            Covk[ind] = all_covs[k][p]

    fff_var = fff_aux(n_iter, n_qubit, O_t.shape[0], O_t.shape[1], O_t, CovOO, Cov0, Covk, uops_fff, mix)

    new_obt, new_tbts_fff, meas_alloc, var_new_fff = fff_multi_iter(obt, tbts_fff, psi_appr, vars_appr_fff, fff_var, fff_method)
    new_all_ops, new_tbts, var_new = reorganize_fff_obj(new_obt, new_tbts_fff, var_new_fff, all_OPS, tbts, vars_appr)

    print("Allocating measurements")
    meas_alloc = ferm.compute_meas_alloc(var_new, new_obt, new_tbts, n_qubit, mix)
    if metric_estim:
        print("===================================================")
        print("FCI info after optimization.")
        compute_and_print_ev_var(psi_fci, h_ferm, new_all_ops, meas_alloc=meas_alloc)
        print("===================================================")
    _, uop_oe = np.linalg.eig(new_obt)
    all_uops = [uop_oe] + [U_OPS[i] for i in range(len(U_OPS))]

    new_c_obt = np.einsum("pa, qb, pq", uop_oe, uop_oe, new_obt)
    new_c_tbts = np.einsum('ipa, iqb, irc, isd, ipqrs -> iabcd', U_OPS, U_OPS, U_OPS, U_OPS, new_tbts)
    return new_all_ops, np.array(all_uops), new_c_obt, new_c_tbts, meas_alloc

def do_svd(h_ferm, n_elec):
    spin_orb = False
    obtb = ferm.get_obt_tbt(h_ferm, spin_orb = spin_orb)
    obt = obtb[0]
    tbt_ham_opt = obtb[1]
    n_qubit = ferm.qubit_number(h_ferm)
    if spin_orb != True:
        obt = ferm.obt_orb_to_so(obt)

    n = obt.shape[0]
    CARTAN_TBTS_tmp, TBTS_tmp, OPS_tmp, U_OPS_tmp = ferm.lr_decomp(tbt_ham_opt, spin_orb=spin_orb)
    if spin_orb != True:
        U_OPS = ferm.convert_u_to_so(U_OPS_tmp)
        tbts = np.zeros([len(OPS_tmp), n, n, n, n])
        cartan_tbts = np.zeros([len(OPS_tmp), n, n, n, n])
        for i in range(len(OPS_tmp)):
            tbts[i,:,:,:,:] = ferm.tbt_orb_to_so(TBTS_tmp[i,:,:,:,:])
            cartan_tbts[i, :, :, :, :] = ferm.tbt_orb_to_so(CARTAN_TBTS_tmp[i, :, :, :, :])
    else:
        U_OPS = U_OPS_tmp
        tbts = TBTS_tmp
        cartan_tbts = CARTAN_TBTS_tmp

    all_OPS = [ferm.obt_to_ferm(obt, True)]
    for i in range(len(OPS_tmp)):
        all_OPS.append(OPS_tmp[i])

    _, uop_oe = np.linalg.eig(obt)
    all_uops = [uop_oe] + [U_OPS[i] for i in range(len(U_OPS))]
    cartan_obt = np.einsum("pa, qb, pq", uop_oe, uop_oe, obt)

    _, psis_appr = get_wavefunction(jordan_wigner(h_ferm), "cisd", "None", n_elec, save=False)
    print("Allocating measurements")
    _, vars_appr = compute_ev_var_all_ops(psis_appr[0], n_qubit, all_OPS, trunc=False)

    meas_alloc = ferm.compute_meas_alloc(vars_appr, obt, tbts, n_qubit, mix = 0.0)

    return all_uops, cartan_obt, cartan_tbts, meas_alloc


def get_fermion_wise(H, U, qubit_list = []):
    '''
    Return z_form and orbital rotations over qubits at qubit_list
    '''

    H = ferm.cartan_tbt_to_ferm(H, spin_orb = True)
    z_form = QubitHamiltonian(jordan_wigner(H))

    circuit = ferm.get_orb_rot(U, qubit_list=qubit_list, tol = 1e-12)
    return [z_form, circuit]

def get_init_ops(h_ferm, mol_name, calc_type, spin_orb, save=True):
    '''
    Parameters
     ----------
    mol_name -
    Name of the molecule
    calc_type -
    Method used to obtain fragments of hamiltonian (LR Decomposition/ GFRO).
    spin_orb -
    If true, spin-orbitals is used, if false, spin symmetry is used to reduce the number of orbitals by half (assumes that spin-up and spin-down are identical)

    Returns
    -------
    h_ferm Fermionic Hamiltonian of the molecular system.
    obt Tuple of one body integrals.
    tbt Tuple of two body integrals.
    n_qubits Number of qubits in the molecular system.
    all_OPS Fragments of Hamiltonian in Fermionic form
    U_OPS Orbital rotations
    tbts LR Decomposition of two body integrals
    cartan_tbts Polynomial functions of Pauli Z under qubit fermion mappings
    '''
    path = "SAVE/" + mol_name + "/"
    if os.path.isfile(path + "tensor_terms.pkl"):
        print("Using saved Hamiltonian from {}. Run with a different mol_name if this is not desired.".format(path))
        with open(path + "tensor_terms.pkl", 'rb') as file:
            INIT = pickle.load(file)
        obt = INIT[0]
        tbt_ham_opt = INIT[1]
        with open(path + "ham.pkl", 'rb') as file:
            h_ferm = pickle.load(file)
    else:
        obtb = ferm.get_obt_tbt(h_ferm, spin_orb = spin_orb)
        obt = obtb[0]
        tbt_ham_opt = obtb[1]
        if save:
            Path(path).mkdir(exist_ok=True, parents=True)
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

    if calc_type.lower() == "lr":
        if os.path.isfile(path + "lr.pkl"):
            print("Using saved LR decomposition saved in {}. Run with a different mol_name if this is not desired.".format(path))
            with open(path + "lr.pkl", 'rb') as file:
                INIT = pickle.load(file)
            CARTAN_TBTS_tmp = INIT[0]
            TBTS_tmp = INIT[1]
            U_OPS_tmp = INIT[2]
        else:
            CARTAN_TBTS_tmp, TBTS_tmp, OPS_tmp, U_OPS_tmp = ferm.lr_decomp(tbt_ham_opt, spin_orb=spin_orb)
            if save:
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

def get_wavefunction(Hq, wf_type, mol_name, n_elec, N=1, save=True):
    '''
    Parameters
     ----------
    h_ferm -
    Molecular electronic Hamiltonian in the second quantized form.
    wf_type -
    Type of the wavefunction: CISD or FCI.
    mol_name -
    Name of molecule
    N -
    Number of eigenstates/eigenenergies to return.

    Returns
    -------
    energies Eigenenergies
    psi Eigenstates
    '''
    n_qubits = count_qubits(Hq)
    if wf_type.lower() == "fci":
       return get_fci_states(Hq, mol_name, n_elec, n_qubits, N=N, save=save)
    elif wf_type.lower() == "cisd":
       return get_cisd_states(Hq, mol_name, n_elec, n_qubits, N=N, save=save)

def get_fci_states(Hq, mol_name, n_elec, n_qubits, N=1, save=True):
    '''
    Parameters
     ----------
    Hq -
    Jordan wigner transform of fermionic hamiltonian
    mol_name -
    Name of molecule
    n_elec -
    Number of electroncs in the system
    n_qubit -
    Number of qubits in the hamiltonian

    Returns
    -------
    e_fci Energy of FCI ground state
    psi_fci Wavefunction of FCI ground state
    '''

    path = "SAVE/" + mol_name.lower() + "/"
    if os.path.isfile(path + "psi_fci.pkl"):
        print("Using saved psi_fci in {}. Run with a different mol_name if this is not desired.".format(path))
        with open(path + "psi_fci.pkl", 'rb') as file:
            INIT = pickle.load(file)
        e_fci = INIT[0]
        psi_fci = INIT[1]
        return e_fci, psi_fci

    Nop = FermionOperator.zero()
    for i in range(n_qubits):
        Nop += FermionOperator("{}^ {}".format(i, i))
    Nop = jordan_wigner(Nop)

    M = 4 * N
    sparse_H = get_sparse_operator(Hq, n_qubits)
    size_H = sparse_H.get_shape()[0]
    if M >= size_H - 1:
        M = size_H - 2
    w,v = sp.sparse.linalg.eigsh(sparse_H, k = max(10, M), which = "SA")
    srt_arg = np.argsort(w)
    w = w[srt_arg]
    v = v[:, srt_arg]
    values = []
    vectors = []
    for i in range(len(w)):
        Nel = expectation(get_sparse_operator(Nop, n_qubits), v[:,i])
        if np.abs(Nel - n_elec) < 1e-6:
            values.append(w[i])
            vectors.append(v[:,i])
        if len(values) == N or i == len(w)-1:
            if save:
                Path(path).mkdir(exist_ok=True)
                with open(path + "psi_fci.pkl", 'wb') as file:
                    pickle.dump([values, vectors], file)
            return values, vectors

def get_cisd_states(Hq, mol_name, n_elec, n_qubits, N=1, save=True):
    '''
    Parameters
     ----------
    Hq -
    Jordan wigner transform of fermionic hamiltonian
    mol_name -
    Name of molecule
    n_elec -
    Number of electrons in the system
    n_qubit -
    Number of qubits in the hamiltonian

    Returns
    -------
    e_cisd Energy of CISD ground state
    psi_cisd Wavefunction of CISD ground state
    '''
    path = "SAVE/" + mol_name.lower() + "/"
    if os.path.isfile(path + "psi_cisd.pkl"):
        print("Using saved psi_cisd in {}. Run with a different mol_name if this is not desired.".format(path))
        with open(path + "psi_cisd.pkl", 'rb') as file:
            INIT = pickle.load(file)
        e_cisd = INIT[0]
        psi_cisd = INIT[1]
        return e_cisd, psi_cisd

    indices = ferm.get_jw_cisd_basis_states(n_elec, n_qubits)
    H_mat_cisd = ferm.create_hamiltonian_in_subspace(indices, Hq, n_qubits)
    size_H = H_mat_cisd.get_shape()[0]
    if N >= size_H - 1:
        M = size_H - 1
    else:
        M = N
    w,v = sp.sparse.linalg.eigsh(H_mat_cisd, k = M, which = "SA")
    order = np.argsort(w)
    values = w[order].tolist()
    vectors = []
    for i in order:
        wfs = np.zeros(2**n_qubits)
        for iidx, iindx in enumerate(indices):
            wfs[iindx] = v[iidx, i]
        wfs = wfs/np.linalg.norm(wfs)
        vectors.append(wfs)
    if save:
        Path(path).mkdir(exist_ok=True)
        with open(path + "psi_cisd.pkl", 'wb') as file:
            pickle.dump([values, vectors], file)

    return values, vectors

def compute_and_print_ev_var(psi, h_ferm, all_OPS, meas_alloc=None):
    '''
    Parameters
     ----------
    psi Wavefunction
    h_ferm Fermionic hamiltonian of the molecule
    all_OPS Fermionic fragments
    meas_alloc If given, the shot per fragment is allocated according to meas_alloc. If None, then
    it is computed according to the variances [see Quantum 5, 385 (2021)].

    Prints out the variances and expectation of each fragment over psi.
    '''
    n_qubit = ferm.qubit_number(h_ferm)
    h_const = h_ferm.constant
    exps, variances = compute_ev_var_all_ops(psi, n_qubit, all_OPS)
    if meas_alloc is None: meas_alloc = ferm.compute_meas_alloc(variances)
    scaled_variances = np.divide(variances,meas_alloc)

    scaled_variances_sum = np.sum(scaled_variances)
    print("Full variances:")
    print(variances)
    print("Expectations")
    print(exps)
    print("Variance metric value is {}".format(scaled_variances_sum))
    print("Exp value is {}".format(np.sum(exps) + h_const))

def compute_ev_var_all_ops(psi, n_qubit, all_OPS, trunc=False):
    '''
    Parameters
     ----------
    psi Wavefunction
    n_qubit Number of qubits
    all_OPS Fermionic fragments

    Returns
    -------
    exps Expectations of the fragments over psi
    variances Variances of the fragments over psi
    '''
    num_frags = len(all_OPS)
    exps = np.zeros(num_frags)
    variances = np.zeros(num_frags)
    for i in range(num_frags):
        exps[i] = ferm.expectation_value(all_OPS[i], psi, n_qubit, trunc=trunc)
        variances[i] = ferm.variance_value(all_OPS[i], psi, n_qubit, trunc=trunc)
    return exps, variances

def init_ev_dict(mol_name, psi, n_qubit, trunc=False, spin_orb=True, save=True):
    '''
    Parameters
     ----------
    psi Wavefunction
    n_qubit Number of qubits
    mol_name Name of molecule

    Returns
    -------
    ev_dict_all Returns a dictionary of expectaion and variances of fermionic operators over psi
    '''
    if spin_orb:
        n = n_qubit
    else:
        n = n_qubit // 2
    path = "SAVE/" + mol_name.lower() + "/"
    if os.path.isfile(path + "ev_dict.pkl"):
        with open(path + "ev_dict.pkl", 'rb') as file:
            ev_dict_all = pickle.load(file)
    else:
        ev_dict_E = ferm.get_E(psi, n, n_qubit, trunc=trunc)
        ev_dict_EE = ferm.get_EE(psi, n, n_qubit, trunc=trunc)
        ev_dict_all = ferm.reorganize(n, ev_dict_E, ev_dict_EE)
        if save:
            Path(path).mkdir(exist_ok=True)
            with open(path + "ev_dict.pkl", 'wb') as file:
                pickle.dump(ev_dict_all, file)
    return ev_dict_all

def check_method(method):
    if method.lower() not in ["full", "r1", "r2"]:
       raise TequilaException("method has to be specified as one from Full, R1 or R2")

def compute_O_t(U_OPS, method, tbts, mol_name, save=True):
    '''
    Parameters
     ----------
    U_OPS Orbital rotations
    tbts Polynomial functions of Pauli Z under qubit fermion mappings
    mol_name Name of molecule
    method FFF method (Full / R1 / R2)
    Returns
    -------
    O_t O_alpha (arXiv:2208.14490v3 - Section 2.3)
    '''
    check_method(method)

    path = "SAVE/" + mol_name.lower() + "/" + method.lower() + "/"
    if os.path.isfile(path + "O_t.pkl"):
        with open(path + "O_t.pkl", 'rb') as file:
            O_t = pickle.load(file)
        return O_t

    num_frags = U_OPS.shape[0]
    n = U_OPS.shape[1]
    if method.lower() == "full":
        n_param = n//2
    else:
        n_param = 1
        ratios = np.zeros([num_frags,n])
        for frag1 in range(num_frags):
            for p1 in range(n):
                if method.lower() == "r1":
                    ratios[frag1, p1] = tbts[frag1, p1,p1,p1,p1]
                else:
                    ratios[frag1, p1] = np.sum([tbts[frag1,p1,p1,r1,r1] for r1 in range(n)])
    O_t_tmp = np.zeros([num_frags, n, n, n])
    O_t = np.zeros([num_frags, n_param, n, n])
    Otmp = np.zeros([n,n])
    for k in range(num_frags):
        Umat = U_OPS[k, :, :]
        for p in range(n):
            for r, s in product(range(n), repeat=2):
                Otmp[r,s] = Umat[r,p] * Umat[s,p].conjugate()
            O_t_tmp[k,p,:,:] = Otmp
    if method.lower() == "full":
        for k in range(num_frags):
            for p in range(n_param):
                for alpha in range(2):
                    O_t[k,p,:,:] += O_t_tmp[k,2*(p)+alpha,:,:]
    else:
        for k in range(num_frags):
            for p1 in range(n):
                O_t[k,0,:,:] += ratios[k,p1] * O_t_tmp[k,p1,:,:]

    if save:
        Path(path).mkdir(exist_ok=True)
        with open(path + "O_t.pkl", 'wb') as file:
                pickle.dump(O_t, file)

    return O_t

def compute_all_covs(all_OPS, O_t, psi, n_qubit, mol_name, method, trunc=False, save=True):
    '''
    Parameters
     ----------
    psi Wavefunction
    n_qubit Number of qubits
    all_OPS Fermionic fragments
    O_t O_alpha (arXiv:2208.14490v3 - Section 2.3)
    mol_name Name of molecule
    method FFF method (Full / R1 / R2)

    Returns
    -------
    all_covs --------------------
    '''
    path = "SAVE/" + mol_name.lower() + "/" + method.lower() + "/"
    check_method(method)
    if os.path.isfile(path + "all_covs.pkl"):
        with open(path + "all_covs.pkl", 'rb') as file:
            all_covs = pickle.load(file)
        return all_covs

    all_covs = []
    for frag_idx in range((len(all_OPS) - 1)):
        ops1 = []
        for p in range(O_t.shape[1]):
            ops1.append(ferm.obt_to_ferm(O_t[frag_idx, p, :, :], True))
        all_covs.append(ferm.compute_covk(ops1, all_OPS[frag_idx+1], psi, n_qubit, trunc=trunc))

    if save:
        Path(path).mkdir(exist_ok=True)
        with open(path + "all_covs.pkl", 'wb') as file:
            pickle.dump(all_covs, file)

    return all_covs

def compute_cov_OO(O_t, ev_dict_all, mol_name, method, save=True):
    '''
    Parameters
     ----------
    O_t O_alpha (arXiv:2208.14490v3 - Section 2.3)
    ev_dict_all Returns a dictionary of expectaion and variances of fermionic operators over psi
    mol_name Name of molecule
    method FFF method (Full / R1 / R2)

    Returns
    -------
    covmat Dictionary of covariances between O_alpha's
    '''
    path = "SAVE/" + mol_name.lower() + "/" + method.lower() + "/"
    check_method(method)
    if os.path.isfile(path + "cov_OO.pkl"):
        with open(path + "cov_OO.pkl", 'rb') as file:
            covmat = pickle.load(file)
        return covmat

    nf = O_t.shape[0]
    n = O_t.shape[1]
    nmat = nf * n
    covmat = np.zeros([nmat, nmat])
    for k in range(nf):
        for p in range(n):
            for l in range(nf):
                for q in range(n):
                    ind1 = n * (k) + p
                    ind2 = n * (l) + q
                    covmat[ind1, ind2] = ferm.covariance_ob_ob(O_t[k,p,:,:], O_t[l,q,:,:], ev_dict_all[0])

    if save:
        Path(path).mkdir(exist_ok=True)
        with open(path + "cov_OO.pkl", 'wb') as file:
            pickle.dump(covmat, file)

    return covmat

def compute_cov_O(O_t, H0, ev_dict_all, mol_name, method, save=True):
    '''
    Parameters
     ----------
    O_t O_alpha (arXiv:2208.14490v3 - Section 2.3)
    H0 One body term of the Hamiltonian
    ev_dict_all Returns a dictionary of expectaion and variances of fermionic operators over psi
    mol_name Name of molecule
    method FFF method (Full / R1 / R2)

    Returns
    -------
    covvec Covariance of O_alpha's and original fragments
    '''
    path = "SAVE/" + mol_name.lower() + "/" + method.lower() + "/"
    check_method(method)
    if os.path.isfile(path + "cov_O.pkl"):
        with open(path + "cov_O.pkl", 'rb') as file:
            covvec = pickle.load(file)
        return covvec

    nf = O_t.shape[0]
    n = O_t.shape[1]
    nvec = nf * n
    covvec = np.zeros(nvec)
    for k in range(nf):
        for p in range(n):
            ind = n * (k) + p
            covl = ferm.covariance_ob_ob(O_t[k,p,:,:], H0, ev_dict_all[0])
            covr = ferm.covariance_ob_ob(H0, O_t[k,p,:,:], ev_dict_all[0])
            covvec[ind] = covl + covr

    if save:
        Path(path).mkdir(exist_ok=True)
        with open(path + "cov_O.pkl", 'wb') as file:
            pickle.dump(covvec, file)

    return covvec

class fff_aux:
    '''
    Class containing all variables needed for FFF.
    '''
    def __init__(self, n_iter, nq, nf, n, o_t, coo, c0, ck, uops, mix):
        self.n_iter = n_iter
        self.nq = nq
        self.nf = nf
        self.n = n
        self.o_t = o_t
        self.coo = coo
        self.c0 = c0
        self.ck = ck
        self.uops = uops
        self.mix = mix

def fff_multi_iter(obt, tbts, psi, varbs, fff_var, method):
    '''
    Parameters
     ----------
    psi Wavefunction
    obt One body terms
    tbts Two body terms
    varbs Variances of the fragments
    fff_var Class containing all the variables needed for FFF
    method FFF method (Full / R1 / R2)

    Returns
    -------
    new_obt Repartitioned one body term
    new_tbts Repartitioned two body terms
    m0 Optimal Measurement Allocation
    '''
    check_method(method)

    ntmp = obt.shape[0]
    obt_list = np.zeros([ fff_var.n_iter+1, ntmp, ntmp])
    tbts_list = np.zeros([fff_var.n_iter+1, fff_var.nf, ntmp, ntmp, ntmp, ntmp])
    var_list = np.zeros([fff_var.n_iter+1, fff_var.nf+1])
    obt_list[0,:,:] = obt
    tbts_list[0,:,:,:,:,:] = tbts
    var_list[0,:] = varbs
    new_c0 = fff_var.c0
    new_ck = fff_var.ck
    for i in range(fff_var.n_iter):
        print("Progress: iteration #{} out of {}".format(i + 1, fff_var.n_iter))
        new_obt, new_tbts, new_vars, new_c0, new_ck = ferm.fff_1_iter(obt_list[i,:,:], tbts_list[i,:,:,:,:,:], var_list[i,:], new_c0, new_ck, fff_var)
        obt_list[i+1,:,:] = new_obt
        tbts_list[i+1,:,:,:,:,:] = new_tbts
        var_list[i+1,:] = new_vars
    var0 = var_list[len(var_list)-1,:]
    m0 = ferm.compute_meas_alloc(var0, obt, tbts, fff_var.nq, fff_var.mix)
    return obt_list[len(obt_list)-1,:,:], tbts_list[len(tbts_list)-1,:,:,:,:,:], m0, var0
