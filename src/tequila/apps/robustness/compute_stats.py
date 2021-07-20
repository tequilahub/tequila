import argparse
import _pickle as pickle
from jax.config import config as jax_config
import multiprocessing as mp
import os
import pandas as pd
import sys
import numpy as np
import time

import tequila as tq

from constants import DATA_DIR
from lib.vqe import make_ansatz
from lib.helpers import get_molecule_initializer, make_paulicliques, estimate_ground_state_fidelity
from lib.helpers import print_summary, timestamp_human, Logger
from lib.noise_models import get_noise_model

parser = argparse.ArgumentParser()
parser.add_argument("--molecule", type=str, required=True, choices=['h2', 'lih', 'beh2'])
parser.add_argument("--results_dir", type=str, required=True, help='dir from which to load results')
parser.add_argument("--which", type=str, required=True, default='all', choices=['pauli', 'hamiltonian', 'all'])
parser.add_argument("--samples", "-s", type=int, default=None)
parser.add_argument("--reps", type=int, default=15)
args = parser.parse_args()

geom_strings = {
    'h2': 'H .0 .0 .0\nH .0 .0 {r}',
    'lih': 'H .0 .0 .0\nLi .0 .0 {r}',
    'beh2': 'Be .0 .0 .0\nH .0 .0 {r}\nH .0 .0 -{r}'
}

active_orbitals = {
    'h2': None,
    'lih': None,
    'beh2': None
}

N_PNO = None

columns = ["r", "expectation_values", "variances", "E0", "E1", "gs_fidelity",
           "grouped_pauli_strings", "grouped_pauli_coeffs", "grouped_pauli_expectations",
           "grouped_pauli_eigenvalues",
           "pauli_strings", "pauli_coeffs", "pauli_expectations", "pauli_eigenvalues",
           "nreps", "samples"]


def listener(q, df_save_path, which_stats):
    df = pd.DataFrame(columns=columns)

    while True:
        data = q.get()

        if data == "kill":
            df.sort_values('r', inplace=True)
            df.set_index('r', inplace=True)

            print('saving data as', os.path.join(df_save_path, f'{which_stats}_statistics.pkl'))
            df.to_pickle(path=os.path.join(df_save_path, f'{which_stats}_statistics.pkl'))
            break

        try:
            df = df.append(data, ignore_index=True)
        except Exception as e:
            print('exception occured!', e)

        df.sort_index()


def worker(r, ansatz, hamiltonian, backend, device, noise, samples, vqe_fn, nreps, which_stats, q):
    # load vqe
    with open(vqe_fn.format(r=r), 'rb') as f:
        vqe = pickle.load(f)

    print(f'start computing statistics for r={r}')

    # compute exact solution and spectral gap
    hamiltonian_matrix = hamiltonian.to_matrix()
    eigenvalues, eigenstates = np.linalg.eigh(hamiltonian_matrix)
    lambda0 = min(eigenvalues)  # ground state energy
    lambda1 = min(eigenvalues[eigenvalues > lambda0])  # energy of first excited state

    # compute ground state fidelity
    ground_state_fidelity = estimate_ground_state_fidelity(eigenvalues, eigenstates, ansatz, vqe.variables, backend,
                                                           device, noise, samples)

    # compute stats
    data_hamiltonian = {}
    data_paulis = {}
    if which_stats.lower() in ['hamiltonian', 'all']:
        objective = tq.ExpectationValue(U=ansatz, H=hamiltonian)
        expectation_values = [tq.simulate(objective, variables=vqe.variables, samples=samples, backend=backend,
                                          device=device, noise=noise) for _ in range(nreps)]

        # compute variance
        variances = [tq.simulate(tq.ExpectationValue(U=ansatz, H=(hamiltonian - e) ** 2),
                                 variables=vqe.variables, samples=samples, backend=backend, device=device, noise=noise)
                     for e in expectation_values]

        data_hamiltonian = {'r': r,
                            'expectation_values_hamiltonian': expectation_values,
                            'variances_hamiltonian': variances,
                            'E0': lambda0,
                            'E1': lambda1,
                            'gs_fidelity': ground_state_fidelity,
                            'nreps': nreps,
                            'samples': samples}

    if which_stats.lower() in ['pauli', 'all']:
        # compute pauli expectations w/ grouping
        paulicliques = make_paulicliques(hamiltonian)
        grouped_pauli_strings = [ps.naked() for ps in paulicliques]
        grouped_pauli_coeffs = [ps.coeff.real for ps in paulicliques]
        objectives = [tq.ExpectationValue(U=ansatz + p_str.U, H=p_str.H) for p_str in grouped_pauli_strings]
        grouped_pauli_expectations = [[
            tq.simulate(o, variables=vqe.variables, samples=samples, backend=backend, device=device, noise=noise)
            for o in objectives
        ] for _ in range(nreps)]
        grouped_pauli_eigenvalues = [p_str.compute_eigenvalues() for p_str in grouped_pauli_strings]

        # compute pauli expectations w/o grouping
        pauli_strings = [ps.naked() for ps in hamiltonian.paulistrings]
        pauli_coeffs = [ps.coeff.real for ps in hamiltonian.paulistrings]
        objectives = [
            tq.ExpectationValue(H=tq.QubitHamiltonian.from_paulistrings([p_str]), U=ansatz)
            for p_str in pauli_strings
        ]
        pauli_expectations = [[
            tq.simulate(o, variables=vqe.variables, samples=samples, backend=backend, device=device, noise=noise)
            for o in objectives
        ] for _ in range(nreps)]
        pauli_eigenvalues = [(-1.0, 1.0) for _ in hamiltonian.paulistrings]

        data_paulis = {'r': r,
                       'grouped_pauli_strings': grouped_pauli_strings,
                       'grouped_pauli_coeffs': grouped_pauli_coeffs,
                       'grouped_pauli_expectations': grouped_pauli_expectations,
                       'grouped_pauli_eigenvalues': grouped_pauli_eigenvalues,
                       'pauli_strings': pauli_strings,
                       'pauli_coeffs': pauli_coeffs,
                       'pauli_expectations': pauli_expectations,
                       'pauli_eigenvalues': pauli_eigenvalues,
                       'E0': lambda0,
                       'E1': lambda1,
                       'gs_fidelity': ground_state_fidelity,
                       'nreps': nreps,
                       'samples': samples}

    print(f'finished computing statistics for r={r}')

    # put data in queue
    data = {**data_paulis, **data_hamiltonian}
    data.update({k: np.nan for k in columns if k not in data})

    q.put(data)
    return


def main(results_dir, molecule_name, nreps, which_stats, samples=None):
    with open(os.path.join(results_dir, 'args.pkl'), 'rb') as f:
        loaded_args = pickle.load(f)

    if samples is not None and loaded_args.samples is not None:
        loaded_args.samples = samples

    if f'/{molecule_name.lower()}/' not in results_dir:
        raise ValueError(f'molecule_name {molecule_name.lower()} not in results_dir\n\t{results_dir} !')

    ansatz_name = loaded_args.ansatz
    energies_df = pd.read_pickle(os.path.join(results_dir, 'energies.pkl'))
    bond_dists = energies_df.index.to_list()

    molecule_initializer = get_molecule_initializer(geometry=geom_strings[molecule_name],
                                                    active_orbitals=active_orbitals[molecule_name])

    if loaded_args.gpu:
        try:
            jax_config.update("jax_platform_name", "gpu")
        except RuntimeError:
            jax_config.update("jax_platform_name", "cpu")
            print('WARNING! failed to set "jax_platform_name" to gpu; fallback to CPU')
    else:
        jax_config.update("jax_platform_name", "cpu")

    device = loaded_args.device
    backend = loaded_args.backend
    samples = loaded_args.samples

    if loaded_args.noise == 0:
        backend = 'qulacs'
        noise = None
        device = None
        samples = None
    elif loaded_args.noise == 1:
        noise = get_noise_model(1)
        device = None
    elif loaded_args.noise == 2:  # emulate device noise
        noise = 'device'
    else:
        raise NotImplementedError('noise model {} unknown'.format(loaded_args.noise))

    vqe_result_fn = os.path.join(results_dir, 'vqe_result_r={r}.pkl')
    chemistry_data_dir = DATA_DIR + f'{molecule_name}' + '_{r}'

    # save terminal output to file
    sys.stdout = Logger(print_fp=os.path.join(results_dir, f'{which_stats}_stats_out.txt'))

    # adjust to hcb
    transformation = 'JORDANWIGNER'

    if loaded_args.hcb:
        transformation = 'REORDEREDJORDANWIGNER'

        if 'hcb' not in ansatz_name.lower():
            ansatz_name = 'hcb-' + ansatz_name

    # classical calculations before multiprocessing
    bond_distances_final = []
    molecules, ansatzes, hamiltonians = [], [], []

    for r in bond_dists:
        molecule = molecule_initializer(r=r, name=chemistry_data_dir,
                                        basis_set=loaded_args.basis_set,
                                        transformation=transformation,
                                        n_pno=N_PNO)
        if molecule is None:
            continue

        ansatz = make_ansatz(molecule, ansatz_name)
        hamiltonian = molecule.make_hamiltonian() if not loaded_args.hcb else molecule.make_hardcore_boson_hamiltonian()

        ansatzes.append(ansatz)
        hamiltonians.append(hamiltonian)

        bond_distances_final.append(r)
        molecules.append(molecule)

    print_summary(molecules[0], hamiltonians[0], ansatzes[0], None)

    del molecules

    num_processes = min(len(bond_distances_final), mp.cpu_count()) + 2

    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(num_processes)

    # put listener to work first
    _ = pool.apply_async(listener, (q, results_dir, which_stats))

    # set nreps to 1 if no sampling
    if samples is None and nreps > 1:
        print('setting nreps to 1 since no sampling')
        nreps = 1

    print(f'start_time\t: {timestamp_human()}')

    start_time = time.time()

    # fire off workers
    jobs = []
    for r, ansatz, hamiltonian in zip(bond_distances_final, ansatzes, hamiltonians):
        job = pool.apply_async(
            worker, (r, ansatz, hamiltonian, backend, device, noise, samples, vqe_result_fn, nreps, which_stats, q))
        jobs.append(job)

    # collect results
    for i, job in enumerate(jobs):
        job.get()

    q.put('kill')
    pool.close()
    pool.join()

    print(f'\nend_time\t: {timestamp_human()}')
    print(f'elapsed_time\t: {time.time() - start_time:.4f}s\n')


if __name__ == '__main__':
    main(args.results_dir, args.molecule, args.reps, args.which, args.samples)
