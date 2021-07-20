import argparse
import _pickle as pickle
from collections import namedtuple
import copy
from jax.config import config as jax_config
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import sys
import time

import tequila as tq  # noqa

from constants import ROOT_DIR, DATA_DIR
from lib.vqe import make_ansatz
from lib.helpers import filtered_dists, get_molecule_initializer, compute_energy_classical
from lib.helpers import print_summary, timestamp_human, timestamp, Logger
from lib.noise_models import get_noise_model

parser = argparse.ArgumentParser()
parser.add_argument("--molecule", type=str, choices=['h2', 'lih', 'beh2'])
parser.add_argument("--ansatz", type=str, required=True, choices=['upccgsd', 'spa-gas', 'spa-gs', 'spa-s', 'spa'])
parser.add_argument("--basis_set", "-bs", type=str, default=None)
parser.add_argument("--hcb", action="store_true")
parser.add_argument("--noise", type=int, default=0, choices=[0, 1, 2])
parser.add_argument("--samples", "-s", type=int, default=None)
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--results_dir", type=str, default=os.path.join(ROOT_DIR, "results/"))
parser.add_argument("--optimizer", type=str, default='COBYLA')
parser.add_argument("--backend", type=str, default="qulacs")
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--restarts", type=int, default=1)
parser.add_argument("--rand_dir", type=int, default=1, choices=[0, 1])
args = parser.parse_args()

N_PNO = None

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

columns = ["r", "fci", "mp2", "ccsd", "vqe"]


def listener(q, df_save_as):
    df = pd.DataFrame(columns=columns)

    while 1:
        m = q.get()

        if m == "kill":
            df.sort_values('r', inplace=True)
            df.set_index('r', inplace=True)
            df.to_pickle(path=df_save_as)
            break

        print("{:^14} | ".format("[" + timestamp_human() + "]") + " | ".join(["{:^12}".format(round(v, 6)) for v in m]))

        # add to df
        df.loc[-1] = m
        df.index += 1
        df.sort_index()


def worker(r, ansatz, hamiltonian, optimizer, backend, device, noise, samples, fci, mp2, ccsd, restarts, vqe_fn, q):
    # run vqe
    objective = tq.ExpectationValue(U=ansatz, H=hamiltonian, optimize_measurements=False)

    Result = namedtuple('result', 'energy')
    result = Result(energy=np.inf)

    # restart optimization n_reps times
    for _ in range(restarts):
        init_vals = {k: np.random.normal(loc=0, scale=np.pi / 4.0) for k in objective.extract_variables()}
        temp_result = tq.minimize(objective, method=optimizer, initial_values=init_vals, silent=True,
                                  backend=backend, device=device, noise=noise, samples=samples)

        if temp_result.energy <= result.energy:
            result = copy.deepcopy(temp_result)

    # save SciPyResults
    with open(vqe_fn.format(r=r), 'wb') as f:
        pickle.dump(result, f)

    # put data in queue
    q.put([r, fci, mp2, ccsd, result.energy])


def main():
    molecule_name = args.molecule.lower()
    ansatz_name = args.ansatz

    if args.basis_set is None:
        bond_dists = filtered_dists(DATA_DIR, args.molecule.lower())
    else:
        bond_dists = list(np.arange(start=0.4, stop=5.0, step=0.2).round(2))

    molecule_initializer = get_molecule_initializer(geometry=geom_strings[molecule_name],
                                                    active_orbitals=active_orbitals[molecule_name])

    if args.gpu:
        try:
            jax_config.update("jax_platform_name", "gpu")
        except RuntimeError:
            jax_config.update("jax_platform_name", "cpu")
            print('WARNING! failed to set "jax_platform_name" to gpu; fallback to CPU')
    else:
        jax_config.update("jax_platform_name", "cpu")

    device = args.device
    backend = args.backend
    samples = args.samples

    if args.noise == 0:
        backend = 'qulacs'
        noise = None
        device = None
        samples = None
    elif args.noise == 1:
        noise = get_noise_model(1)
        device = None
    elif args.noise == 2:  # emulate device noise
        noise = 'device'
    else:
        raise NotImplementedError('noise model {} unknown'.format(args.noise))

    # build dir structure
    save_dir = os.path.join(args.results_dir, f"./{molecule_name}/")
    save_dir = os.path.join(save_dir, f"{'basis-set-free' if args.basis_set is None else args.basis_set}/")
    save_dir = os.path.join(save_dir, f"hcb={args.hcb}/{ansatz_name}/")
    save_dir = os.path.join(save_dir, f"noise={args.noise if device is None else device}/")
    save_dir = os.path.join(save_dir, f"{timestamp()}/") if args.rand_dir else save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    results_file = os.path.join(save_dir, 'energies.pkl')
    vqe_result_fn = save_dir + 'vqe_result_r={r}.pkl'
    args_fn = os.path.join(save_dir, 'args.pkl')

    with open(args_fn, 'wb') as f:
        pickle.dump(args, f)
        print(f'saved args to {args_fn}')

    chemistry_data_dir = DATA_DIR + f'{molecule_name}' + '_{r}'

    # save terminal output to file
    sys.stdout = Logger(print_fp=os.path.join(save_dir, 'vqe_out.txt'))

    # adjust to hcb
    transformation = 'JORDANWIGNER'

    if args.hcb:
        transformation = 'REORDEREDJORDANWIGNER'

        if 'hcb' not in ansatz_name.lower():
            ansatz_name = 'hcb-' + ansatz_name

    # classical calculations before multiprocessing
    bond_distances_final = []
    fci_vals, mp2_vals, ccsd_vals, hf_vals = [], [], [], []
    molecules = []
    ansatzes, hamiltonians = [], []

    for r in bond_dists:
        molecule = molecule_initializer(
            r=r, name=chemistry_data_dir, basis_set=args.basis_set, transformation=transformation, n_pno=N_PNO)

        if molecule is None:
            continue

        # make classical computations
        hf_vals.append(compute_energy_classical('hf', molecule))
        fci_vals.append(compute_energy_classical('fci', molecule))
        mp2_vals.append(compute_energy_classical('mp2', molecule))
        ccsd_vals.append(compute_energy_classical('ccsd', molecule))

        # build ansatz
        ansatz = make_ansatz(molecule, ansatz_name)
        hamiltonian = molecule.make_hamiltonian() if not args.hcb else molecule.make_hardcore_boson_hamiltonian()

        ansatzes.append(ansatz)
        hamiltonians.append(hamiltonian)

        bond_distances_final.append(r)
        molecules.append(molecule)

    print_summary(molecules[0], hamiltonians[0], ansatzes[0], None)

    del molecules

    num_processes = min(len(bond_dists), mp.cpu_count()) + 2

    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(num_processes)

    # put listener to work first
    _ = pool.apply_async(listener, (q, results_file))

    print(f'start_time\t: {timestamp_human()}')

    # print progress header
    print("\n{:^14} | ".format("time") + " | ".join(["{:^12}".format(v) for v in columns]))
    print("-" * 15 + "+" + "+".join(["-" * 14 for _ in range(len(columns))]))

    start_time = time.time()

    # fire off workers
    jobs = []
    for r, ansatz, hamiltonian, fci, mp2, ccsd in zip(bond_distances_final, ansatzes, hamiltonians, fci_vals, mp2_vals,
                                                      ccsd_vals):
        job = pool.apply_async(
            worker,
            (r, ansatz, hamiltonian, args.optimizer, backend, device, noise, samples, fci, mp2, ccsd,
             args.restarts, vqe_result_fn, q))

        jobs.append(job)

    # collect results
    for i, job in enumerate(jobs):
        job.get()

    q.put('kill')
    pool.close()
    pool.join()

    print(f'\nend_time\t: {timestamp_human()}')
    print(f'elapsed_time\t: {time.time() - start_time:.4f}s')


if __name__ == '__main__':
    main()
