import argparse
import numpy as np
import os
import pandas as pd
import sys
from tabulate import tabulate

from lib.robustness_interval import compute_robustness_interval
from lib.helpers import Logger

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str, required=False, help='dir from which to load results')
args = parser.parse_args()

columns = ["r", "vqe", "fci", "mp2", "ccsd",
           "E0", "E1", "spectral_gap", "true_fidelity", "variance",
           "fidelity_eckart", "fidelity_mcclean", "new_fidelity_bound",
           "grouped_sdp_lower_bound", "grouped_sdp_upper_bound",
           "singleterms_sdp_lower_bound", "singleterms_sdp_upper_bound",
           "gram_exp_lower_bound", "gram_eigval_lower_bound",
           "gram_eigval_upper_bound"]


def main(results_dir):
    # open df with stats
    stats_df = pd.read_pickle(os.path.join(results_dir, 'all_statistics.pkl'))
    energies_df = pd.read_pickle(os.path.join(results_dir, 'energies.pkl'))

    # setup df for intervals
    df = pd.DataFrame(columns=columns)

    # save terminal output to file
    sys.stdout = Logger(print_fp=os.path.join(results_dir, 'intervals_out.txt'))

    for r in stats_df.index:
        fidelity = stats_df.loc[r]['gs_fidelity']

        # compute SDP expectation interval without grouping
        stats = {'expectation_values': stats_df.loc[r]['pauli_expectations'],
                 'pauli_strings': stats_df.loc[r]['pauli_strings'],
                 'pauli_coeffs': stats_df.loc[r]['pauli_coeffs'],
                 'pauli_eigenvalues': stats_df.loc[r]['pauli_eigenvalues']}

        sdp_bounds = compute_robustness_interval(method='sdp', kind='expectation', statistics=stats, fidelity=fidelity)

        # compute SDP expectation interval with grouping
        stats = {'expectation_values': stats_df.loc[r]['grouped_pauli_expectations'],
                 'pauli_strings': stats_df.loc[r]['grouped_pauli_strings'],
                 'pauli_coeffs': stats_df.loc[r]['grouped_pauli_coeffs'],
                 'pauli_eigenvalues': stats_df.loc[r]['grouped_pauli_eigenvalues']}

        sdp_bounds_grouped = compute_robustness_interval(method='sdp', kind='expectation', statistics=stats,
                                                         fidelity=fidelity)

        # compute Gramian Expectation lower bound
        pauli_coeffs = stats_df.loc[r]['pauli_coeffs']
        normalization_constant = pauli_coeffs[0] - np.sum(np.abs(pauli_coeffs[1:]))
        stats = {'expectation_values': stats_df.loc[r]['expectation_values_hamiltonian'],
                 'variances': stats_df.loc[r]['variances_hamiltonian']}

        gramian_bound = compute_robustness_interval(method='gramian', kind='expectation', statistics=stats,
                                                    fidelity=fidelity, normalization_const=normalization_constant)

        # compute Gramian Eigenvalue Interval
        stats = {'expectation_values': stats_df.loc[r]['expectation_values_hamiltonian'],
                 'variances': stats_df.loc[r]['variances_hamiltonian']}

        gramian_eigval_bounds = compute_robustness_interval(method='gramian', kind='eigenvalues', statistics=stats,
                                                            fidelity=fidelity)

        print(gramian_eigval_bounds.lower_bound)

        # fidelity bounds
        vqe_energy = np.mean(stats_df.loc[r]['expectation_values_hamiltonian'])
        vqe_variance = np.mean(stats_df.loc[r]['variances_hamiltonian'])
        spectral_gap = stats_df.loc[r]["E1"] - stats_df.loc[r]["E0"]

        if vqe_energy <= 0.5 * (stats_df.loc[r]["E0"] + stats_df.loc[r]["E1"]):
            eckart_bound = mcclean_bound = fidelity_bound = 0.5
            if spectral_gap > np.sqrt(vqe_variance):
                mcclean_bound = max(0.5, 1 - np.sqrt(vqe_variance) / spectral_gap)

            if spectral_gap * 0.5 > np.sqrt(vqe_variance):
                fidelity_bound = 0.5 * (1 + np.sqrt(1.0 - 4 * vqe_variance / (spectral_gap ** 2)))
        else:
            eckart_bound = mcclean_bound = fidelity_bound = 0

        df.loc[-1] = [r,
                      vqe_energy,
                      energies_df.loc[r]["fci"],
                      energies_df.loc[r]["mp2"],
                      energies_df.loc[r]["ccsd"],
                      stats_df.loc[r]["E0"],
                      stats_df.loc[r]["E1"],
                      stats_df.loc[r]["E1"] - stats_df.loc[r]["E0"],
                      stats_df.loc[r]["gs_fidelity"],
                      np.mean(stats_df.loc[r]['variances_hamiltonian']),
                      eckart_bound, mcclean_bound, fidelity_bound,
                      sdp_bounds_grouped.lower_bound,
                      sdp_bounds_grouped.upper_bound,
                      sdp_bounds.lower_bound,
                      sdp_bounds.upper_bound,
                      gramian_bound.lower_bound,
                      gramian_eigval_bounds.lower_bound,
                      gramian_eigval_bounds.upper_bound]
        df.index += 1
        df.sort_index()

    # save df
    df.sort_values('r', inplace=True)
    df.set_index('r', inplace=True)
    df.to_pickle(path=os.path.join(results_dir, 'intervals.pkl'))

    print('\n\nComputed Intervals:\n')
    print(tabulate(df, headers='keys', tablefmt='presto', floatfmt=".9f"))


if __name__ == '__main__':
    main('/Users/maurice/Desktop/dummy/h2/basis-set-free/hcb=False/spa/noise=0/210720_131901')
