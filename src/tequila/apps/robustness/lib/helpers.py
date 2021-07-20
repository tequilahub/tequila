import sys
from datetime import datetime as dt
import numpy as np
import os
from psi4 import SCFConvergenceError

import tequila as tq


def timestamp():
    return dt.now().strftime('%y%m%d_%H%M%S')


def timestamp_human():
    return dt.now().strftime('%H:%M:%S')


def estimate_ground_state_fidelity(eigenvalues, eigenstates, ansatz, variables, backend, device=None, noise=None, samples=None,
                                   tol=2 * 1.5e-3):
    fidelities = []

    # sort eigenvalues
    ind_sorted = np.argsort(eigenvalues)
    min_eigval = eigenvalues[ind_sorted[0]]
    deg_eigvals = []

    if noise is None:
        for i in ind_sorted:  # takes care of degenerate eigenvalues
            if abs(min_eigval - eigenvalues[i]) > tol:
                break

            deg_eigvals.append(eigenvalues[i])

            # compute fidelity
            exact_wfn = tq.QubitWaveFunction.from_array(eigenstates[:, i])
            ansatz_wfn = tq.simulate(ansatz, backend=backend, variables=variables)
            f = abs(exact_wfn.inner(ansatz_wfn)) ** 2

            fidelities.append(f)

        # if len(fidelities) > 1:
        #     print(f'Warning! found {len(fidelities)} degenerate eigenvalues within tol = {tol};' +
        #           f'\n\teigenvalues:\t{deg_eigvals}\n\tfidelities:\t{fidelities}')

        return max(fidelities)

    for i in ind_sorted:  # takes care of degenerate eigenvalues
        if abs(min_eigval - eigenvalues[i]) > tol:
            break

        deg_eigvals.append(eigenvalues[i])

        # compute fidelity
        exact_wfn = tq.QubitWaveFunction.from_array(eigenstates[:, i])
        exact_wfn = tq.paulis.Projector(wfn=exact_wfn)
        fidelity = tq.ExpectationValue(U=ansatz, H=exact_wfn)
        f = tq.simulate(objective=fidelity, variables=variables, backend=backend, device=device, noise=noise,  # noqa
                        samples=samples)

        fidelities.append(f)

    # if len(fidelities) > 1:
    #     print(f'Warning! found {len(fidelities)} degenerate eigenvalues within tol = {tol};' +
    #           f'\n\teigenvalues:\t{deg_eigvals}\n\tfidelities:\t{fidelities}')

    return max(fidelities)


class Logger:
    def __init__(self, print_fp=None):
        self.terminal = sys.stdout
        self.log_file = "logfile.txt" if print_fp is None else print_fp
        self.encoding = sys.stdout.encoding

    def write(self, message):
        self.terminal.write(message)
        with open(self.log_file, "a") as log:
            log.write(message)

    def flush(self):
        pass


def filtered_dists(data_dir, mol_str):
    """
    returns the list of bond distances corresponding to molecule files in data_dir
    """
    dists = []
    for fn in os.listdir(data_dir):

        if mol_str in fn and "_htensor.npy" in fn:
            if len(fn.split(mol_str)[0]) == 0:
                dist = float(fn.split(f"{mol_str}_")[-1].split("_htensor.npy")[0])
                dists.append(dist)

    return dists


class PauliClique:
    """
    Small Helper Class for cliques of computing Pauli operators
    that are combined to a diagonal operator
    Op = c_i h_i
    where each h_i is a PauliString of Units and Pauli-Z
    class.U transforms into the eigenbasis where Op is diagonal
    """

    def __init__(self, coeff, H, U, n_qubits):
        assert H.is_all_z()
        self.n_qubits = n_qubits
        self.U = U
        self.paulistrings = H.paulistrings
        self.coeff = coeff

    def compute_eigenvalues(self, sort=True):
        """
        Returns
            The eigenvalues of the diagonal operator
        -------
        """
        n_qubits = self.n_qubits
        eig = np.asarray([0.0 for n in range(2 ** n_qubits)], dtype=float)
        for ps in self.paulistrings:
            x = np.asarray([1.0 for n in range(2 ** n_qubits)], dtype=int)
            paulis = [[1, 1]] * n_qubits
            for d in ps.keys():
                try:
                    paulis[d] = [1, -1]
                except:
                    raise Exception("weird {} with len={} with d={}".format(paulis, len(paulis), d))
            for i in range(2 ** n_qubits):
                binary_array = tq.BitString.from_int(integer=i, nbits=n_qubits).array
                for j, k in enumerate(binary_array):
                    x[i] *= paulis[j][k]
            eig += ps.coeff * x
        if sort:
            eig = sorted(eig)
        return eig

    def normalize(self):
        """
        Returns
            Normalized PauliClique with eigenvalues between -1 and 1
        -------
        """

        eig = self.compute_eigenvalues(sort=True)
        lowest = eig[0]
        highest = eig[-1]
        highest_abs = max([abs(lowest), abs(highest)])
        normalized_ps = []
        for ps in self.paulistrings:
            normalized_ps.append(tq.PauliString(coeff=ps.coeff / highest_abs, data=ps._data))

        return PauliClique(coeff=self.coeff * highest_abs, H=tq.QubitHamiltonian.from_paulistrings(normalized_ps),
                           U=self.U, n_qubits=self.n_qubits)

    def naked(self):
        return PauliClique(coeff=1.0, H=self.H, U=self.U, n_qubits=self.n_qubits)

    def __len__(self):
        return len(self.paulistrings)

    @property
    def H(self):
        return tq.QubitHamiltonian.from_paulistrings(self.paulistrings)


def make_paulicliques(H):
    E = tq.ExpectationValue(H=H, U=tq.QCircuit(), optimize_measurements=True)
    result = []
    for clique in E.get_expectationvalues():
        result.append(PauliClique(H=clique.H[0], U=clique.U, coeff=1.0, n_qubits=H.n_qubits))
    return result


def initialize_molecule(r, geometry, name, basis_set, active_orbitals, transformation, n_pno):
    try:
        if basis_set is None:
            return tq.Molecule(geometry=geometry.format(r=r),
                               basis_set=basis_set,
                               transformation=transformation,
                               name=name.format(r=r),
                               n_pno=n_pno)
        else:
            return tq.Molecule(geometry=geometry.format(r=r),
                               basis_set=basis_set,
                               active_orbitals=active_orbitals,
                               transformation=transformation,
                               name=name.format(r=r),
                               n_pno=n_pno)

    except SCFConvergenceError:
        print('WARNING! could not intialize molecule with bond distance {r}'.format(r=r))
        return None


def get_molecule_initializer(geometry, active_orbitals):
    def initializer(r, name, basis_set, transformation, n_pno):
        return initialize_molecule(r, geometry, name, basis_set, active_orbitals, transformation, n_pno)

    return initializer


def print_summary(molecule, hamiltonian, ansatz, use_grouping):

    pauli_terms_with_grouping = tq.ExpectationValue(H=hamiltonian, U=tq.QCircuit(),
                                                    optimize_measurements=True).count_expectationvalues()

    print(f"""---- molecule summary ----
molecule: {molecule}
    
n_orbitals\t: {molecule.n_orbitals}
n_electrons\t: {molecule.n_electrons}
    
Hamiltonian:
num_terms\t: {len(hamiltonian)}
num_qubits\t: {hamiltonian.n_qubits}
pauli_groups\t: {pauli_terms_with_grouping}
use_grouping\t: {use_grouping}
    
Ansatz:
num_params\t: {len(ansatz.extract_variables())}
    """)


def compute_energy_classical(method, molecule):
    try:
        return molecule.compute_energy(method)
    except Exception:  # noqa
        return np.nan
