"""
Non Formatted Psi4 Tutorial
Get a consistent dissociation curve
The geometries are in data/h2o_*.xyz
Psi4 Outputs will also be in data/
"""
import psi4
import tequila as tq
import matplotlib.pyplot as plt
import tequila.simulators.simulator_api

import numpy

def generate_h2o_xyz_files(start=0.75, inc=0.05, steps=30):
    water_geometry = """
     0 1
     O
     H 1 {0}
     H 1 {0} 2 107.6
    """

    files = []
    Rvals = [start + inc * i for i in range(steps)]
    for i, R in enumerate(Rvals):
        mol = psi4.geometry(water_geometry.format(R))
        mol.set_name("R = {}".format(R))
        name = "data/h2o_{}.xyz".format(i)
        files.append(name)
        with open(name, "w") as f:
            f.write(mol.save_string_xyz_file())

    return files

if __name__ == "__main__":

    # generate files/get list of all generated files
    files = generate_h2o_xyz_files()

    # define parameters
    threads = 4 # psi4 threads
    optimizer_method = "cobyla"
    active = {"A1": [2, 3], "B1": [0], "B2": [0, 1]} # active space for VQE
    basis_set = "sto-3g" # basis set
    transformation = "jordan-wigner" # qubit transformation
    guess_wfn = None # guess_wfn to ensure convergence along the dissociation curve
    options = {
        'reference': 'rhf',
        'df_scf_guess': 'False',
        'scf_type': 'direct',
        'e_convergence': '6',
        'guess': 'read',
        "frozen_docc": [2, 0, 0, 0]
    } # psi4 options

    ref_methods = ["hf", "mp2", "mp3", "ccsd", "fci"] # compute reference values with psi4

    ref_energies = {m: [] for m in ref_methods}
    energies = []

    # compute stuff
    for i, file in enumerate(files):
        if i < 10: break
        print("computing point {}".format(i))
        mol = tq.chemistry.Molecule(geometry=file, basis_set=basis_set, transformation=transformation, threads=threads,
                                    guess_wfn=guess_wfn, options=options, active_orbitals=active)
        guess_wfn = mol.ref_wfn

        for m in ref_methods:
            print("computing ", m)
            try:
                ref_energies[m] += [mol.compute_energy(method=m, guess_wfn=mol.ref_wfn)]
            except:
                ref_energies[m] += [None]

        H = mol.make_hamiltonian()
        Uhf = mol.prepare_reference()

        # stupid excuse for an vqe, but it tests consistency
        # just evaluates the hartree fock energy
        U = Uhf
        hf = tq.simulate(tq.ExpectationValue(U=U, H=H))
        energies += [hf]

    plt.plot(energies, label="QHF", marker="o", linestyle="--")
    for m in ref_methods:
        values = ref_energies[m]
        plt.plot(values, label=str(m))
    plt.legend()
    plt.show()
