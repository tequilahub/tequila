"""
Non Formatted Psi4 Tutorial
Get a consistent dissociation curve
The geometries are in data/h2o_*.xyz
Psi4 Outputs will also be in data/
"""
import psi4
import tequila as tq
import matplotlib.pyplot as plt


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
    threads = 4
    active = {"A1": [2, 3], "B1": [0], "B2": [1]}
    basis_set = "sto-3g"
    transformation = "jordan-wigner"
    guess_wfn = None
    hf_energies = []
    options = {
        'reference': 'rhf',
        'df_scf_guess': 'False',
        'scf_type': 'direct',
        'e_convergence': '6',
        'guess': 'read',
        'maxiter': 200
    }
    ref_methods = ["hf", "mp2", "mp3", "fci"]
    ref_energies = {m: [] for m in ref_methods}
    energies = []

    # compute stuff
    for i, file in enumerate(files):
        print("computing point {}".format(i))
        mol = tq.chemistry.Molecule(geometry=file, basis_set=basis_set, transformation=transformation, threads=threads,
                                    guess_wfn=guess_wfn, options=options)
        hf_energies.append(mol.energies["hf"])
        guess_wfn = mol

        for m in ref_methods:
            ref_energies[m] += [mol.compute_energy(method=m, guess_wfn=mol.ref_wfn, options=options)]

        H = mol.make_active_space_hamiltonian(active_orbitals=active)

        Uhf = mol.prepare_reference(active_orbitals=active)

        U = Uhf
        hf = tq.simulate(tq.ExpectationValue(U=U, H=H))
        energies += [hf]

    plt.plot(energies, label="hf-test", marker="o", linestyle="--")
    for m in ref_methods:
        values = ref_energies[m]
        plt.plot(values, label=str(m), linestyle="--")
    plt.legend()
    plt.show()
