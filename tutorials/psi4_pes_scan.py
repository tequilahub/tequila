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
    Rvals = [start + inc*i for i in range(steps)]
    for i,R in enumerate(Rvals):
        mol = psi4.geometry(water_geometry.format(R))
        mol.set_name("R = {}".format(R))
        name = "data/h2o_{}.xyz".format(i)
        files.append(name)
        print(i, " ", R)
        with open(name, "w") as f:
            f.write(mol.save_string_xyz_file())

    return files

if __name__ == "__main__":

    # generate files/get list of all generated files
    files = generate_h2o_xyz_files()

    # create tequila objects
    # use previous result as guess
    threads = 1
    basis_set = "sto-3g"
    transformation = "jordan-wigner"
    guess_wfn = None
    hf_energies = []
    hf_energies_from_scratch = []
    options = {
        'reference': 'rhf',
        'df_scf_guess': 'False',
        'scf_type': 'direct',
        'e_convergence': '8',
        'guess': 'read'
    }
    for i,file in enumerate(files):
        mol = tq.chemistry.Molecule(geometry=file, basis_set=basis_set, transformation=transformation, threads=threads, guess_wfn=guess_wfn, options=options)
        hf_energies.append(mol.energies["hf"])
        guess_wfn = mol

    plt.plot(hf_energies, label="guess=read", marker = "o", linestyle="--")
    plt.legend()
    plt.show()




