
from openvqe import HamiltonianQC, ParametersQC, ParametersHamiltonian, ParameterError

if __name__ == "__main__":

    parameters_qc = ParametersQC(geometry=" h 0.0 0.0 1.0\n h 0.0 0.0 -1.0", basis_set="sto-3g")

    parameters_qc.transformation="JW"
    hqc = HamiltonianQC(parameters_qc)
    hqc.greet()
    print("\n\nJordan-Wigner, parameters are:\n", parameters_qc, "\n\n")
    print("HMOL:\n",hqc.get_hamiltonian())
    print("HQUBIT:\n",hqc())

    parameters_qc.transformation="BK"
    print("\n\nBravyi-Kitaev, parameters are:\n", parameters_qc, "\n\n")
    hqc = HamiltonianQC(parameters_qc)
    hqc.greet()
    print("HQUBIT:\n",hqc())


    molecule = HamiltonianQC.get_molecule(parameters_qc)
    print("\n\nMOLECULE:\n", molecule)


    print("\nException Handling:\n")
    try:
        parameters_qc = ParametersQC(geometry=" h 0.0 0.0 1.0\n h 0.0 0.0 -1.0", basis_set="sto-3g")
        parameters_qc.transformation = "lalala"
        hqc = HamiltonianQC(parameters_qc)
        H=hqc()
    except ParameterError:
        print("You chose a weird parameter")

    try:
        from openvqe.hamiltonian.hamiltonian_hubbard import HamiltonianHubbard
        hqc = HamiltonianHubbard(ParametersHamiltonian())
        H = hqc.get_hamiltonian()
    except NotImplementedError:
        print("Hubbard not yet implemented")

