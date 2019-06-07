
from openvqe import HamiltonianQC, ParametersQC, ParametersHamiltonian, OvqeParameterError, OvqeException

if __name__ == "__main__":

    print("Demo for QC-Hamiltonian: Get JW and BK Transformed Qubit Hamiltonians:\n")

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


    print("\nException Handling:\n")
    try:
        print("Will initialize a parameter to an unknown command")
        parameters_qc = ParametersQC(geometry=" h 0.0 0.0 1.0\n h 0.0 0.0 -1.0", basis_set="sto-3g")
        parameters_qc.transformation = "lalala"
        hqc = HamiltonianQC(parameters_qc)
        H=hqc()
    except OvqeException as e:
        print("You chose a weird parameter")
        print(e)

    try:
        print("\nWill call a method which is not implemented")
        from openvqe.hamiltonian.hamiltonian_hubbard import HamiltonianHubbard
        hqc = HamiltonianHubbard(ParametersHamiltonian())
        H = hqc.get_hamiltonian()
    except NotImplementedError as e:
        print("Hubbard not yet implemented")
        print(e)

    try:
        print("\nWill overwrite parameters to wrong type after correct initialization")
        parameters_qc = ParametersQC(geometry=" h 0.0 0.0 1.0\n h 0.0 0.0 -1.0", basis_set="sto-3g")
        parameters_qc.transformation = "JW"
        hqc = HamiltonianQC(parameters_qc)
        # now overwrite parameters after successful initialization
        parameters_h = ParametersHamiltonian()
        hqc.parameters = parameters_h
        H=hqc() # raises exceition because the self.verify function is called here)
    except OvqeException as e:
        print("catched OpenVQEException")
        print(e)

