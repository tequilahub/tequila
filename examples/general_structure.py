
from openvqe.hamiltonian import HamiltonianQC, ParametersQC
from openvqe import OpenVQEException

if __name__ == "__main__":

    print("Demo for QC-Hamiltonian: Get JW and BK Transformed Qubit Hamiltonians:\n")

    parameters_qc = ParametersQC(geometry="data/h2.xyz", basis_set="sto-3g")
    parameters_qc.transformation="JW"
    hqc = HamiltonianQC(parameters_qc)
    hqc.greet()
    print("\n\nJordan-Wigner, parameters are:\n", parameters_qc, "\n\n")
    print("HMOL:\n", hqc.get_fermionic_hamiltonian())
    print("HQUBIT:\n",hqc())

    parameters_qc.transformation="BK"
    print("\n\nBravyi-Kitaev, parameters are:\n", parameters_qc, "\n\n")
    hqc = HamiltonianQC(parameters_qc)
    hqc.greet()
    print("HQUBIT:\n",hqc())

    print("Same Thing but different initialization style: Agnostic of parameter-type\n")
    hqc = HamiltonianQC()
    hqc.parameters.geometry = "data/h2.xyz"
    hqc.parameters.basis_set = "sto-3g"
    hqc.parameters.transformation = "JW"
    print("current parameters are:\n", hqc.parameters)
    print("HMOL:\n", hqc.get_fermionic_hamiltonian())
    print("HQUBIT:\n",hqc())




    print("\nException Handling:\n")
    try:
        print("Will initialize a parameter to an unknown command")
        parameters_qc = ParametersQC(geometry=" h 0.0 0.0 1.0\n h 0.0 0.0 -1.0", basis_set="sto-3g")
        parameters_qc.transformation = "lalala"
        hqc = HamiltonianQC(parameters_qc)
        H=hqc()
    except OpenVQEException as e:
        print("You chose a weird parameter")
        print(e)

    try:
        print("\nWill call a method which is not implemented")
        from openvqe.hamiltonian.hamiltonian_hubbard import HamiltonianHubbard
        hqc = HamiltonianHubbard()
        H = hqc.get_fermionic_hamiltonian()
    except NotImplementedError as e:
        print("Hubbard not yet implemented")
        print(e)


