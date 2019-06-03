

from openvqe.hamiltonian.hamiltonian_qc import HamiltonianQC
from openvqe.parameters import  ParametersQC

if __name__ == "__main__":

    parameters_qc = ParametersQC(geometry=" h 0.0 0.0 1.0\n h 0.0 0.0 -1.0", basis_set="sto-3g")
    print(type(parameters_qc))
    #parameters_qc.geometry=" h 0.0 0.0 1.0\n h 0.0 0.0 -1.0"
    #parameters_qc.basis_set='sto-3g'
    print(parameters_qc)
    print(parameters_qc.get_geometry())

    hqc = HamiltonianQC(parameters_qc)
    hqc.greet()
    #hqc.get_Hamiltonian()
    hqc.get_molecular_Hamiltonian()