from openvqe import HamiltonianQC, ParametersQC, ParametersHamiltonian, OVQEException, ParametersUCC
from openfermionpsi4._psi4_conversion_functions import parse_psi4_ccsd_amplitudes
from openvqe.ansatz.ansatz_ucc import ManyBodyAmplitudes
from openvqe.ansatz.ansatz_ucc import AnsatzUCC
import openvqe
import cirq

if __name__ == "__main__":
    print("Demo for closed-shell UCC with psi4-CCSD trial state and first order Trotter decomposition")

    print("First get the Hamiltonian:")
    parameters_qc = ParametersQC(geometry="data/h2.xyz", basis_set="sto-3g")
    parameters_qc.transformation = "JW"
    parameters_qc.psi4.delete_output = False
    parameters_qc.psi4.delete_input = False
    parameters_qc.psi4.run_ccsd = True
    parameters_qc.filename = "psi4"
    hqc = HamiltonianQC(parameters_qc)
    print("The Qubit Hamiltonian is:\n", hqc())

    print("Parse Guess CCSD amplitudes from the PSI4 calculation")
    # get initial amplitudes from psi4
    filename = parameters_qc.filename
    print("filename=", filename+".out")
    print("n_electrons=", hqc.n_electrons())
    print("n_orbitals=", hqc.n_orbitals())

    # @todo get a functioning guess factory which does not use this parser
    singles, doubles = parse_psi4_ccsd_amplitudes(number_orbitals=hqc.n_orbitals()*2, n_alpha_electrons=hqc.n_electrons()//2,
                                                  n_beta_electrons=hqc.n_electrons()//2, psi_filename=filename + ".out")

    # long output
    #print("Singles Amplitudes:\n", singles)
    #print("Doubles Amplitudes:\n", doubles)

    amplitudes = ManyBodyAmplitudes(one_body=singles, two_body=doubles)

    print("Construct the AnsatzUCC class")

    parameters_ucc = ParametersUCC(backend="cirq", decomposition='trotter', trotter_steps=1)
    print(parameters_ucc)
    ucc = AnsatzUCC(parameters=parameters_ucc, hamiltonian=hqc)
    circuit = ucc(angles=amplitudes)

    print("created the following circuit:")
    print(circuit)


    print("run the circuit:")
    simulator = cirq.Simulator()
    result = simulator.simulate(program=circuit)
    print("resulting state is:")
    print("|psi>=",result.dirac_notation(decimals=3))


    # print("Testing Exception Handling:")
    # print("Wrong initialization")
    # wrong=openvqe.HamiltonianBase(parameters=openvqe.parameters.ParametersBase())
    # ucc2 = AnsatzUCC(parameters=parameters_ucc, hamiltonian=wrong)