from openvqe import HamiltonianQC, AnsatzUCC, ParametersQC, ParametersUCC
from openfermionpsi4._psi4_conversion_functions import parse_psi4_ccsd_amplitudes
from openvqe.ansatz.ansatz_ucc import ManyBodyAmplitudes
from openvqe.ansatz.ansatz_ucc import AnsatzUCC
from openvqe.circuit.compiler import compile_trotter_evolution
from openvqe.circuit.simulator_cirq import SimulatorCirq
from openvqe.circuit.simulator_pyquil import SimulatorPyquil
from openvqe.tools.expectation_value_cirq import expectation_value_cirq
import numpy

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
    print("parameters=", hqc.parameters)
    print("The Qubit Hamiltonian is:\n", hqc())

    print("Parse Guess CCSD amplitudes from the PSI4 calculation")
    # get initial amplitudes from psi4
    filename = parameters_qc.filename
    print("filename=", filename + ".out")
    print("n_electrons=", hqc.n_electrons())
    print("n_orbitals=", hqc.n_orbitals())

    # @todo get a functioning guess factory which does not use this parser
    singles, doubles = parse_psi4_ccsd_amplitudes(number_orbitals=hqc.n_orbitals() * 2,
                                                  n_alpha_electrons=hqc.n_electrons() // 2,
                                                  n_beta_electrons=hqc.n_electrons() // 2,
                                                  psi_filename=filename + ".out")

    amplitudes = ManyBodyAmplitudes(one_body=singles, two_body=doubles)

    print("Construct the AnsatzUCC class")

    parameters_ucc = ParametersUCC(backend="cirq", decomposition='trotter', trotter_steps=1)
    print(parameters_ucc)
    ucc = AnsatzUCC(parameters=parameters_ucc, hamiltonian=hqc)
    ucc_operator = ucc(angles=amplitudes)

    abstract_circuit = compile_trotter_evolution(cluster_operator=ucc_operator, steps=1, anti_hermitian=True)


    print("Simulate with Cirq:")
    simulator = SimulatorCirq()


    print("run the circuit:")
    result = simulator.simulate_wavefunction(abstract_circuit=abstract_circuit, returntype=None, initial_state=ucc.initial_state())
    print("resulting state is:")
    print("|psi>=", result)

    energy = expectation_value_cirq(final_state=result.wavefunction,
                                                   hamiltonian=hqc(),
                                                   n_qubits=hqc.n_qubits())

    print("energy = ", energy)

    print("\n\nSame with Pyquil:")
    simulator = SimulatorPyquil()
    result = simulator.simulate_wavefunction(abstract_circuit=abstract_circuit, initial_state=ucc.initial_state())
    print("resulting state is:")
    print("|psi>=", result.result)

    print("\n\nSymbolic computation, just for fun and to test flexibility")
    # replacing the the circuit with something shorter
    from openvqe.circuit.circuit import QCircuit, H, CNOT, X, Ry
    import numpy

    abstract_circuit =  Ry(target=0, angle=0.0685 * numpy.pi) + CNOT(control=0, target=1) + CNOT(control=0,target=2) + CNOT(control=1, target=3) + X(0) + X(1)

    from openvqe.circuit.simulator_symbolic import SimulatorSymbolic

    simulator = SimulatorSymbolic()
    result = simulator.simulate_wavefunction(abstract_circuit=abstract_circuit, initial_state=0)
    print(result)



