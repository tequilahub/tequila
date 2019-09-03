from openvqe.hamiltonian import HamiltonianQC, ParametersQC
from openvqe.ansatz import AnsatzUCC
from openvqe.circuit.compiler import compile_trotter_evolution
from openvqe.simulator.simulator_cirq import SimulatorCirq
from openvqe.simulator.simulator_pyquil import SimulatorPyquil
from openvqe.tools.expectation_value_cirq import expectation_value_cirq
from openvqe.circuit.gates import Ry, CNOT, X
from openvqe.objective import Objective
from openvqe.circuit.gradient import grad
from openvqe.circuit import QCircuit
import numpy

if __name__ == "__main__":
    print("Demo for closed-shell UCC with psi4-CCSD trial state and first order Trotter decomposition")

    print("First get the Hamiltonian:")
    parameters_qc = ParametersQC(geometry="data/h2.xyz", basis_set="sto-3g")
    parameters_qc.transformation = "JW"
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

    amplitudes = hqc.parse_ccsd_amplitudes()
    print("# of amplitudes: ",len(amplitudes))
    print("amplitudes:\n", amplitudes)

    print("Construct the AnsatzUCC class")

    ucc = AnsatzUCC()
    ucc_operator = ucc(angles=amplitudes)
    print("# ucc terms: ", len(ucc_operator.terms))
    print("# ucc terms:\n", ucc_operator.terms )

    abstract_circuit = compile_trotter_evolution(cluster_operator=ucc_operator, steps=1, anti_hermitian=True)

    print("Simulate with Cirq:")
    simulator = SimulatorCirq()

    print("run the circuit:")
    result = simulator.simulate_wavefunction(abstract_circuit=abstract_circuit, returntype=None,
                                             initial_state=ucc.initial_state(hqc))
    print("resulting state is:")
    print("|psi>=", result)
    print("circuit\n", result.circuit)
    print("initial state was: ", ucc.initial_state(hamiltonian=hqc))

    energy = expectation_value_cirq(final_state=result.wavefunction,
                                    hamiltonian=hqc(),
                                    n_qubits=hqc.n_qubits())

    print("energy = ", energy)

    O = Objective(observable=hqc, unitaries=abstract_circuit)
    energy = SimulatorCirq().expectation_value(objective=O, initial_state=ucc.initial_state(hqc))

    print("energy = ", energy)

    dO = grad(O)
    # we only have one amplitude
    gradient = 0.0
    for dOi in dO:
        value = SimulatorCirq().expectation_value(objective=dOi, initial_state=ucc.initial_state(hqc))
        gradient += value
    print("gradient = ", gradient)

    exit()
    print("\n\nSame with Pyquil:")
    simulator = SimulatorPyquil()
    result = simulator.simulate_wavefunction(abstract_circuit=abstract_circuit, initial_state=ucc.initial_state(hqc))
    print("resulting state is:")
    print("|psi>=", result.result)

    print("\n\nSymbolic computation, just for fun and to test flexibility")

    abstract_circuit = Ry(target=0, angle=0.0685 * numpy.pi) \
                       * CNOT(control=0, target=1) * CNOT(control=0,target=2) \
                       * CNOT(control=1, target=3) * X(0) * X(1)

    from openvqe.simulator.simulator_symbolic import SimulatorSymbolic

    simulator = SimulatorSymbolic()
    result = simulator.simulate_wavefunction(abstract_circuit=abstract_circuit, initial_state=0)
    print(result)
