from openvqe.hamiltonian import HamiltonianPsi4, ParametersQC
from openvqe.ansatz import AnsatzUCC
from openvqe.simulator.simulator_cirq import SimulatorCirq
from openvqe.simulator.simulator_pyquil import SimulatorPyquil
from openvqe.circuit.gates import Ry, CNOT, X
from openvqe.objective import Objective
from openvqe.circuit.gradient import grad
from openvqe.circuit import gates
from openvqe.circuit.exponential_gate import DecompositionFirstOrderTrotter
from openvqe.ansatz import prepare_product_state
from openvqe import numpy

if __name__ == "__main__":
    print("Demo for closed-shell UCC with psi4-CCSD trial state and first order Trotter decomposition")

    # Configure Psi4
    parameters_qc = ParametersQC(geometry="data/h2.xyz", basis_set="sto-3g")
    parameters_qc.transformation = "Jordan-Wigner"
    # we will use already converged CCSD amplitudes amplitudes in this example
    # so psi4 should run ccsd
    parameters_qc.psi4.run_ccsd = True
    # psi4 outputfile
    parameters_qc.filename = "psi4"

    # Initialize the Hamiltonian
    # This will call OpenFermion as well as Psi4
    H = HamiltonianPsi4(parameters_qc)
    H.initialize_hamiltonian()

    # print out the Hamiltonian
    print("The Hamiltonian is:\n", H)


    energies = []
    gradients = []
    for factor in [0.2,0.4,0.6,0.8,1.0]:
        # get initial amplitudes from psi4
        amplitudes = H.parse_ccsd_amplitudes()
        amplitudes = factor*amplitudes
        print("Number of read-in Amplitudes: ",len(amplitudes))

        # Prepare Reference State
        cref = prepare_product_state(H.reference_state())

        # Construct the UCC ansatz
        ucc = AnsatzUCC(decomposition=DecompositionFirstOrderTrotter(steps=1, threshold=0.0))
        cucc = ucc(angles=amplitudes)

        # assemble
        print("cref=", cucc, "\n")
        abstract_circuit = cref + cucc

        # Initialize the Simulator
        simulator = SimulatorCirq()
        #simulator = SimulatorPyquil()

        print("abstract_circuit\n", abstract_circuit)
        result = simulator.simulate_wavefunction(abstract_circuit=abstract_circuit)
        print("resulting state is:")
        print("|psi>=", result.wavefunction)
        print("initial state was: ", ucc.initial_state(hamiltonian=H))

        O = Objective(observable=H, unitaries=abstract_circuit)
        energy = SimulatorCirq().expectation_value(objective=O)
        energies.append(energy)
        print("energy = ", energy)



        dO = grad(O)
        # we only have one amplitude
        gradient = 0.0
        for dOi in dO:
            value = SimulatorCirq().expectation_value(objective=dOi)
            print("component: ", value)
            gradient += value
        print("gradient = ", gradient)
        gradients.append(gradient)


    print(energies)
    print(gradients)

    exit()

    print("\n\nSame with Pyquil:")
    simulator = SimulatorPyquil()
    result = simulator.simulate_wavefunction(abstract_circuit=abstract_circuit, initial_state=ucc.initial_state(H))
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


    simulator = SimulatorCirq()
    asd = gates.X(0)+gates.X(1)+abstract_circuit + gates.Measurement(target=[0,1,2,3])
    test = simulator.run(abstract_circuit=asd, samples=100)
    print("counts=", test.counts)