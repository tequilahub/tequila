"""
Play around with UCC
This is far from optimal and needs major improvements
"""

from openvqe.hamiltonian import HamiltonianQC
from openvqe.quantumchemistry.qc_base import ParametersQC
from openvqe.ansatz import AnsatzUCC
from openvqe.simulator.simulator_cirq import SimulatorCirq
from openvqe.objective import Objective
from openvqe.circuit.gradient import grad
from openvqe.circuit.exponential_gate import DecompositionFirstOrderTrotter
from openvqe.ansatz import prepare_product_state

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
    H = HamiltonianQC(parameters_qc)
    H.make_hamiltonian()

    # print out the Hamiltonian
    print("The Hamiltonian is:\n", H)

    energies = []
    gradients = []
    for factor in [1.0]:
        # get initial amplitudes from psi4
        amplitudes = H.parse_ccsd_amplitudes()
        amplitudes = factor * amplitudes
        # the only non-zero amplitudes
        # todo simpify the interface
        print("amplitude: ", amplitudes(i=0, a=2, j=1, b=3))
        print("amplitude: ", amplitudes(i=1, a=3, j=0, b=2))
        print("amplitude: ", amplitudes[(2, 0, 3, 1)]) # format is a i b j
        print("amplitude: ", amplitudes[(3, 1, 2, 0)])
        # amplitudes[(2, 0, 3, 1)] = 0.000001
        # amplitudes[(3, 1, 2, 0)] = 0.000001
        # print("amplitude: ", amplitudes(i=0, a=2, j=1, b=3))
        # print("amplitude: ", amplitudes(i=1, a=3, j=0, b=2))
        print("Number of read-in Amplitudes: ", len(amplitudes))

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
        # simulator = SimulatorPyquil()

        print("abstract_circuit\n", abstract_circuit)
        result = simulator.simulate_wavefunction(abstract_circuit=abstract_circuit)
        print("abstract_circuit\n", result.circuit)
        print("resulting state is:")
        print("|psi>=", result.wavefunction)
        print("initial state was: ", H.reference_state(), " = |", H.reference_state().binary, ">")

        O = Objective(observable=H, unitaries=abstract_circuit)
        energy = SimulatorCirq().simulate_objective(objective=O)
        energies.append(energy)
        print("energy = ", energy)

        # Variable initialization does not work yet in the Psi4 interface
        # # Gradients for UCC clearly need improvements (i.e. parameter tracking)
        # dO = grad(O)
        # # we only have one amplitude
        # gradient = 0.0
        # gradients = []
        # for dOi in dO:
        #     value = SimulatorCirq().simulate_objective(objective=dOi)
        #     print("component: ", value)
        #     gradient += value
        #     gradients.append(value)
        #
        # print("gradient = ", gradient)
        # gradients.append(gradient)
