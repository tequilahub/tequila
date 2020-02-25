"""
Play around with UCC
This is far from optimal and needs major improvements
"""
import tequila.simulators.simulator_api
from tequila import pick_backend
from tequila.objective import Objective
from tequila.optimizers.optimizer_scipy import minimize

# you need psi4 to be installed for this example
import tequila.quantumchemistry as qc
if not qc.has_psi4:
    raise Exception("You need Psi4 for this examples: Easy install with conda install psi4 -c psi4")
# pyscf is coming soon

# initialize your favorite Simulator
samples = None# none means full wavefunction simulation
simulator = pick_backend(samples=samples)
from tequila.simulators.simulator_cirq import SimulatorCirq
simulator = SimulatorCirq

if __name__ == "__main__":

    # initialize the QuantumChemistry Module
    qc_param = qc.ParametersQC(geometry="data/h2.xyz", basis_set="sto-3g")
    psi4_interface = qc.QuantumChemistryPsi4(parameters=qc_param, transformation="jordan-wigner")

    # get the Hamiltonian in QubitForm
    H = psi4_interface.make_hamiltonian()

    # configure the trotterization
    #trotter = DecompositionFirstOrderTrotter(steps=1)

    # get the UCC circuit
    U = psi4_interface.make_uccsd_ansatz(trotter_steps=1, initial_amplitudes="ccsd", include_reference_ansatz=True)

    print(U)

    # make an objective
    O = Objective.ExpectationValue(H=H,U=U)

    angles = O.extract_variables()
    print(angles)

    # compute full energy
    E = tequila.simulators.simulator_api.simulate_objective(O)

    print("Energy = ", E)
    print("CCSD Parameters:\n", U.extract_variables())

    # overwrite the initial amplitudes to be zero
    initial_amplitudes = qc.OldAmplitudes(data={(2, 0, 3, 1): 0.0, (3, 1, 2, 0): 0.0})
    # overwrite the initial amplitudes to be MP2
    #initial_amplitudes = psi4_interface.compute_mp2_amplitudes()

    print("initial amplitudes:\n", initial_amplitudes)

    result = minimize(objective=O, initial_values=initial_amplitudes.export_parameter_dictionary(), samples=samples, backend=simulator, maxiter=10, method="TNC")

    print("final angles are:\n", angles)
    print("final energy = ", result.energy)

    # plot results
    result.history.plot(property='energies')
    result.history.plot(property='gradients')
    result.history.plot(property='angles')