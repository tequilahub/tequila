"""
Play around with UCC
This is far from optimal and needs major improvements
"""

from openvqe.simulator import pick_simulator
from openvqe.objective import Objective
from openvqe.optimizers import scipy_optimizers, GradientDescent

from matplotlib import pyplot as plt

# you need psi4 to be installed for this example
import openvqe.quantumchemistry as qc
if not qc.has_psi4:
    raise Exception("You need Psi4 for this examples: Easy install with conda install psi4 -c psi4")
# pyscf is coming soon

# initialize your favorite Simulator
samples = None # none means full wavefunction simulation
simulator = pick_simulator(samples=samples)

if __name__ == "__main__":

    # initialize the QuantumChemistry Module
    qc_param = qc.ParametersQC(geometry="data/h2.xyz", basis_set="sto-3g")
    psi4_interface = qc.QuantumChemistryPsi4(parameters=qc_param, transformation="jordan-wigner")

    # get the Hamiltonian in QubitForm
    H = psi4_interface.make_hamiltonian()

    # configure the trotterization
    #trotter = DecompositionFirstOrderTrotter(steps=1)

    # get the UCC circuit
    U = psi4_interface.make_uccsd_ansatz(trotter_steps=1, initial_amplitudes="mp2", include_reference_ansatz=True)

    print(U)

    # make an objective
    O = Objective(observable=H, unitaries=U)

    angles = O.extract_parameters()
    print(angles)

    # compute energy
    E = simulator().simulate_objective(objective=O)

    print("Energy = ", E)
    print("CCSD Parameters:\n", U.extract_parameters())

    # overwrite the initial amplitudes to be zero
    initial_amplitudes = qc.Amplitudes(data={(2, 0, 3, 1): 0.0, (3, 1, 2, 0): 0.0 })
    # overwrite the initial amplitudes to be MP2
    #initial_amplitudes = psi4_interface.compute_mp2_amplitudes()

    print("initial amplitudes:\n", initial_amplitudes)

    optimizer = GradientDescent(samples=samples, simulator=simulator, stepsize=0.1, maxiter=10, minimize=True)
    angles = optimizer(objective=O, initial_values=initial_amplitudes.export_parameter_dictionary())

    E = optimizer.energies[-1]

    print("final angles are:\n", angles)
    print("final energy = ", E)

    # plot results
    optimizer.plot(plot_energies=True, plot_gradients=None)
    optimizer.plot(plot_energies=False, plot_gradients=True)  # plot only a specific gradient with plot_gradients=["key"]

