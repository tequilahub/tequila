"""
Play around with UCC
This is far from optimal and needs major improvements
"""

from tequila.simulators import pick_simulator
from tequila.objective import Objective
from tequila.circuit.exponential_gate import DecompositionFirstOrderTrotter
from tequila.optimizers import  GradientDescent

from matplotlib import pyplot as plt

# you need psi4 to be installed for this example
import tequila.quantumchemistry as qc

if not qc.has_pyscf:
    raise Exception("You need PySCF for this examples: Easy install with conda install -c pyscf pyscf")

# initialize your favorite Simulator
samples = None # none means full wavefunction simulation
simulator = pick_simulator(samples=samples)

if __name__ == "__main__":

    raise Exception("Under Development")
    # initialize the QuantumChemistry Module
    qc_param = qc.ParametersQC(geometry="data/h2.xyz", basis_set="sto-3g")
    pyscf_interface = qc.QuantumChemistryPySCF(parameters=qc_param, transformation="jordan-wigner")

    # get the Hamiltonian in QubitForm
    H = pyscf_interface.make_hamiltonian()

    # configure the trotterization
    trotter = DecompositionFirstOrderTrotter(steps=1)

    # get the UCC circuit
    U = pyscf_interface.make_uccsd_ansatz(decomposition=trotter, initial_amplitudes="ccsd", include_reference_ansatz=True)
    print("CCSD Parameters:\n", U.extract_parameters())
    # make an objective
    O = Objective(observable=H, unitaries=U)

    angles = O.extract_parameters()

    # compute energy
    E = simulator().simulate_objective(objective=O)

    print("Energy = ", E)


    # optimize with the scipy interface


    # overwrite the initial amplitudes to be zero
    initial_amplitudes = qc.Amplitudes(data={(2, 0, 3, 1): 0.0, (3, 1, 2, 0): 0.0 })
    init = dict()
    for k, v in initial_amplitudes.items():
        # this sucks and should be more convenient
        init[str(k)] = v
    #E, angles, res = scipy_optimizers.minimize(O, return_all=True, simulators=simulators, samples=samples, initial_values=init)
    optimizer = GradientDescent(samples=samples, simulator=simulator, stepsize=0.1, maxiter=10, minimize=True)
    angles = optimizer(objective=O, initial_values=init)  # take the current values of the circuit as the initial ones, alternativ use initial_values={"a": 2.0, "b": 2.0}
    E = optimizer.energies[-1]

    print("final angles are:\n", angles)
    print("final energy = ", E)

    # plot results
    optimizer.plot(plot_energies=True, plot_gradients=None)
    optimizer.plot(plot_energies=False, plot_gradients=True)  # plot only a specific gradient with plot_gradients=["key"]

    #plt.plot(res.evals)
    #plt.show()
