"""
Example of a simple one Qubit VQE optimized with simple GradientDescent
"""

from openvqe.simulator.simulator_qiskit import SimulatorQiskit
from openvqe.simulator.simulator_cirq import SimulatorCirq
from openvqe.circuit import gates
from openvqe.hamiltonian import paulis
from openvqe.objective import Objective
from openvqe.circuit.gradient import grad
from openvqe import numpy
from openvqe.optimizers import GradientDescent #Doesn't do much, but is convenient for the plots in this example

# some variables to play around with for this example
stepsize = 0.1
initial_angle = 0
max_iter = 100
optimal_angle = numpy.pi / 2
samples = 1000  # number of samples/measurements for the simulator, will have no effect if full wavefunction simulation is used
use_full_wavefunction_simulation = True

if __name__ == "__main__":

    # initialize a Hamiltonian
    H = paulis.X(qubit=0)
    # initialize the initial angles
    angles = [initial_angle]
    # initialize the simulator, for full wavefunction simulation use cirq or pyquil
    simulator = SimulatorQiskit()
    if use_full_wavefunction_simulation:
        simulator = SimulatorCirq()

    # initialize the optimizer
    optimizer = GradientDescent(stepsize=stepsize)

    # do the optimization
    E = 100.0
    for iter in range(max_iter):
        print("angle=", angles[0])

        # initialize an Ansatz for the Wavefunction
        U = gates.Ry(target=0, angle=angles[0])

        # initialize an objective and compute Energy and gradient
        O = Objective(unitaries=U, observable=H)

        # compute the energy
        if use_full_wavefunction_simulation:
            Enew = simulator.simulate_objective(objective=O)
        else:
            Enew = simulator.measure_objective(objective=O, samples=samples)


        diff = Enew - E
        E = Enew
        print("E   =", E)
        print("diff=", diff)

        # compute the gradient, more sensitive to number of samples than energy
        dO = grad(O)
        dE = None
        if use_full_wavefunction_simulation:
            dE = [simulator.simulate_objective(objective=dOi) for dOi in dO]
        else:
            dE = [simulator.measure_objective(objective=dOi, samples=samples) for dOi in dO]
        print("dE  =", dE)

        # update the angle
        angles = optimizer(angles=angles, energy=E, gradient=dE)

    print("angle after ", max_iter, "iterations : ", angles[0])
    print("optimal angle : ", optimal_angle)

    # plot progress
    optimizer.plot(plot_energies=True, plot_gradients=True, filename=None)


