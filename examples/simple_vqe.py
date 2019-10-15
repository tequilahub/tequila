"""
Example of a one Qubit VQE
"""
from openvqe.simulator.simulator_qiskit import SimulatorQiskit
from openvqe.circuit import gates
from openvqe.hamiltonian import paulis
from openvqe.objective import Objective
from openvqe.circuit.gradient import grad
from openvqe import numpy
from openvqe.optimizers import GradientDescent

# some variables for this example
stepsize = 0.1
initial_angle = 0
max_iter = 100
optimal_angle = numpy.pi / 2
samples = 1000  # number of samples/measurements for the simulator

if __name__ == "__main__":

    # initialize a Hamiltonian
    H = paulis.X(qubit=0)
    # initialize the initial angle
    angles = [initial_angle]
    # initialize the simulator
    simulator = SimulatorQiskit()
    # initialize the optimizer
    optimizer = GradientDescent(stepsize=stepsize)

    # do the optimization
    for iter in range(max_iter):
        print("angle=", angles[0])

        # initialize an Ansatz for the Wavefunction
        U = gates.Ry(target=0, angle=angles[0])

        # initialize an objective and compute Energy and gradient
        O = Objective(unitaries=U, observable=H)

        # compute the energy
        E = simulator.measure_objective(objective=O, samples=samples)
        print("E =", E)

        # compute the gradient, more sensitive to number of samples than energy
        dO = grad(O)
        dE = [simulator.measure_objective(objective=dOi, samples=samples) for dOi in dO]
        print("dE=", dE)

        # update the angle
        angles = optimizer(angles=angles, energy=E, gradient=dE)

    print("angle after ", max_iter, "iterations : ", angles[0])
    print("optimal angle : ", optimal_angle)

    # plot progress
    optimizer.plot(plot_energies=True, plot_gradients=True)
