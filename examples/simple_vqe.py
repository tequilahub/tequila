"""
Example of a one Qubit VQE
"""
from openvqe.simulator.simulator_qiskit import SimulatorQiskit
from openvqe.circuit import gates
from openvqe.hamiltonian import paulis
from openvqe.objective import Objective
from openvqe.circuit.gradient import grad
from openvqe import numpy


# A simle handwritten GradientDescent optimizer
# used in this example to easily plot the results
class GradientDescent:

    def __init__(self, stepsize=0.1, save_energies=True, save_gradients=True):
        self.stepsize = stepsize
        self._energies = []
        self._gradients = []
        self.save_energies = save_energies
        self.save_gradients = save_gradients

    def __call__(self, angles, energy, gradient):
        if self.save_energies:
            self._energies.append(energy)
        if self.save_gradients:
            self._gradients.append(gradient)
        return [v - self.stepsize * gradient[i] for i, v in enumerate(angles)]

    def plot(self, plot_energies=True, plot_gradients=False, filename: str = None):
        from matplotlib import pyplot as plt
        if plot_energies:
            plt.plot(self._energies, label="E", color='b', marker='o', linestyle='--')
        if plot_gradients:
            gradients = [numpy.asarray(g).dot(numpy.asarray(g)) for g in self._gradients]
            plt.plot(gradients, label="dE", color='r', marker='o', linestyle='--')
        plt.legend()
        if filename is None:
            plt.show()
        else:
            plt.savefig("filename")


# some variables for this example
stepsize = 0.1
initial_angle = 0
max_iter = 100
optimal_angle = numpy.pi / 2
samples = 1000  # number of samples/measurements for the simulator

if __name__ == "__main__":

    # initialize a Hamiltonian
    H = paulis.PX(qubit=0)
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
