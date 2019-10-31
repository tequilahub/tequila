from openvqe.simulator import pick_simulator
from openvqe.objective import Objective
from openvqe.circuit.gradient import grad


# A very simple handwritten GradientDescent optimizer for demonstration purposes
class GradientDescent:

    def __init__(self, stepsize=0.1, maxiter=100, samples=None, simulator=None, save_energies=True,
                 save_gradients=True):
        self.stepsize = stepsize
        self._energies = []
        self._gradients = []
        self.save_energies = save_energies
        self.save_gradients = save_gradients
        self.maxiter = maxiter
        self.samples = samples
        if simulator is None:
            self.simulator = pick_simulator(samples=samples)
        else:
            self.simulator = simulator

    def __call__(self, angles: dict, energy: float, gradient: dict):

        if self.save_energies:
            self._energies.append(energy)
        if self.save_gradients:
            self._gradients.append(gradient)

        updated = dict()
        for k, v in angles.items():
            updated[k] = v + self.stepsize * gradient[k]

        return updated

    def plot(self, plot_energies=True, plot_gradients: list = None, filename: str = None):
        from matplotlib import pyplot as plt
        if plot_energies:
            plt.plot(self._energies, label="E", color='b', marker='o', linestyle='--')
        if plot_gradients is not None:
            if plot_gradients is True:
                plot_gradients = [k for k in self._gradients[-1].keys()]
            if not hasattr(plot_gradients, "__len__"):
                plot_gradients = [plot_gradients]
            for name in plot_gradients:
                grad = [i[name] for i in self._gradients]
                plt.plot(grad, label="dE_" + name, marker='o', linestyle='--')
        plt.legend()
        if filename is None:
            plt.show()
        else:
            plt.savefig("filename")

    def solve(self, U, H, angles=None):

        simulator = self.simulator()

        if angles is None:
            angles = U.extract_parameters()

        for iter in range(self.maxiter):

            O = Objective(unitaries=U, observable=H)
            if self.samples is None:
                E = simulator.simulate_objective(objective=O)
            else:
                E = simulator.measure_objective(objective=O, samples=self.samples)

            dO = grad(O)

            dE = dict()
            for k, dOi in dO.items():
                if self.samples is None:
                    dE[k] = simulator.simulate_objective(objective=dOi)
                else:
                    dE[k] = simulator.measure_objective(objective=dOi, samples=self.samples)

            angles = self(angles=angles, energy=E, gradient=dE)
            U.update_parameters(parameters=angles)
