from openvqe.simulator import pick_simulator
from openvqe.objective import Objective
from openvqe.circuit.gradient import grad
from openvqe.optimizers.optimizer_base import Optimizer
from openvqe import typing


# A very simple handwritten GradientDescent optimizer for demonstration purposes
class GradientDescent(Optimizer):

    def __init__(self, stepsize=0.1, maxiter=100, samples=None, simulator=None, save_energies=True,
                 save_gradients=True, minimize=True):
        self.stepsize = stepsize
        self._energies = []
        self._gradients = []
        self.save_energies = save_energies
        self.save_gradients = save_gradients
        self.maxiter = maxiter
        self.samples = samples
        self.minimize = minimize
        if simulator is None:
            self.simulator = pick_simulator(samples=samples)
        else:
            self.simulator = simulator

    def update_parameters(self, parameters: typing.Dict[str, float], energy: float, gradient:
        typing.Dict[str, float], *args, **kwargs) -> typing.Dict[str, float]:
        if self.save_energies:
            self._energies.append(energy)
        if self.save_gradients:
            self._gradients.append(gradient)

        updated = dict()
        for k, v in parameters.items():
            if self.minimize:
                updated[k] = v - self.stepsize * gradient[k]
            else:
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

    def __call__(self, objective: Objective, initial_values=None):

        simulator = self.simulator
        if isinstance(simulator, type):
            simulator = simulator()

        angles = initial_values
        if angles is None:
            angles = objective.extract_parameters()
        objective.update_parameters(parameters=angles)

        for iter in range(self.maxiter):

            if self.samples is None:
                E = simulator.simulate_objective(objective=objective)
            else:
                E = simulator.measure_objective(objective=objective, samples=self.samples)

            dO = grad(objective)

            dE = dict()
            for k, dOi in dO.items():
                if self.samples is None:
                    dE[k] = simulator.simulate_objective(objective=dOi)
                else:
                    dE[k] = simulator.measure_objective(objective=dOi, samples=self.samples)

            angles = self.update_parameters(parameters=angles, energy=E, gradient=dE)
            objective.update_parameters(parameters=angles)

        return angles
