from openvqe.simulator import pick_simulator
from openvqe.objective import Objective
from openvqe.circuit.gradient import grad
from openvqe.optimizers.optimizer_base import Optimizer
from openvqe import typing


# A very simple handwritten GradientDescent optimizer for demonstration purposes
class GradientDescent(Optimizer):

    def __init__(self, stepsize=0.1, maxiter=100, samples=None, simulator=None, save_history=True, minimize=True):
        self.stepsize = stepsize
        self.minimize = minimize
        super().__init__(simulator=simulator, maxiter=maxiter, samples=samples, save_history=save_history)

    def update_parameters(self, parameters: typing.Dict[str, float], energy: float, gradient:
    typing.Dict[str, float], *args, **kwargs) -> typing.Dict[str, float]:

        updated = dict()
        for k, v in parameters.items():
            if self.minimize:
                updated[k] = v - self.stepsize * gradient[k]
            else:
                updated[k] = v + self.stepsize * gradient[k]
        return updated

    def __call__(self, objective: Objective, initial_values=None):

        simulator = self.initialize_simulator(samples=self.samples)
        simulator.set_compile_flag(False)

        angles = initial_values
        if angles is None:
            angles = objective.extract_parameters()
        objective.update_parameters(parameters=angles)

        dO = grad(objective)
        O = objective.to_backend(simulator)
        for key in dO.keys():
            dO[key] = dO[key].to_backend(simulator)

        for iter in range(self.maxiter):
            if self.samples is None:
                E = simulator.simulate_objective(objective=O)
            else:
                E = simulator.measure_objective(objective=O, samples=self.samples)

            dE = dict()
            for k, dOi in dO.items():
                if self.samples is None:
                    dE[k] = simulator.simulate_objective(objective=dOi)
                else:
                    dE[k] = simulator.measure_objective(objective=dOi, samples=self.samples)
            angles = self.update_parameters(parameters=angles, energy=E, gradient=dE)

            O.update_parameters(parameters=angles)
            for dOi in dO.values():
                dOi.update_parameters(parameters=angles)

            if self.save_history:
                self.history.energies.append(E)
                self.history.gradients.append(dE)
                self.history.angles.append(angles)
            objective.update_parameters(parameters=angles)


        return E, angles
