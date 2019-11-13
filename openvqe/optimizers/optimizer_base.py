"""
BaseCalss for Optimizers
Suggestion, feel free to propose new things/changes
"""

from openvqe import OpenVQEException
from openvqe import typing, numbers
from openvqe.objective import Objective
from openvqe.simulator import pick_simulator
from dataclasses import dataclass, field

class OpenVQEOptimizerException(OpenVQEException):
    pass

@dataclass
class OptimizerHistory:

    energies: typing.List[numbers.Real] = field(default_factory=list)
    gradients: typing.List[typing.Dict[str,numbers.Real]]= field(default_factory=list)
    angles: typing.List[typing.Dict[str,numbers.Number]]= field(default_factory=list)

    def extract_energies(self):
        return self.energies

    def extract_gradients(self, key:str):
        return [d[key] for d in self.gradients]

    def extract_angles(self, key:str):
        return [d[key] for d in self.angles]

class Optimizer:

    def __init__(self, simulator: typing.Type = None, maxiter: int=None, samples: int = None, save_history: bool = True):

        if isinstance(simulator, type):
            self.simulator = simulator()
        else:
            self.simulator = simulator

        if maxiter is None:
            self.maxiter = 100
        else:
            self.maxiter = maxiter

        self.samples = samples
        self.save_history = save_history
        if save_history:
            self.history = OptimizerHistory()
        else:
            self.history = None

    def __call__(self, objective: Objective, initial_values: typing.Dict[str, numbers.Number] = None) -> typing.Dict[str, numbers.Number]:
        """
        Will try to solve and give back optimized parameters
        :param objective: openvqe Objective object
        :param parameters: initial parameters, if none the optimizers uses what was set in the objective
        :return: optimal parameters
        """
        raise OpenVQEOptimizerException("Try to call BaseClass")

    def update_parameters(self, parameters: typing.Dict[str, float], *args, **kwargs) -> typing.Dict[str, float]:
        """
        :param parameters: the parameters which will be updated
        :return: updated parameters
        """
        raise OpenVQEOptimizerException("Try to call BaseClass")

    def plot(self, filename=None, property:typing.Union[str,typing.List[str]]='energies', *args, **kwargs):
        """
        Convenience function to plot the progress of the optimizer
        :param filename: if given plot to file, otherwise plot to terminal
        """
        if not self.save_history:
            raise OpenVQEOptimizerException("You explicitly set save_history to False which means there is no data to plot")

        properties = None
        if hasattr(property, "len"):
            properties = property
        else:
            properties = [property]

        for p in properties:
            data = None
            if p.lower() == "energies":
                data = self.history.extract_energies()
            elif "angles" in p.lower():
                key = p.lower().split(' ')[1]
                data = self.history.extract_angles(key=key)
            elif "gradients" in p.lower():
                key = p.lower().split(' ')[1]
                print("key=", key)
                data = self.history.extract_gradients(key=key)

            from matplotlib import pyplot as plt
            plt.plot(data, label=p, marker='o', linestyle='--')
            plt.legend()
            plt.show()








    def initialize_simulator(self, samples: int):
        if self.simulator is not None:
            if hasattr(self.simulator, "run"):
                return self.simulator
            else:
                return self.simulator()
        else:
            return pick_simulator(samples=samples)()
