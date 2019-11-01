"""
BaseCalss for Optimizers
Suggestion, feel free to propose new things/changes
"""

from openvqe import OpenVQEException
from openvqe import typing
from openvqe.circuit.variable import Variable
from openvqe.objective import Objective


class OpenVQEOptimizerExcpetion(OpenVQEException):
    pass


class Optimizer:

    def __call__(self, objective: Objective, parameters: typing.Dict[str, float]=None, *args, **kwargs) -> typing.Dict[str, float]:
        """
        Will try to solve and give back optimized parameters
        :param objective: openvqe Objective object
        :param parameters: initial parameters, if none the optimizers uses what was set in the objective
        :return: optimal parameters
        """
        raise OpenVQEOptimizerExcpetion("Try to call BaseClass")

    def update_parameters(self, parameters: typing.Dict[str, float], *args, **kwargs) -> typing.Dict[str, float]:
        """
        :param parameters: the parameters which will be updated
        :return: updated parameters
        """
        raise OpenVQEOptimizerExcpetion("Try to call BaseClass")

    def plot(self, filename=None, *args, **kwargs ):
        """
        Convenience function to plot the progress of the optimizer
        :param filename: if given plot to file, otherwise plot to terminal
        """
        raise OpenVQEOptimizerExcpetion("Try to call BaseClass")
