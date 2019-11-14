from openvqe.circuit.gradient import grad
from openvqe import numpy as np

"""
Define Containers for SciPy usage
"""

class _EvalContainer:
    """
    Container Class to access scipy and keep the optimization history
    This class is used by the SciPy optimizer and should not be used somewhere else
    """

    def __init__(self, objective, param_keys, eval, save_history):
        self.objective = objective
        self.param_keys = param_keys
        self.eval = eval
        self.N = len(param_keys)
        self.save_history = save_history
        if save_history:
            self.history = []
            self.history_angles = []

    def __call__(self, p, *args, **kwargs):
        angles = dict((self.param_keys[i], p[i]) for i in range(self.N))
        self.objective.update_parameters(angles)
        E = self.eval(self.objective)
        if self.save_history:
            self.history.append(E)
            self.history_angles.append(angles)
        return E


class _GradContainer(_EvalContainer):
    """
    Same for the gradients
    Container Class to access scipy and keep the optimization history
    """

    def __call__(self, p, *args, **kwargs):
        dO = grad(self.objective, self.param_keys)
        dE_vec = np.zeros(self.N)
        memory = dict()
        for i in range(self.N):
            dO[self.param_keys[i]].update_parameters(
                dict((self.param_keys[i], p[i]) for i in range(len(self.param_keys))))
            dE_vec[i] = self.eval(dO[self.param_keys[i]])
            memory[self.param_keys[i]] = dE_vec[i]
        self.history.append(memory)
        return dE_vec