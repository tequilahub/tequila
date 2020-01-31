import numpy

"""
Define Containers for SciPy usage
"""


class _EvalContainer:
    """
    Container Class to access scipy and keep the optimization history
    This class is used by the SciPy optimizer and should not be used somewhere else
    """

    def __init__(self, objective, param_keys, samples=None, save_history=True, silent: bool = True):
        self.objective = objective
        self.samples = samples
        self.param_keys = param_keys
        self.N = len(param_keys)
        self.save_history = save_history
        self.silent = silent
        if save_history:
            self.history = []
            self.history_angles = []

    def __call__(self, p, *args, **kwargs):
        angles = dict((self.param_keys[i], p[i]) for i in range(self.N))
        E = self.objective(variables=angles, samples=self.samples)
        if not self.silent:
            print("E=", E, " angles=", angles, " samples=", self.samples)
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
        dO = self.objective
        dE_vec = numpy.zeros(self.N)
        memory = dict()
        variables = dict((self.param_keys[i], p[i]) for i in range(len(self.param_keys)))
        for i in range(self.N):
            dE_vec[i] = dO[self.param_keys[i]](variables=variables, samples=self.samples)
            memory[self.param_keys[i]] = dE_vec[i]
        self.history.append(memory)
        return dE_vec
