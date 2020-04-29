import numpy

"""
Define Containers for SciPy usage
"""
from tequila.objective import format_variable_dictionary
from tequila.tools.qng import evaluate_qng


class _EvalContainer:
    """
    Container Class to access scipy and keep the optimization history
    This class is used by the SciPy optimizer and should not be used somewhere else
    """

    def __init__(self, objective, param_keys, passive_angles=None, samples=None, save_history=True,
                 print_level: int = 2, backend_options=None):
        self.objective = objective
        self.samples = samples
        self.param_keys = param_keys
        self.N = len(param_keys)
        self.save_history = save_history
        self.print_level = print_level
        self.passive_angles = passive_angles
        if backend_options is None:
            self.backend_options = {}
        else:
            self.backend_options = backend_options
        if save_history:
            self.history = []
            self.history_angles = []

    def __call__(self, p, *args, **kwargs):
        angles = dict((self.param_keys[i], p[i]) for i in range(self.N))
        if self.passive_angles is not None:
            angles = {**angles, **self.passive_angles}
        vars = format_variable_dictionary(angles)
        E = self.objective(variables=vars, samples=self.samples, **self.backend_options)
        if self.print_level > 1:
            print("E={:+2.8f}".format(E), " angles=", angles, " samples=", self.samples)
        elif self.print_level > 0:
            print("E={:+2.8f}".format(E))
        if self.save_history:
            self.history.append(E)
            self.history_angles.append(angles)
        return numpy.float64(E)  # jax types confuses optimizers


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
        if self.passive_angles is not None:
            variables = {**variables, **self.passive_angles}
        for i in range(self.N):
            dE_vec[i] = dO[self.param_keys[i]](variables=variables, samples=self.samples, **self.backend_options)
            memory[self.param_keys[i]] = dE_vec[i]
        self.history.append(memory)
        return numpy.asarray(dE_vec, dtype=numpy.float64)  # jax types confuse optimizers


class _QngContainer(_EvalContainer):

    def __init__(self, combos, param_keys, passive_angles=None, samples=None, save_history=True,
                 silent: bool = True, *args, **kwargs):

        super().__init__(objective=None, param_keys=param_keys, passive_angles=passive_angles,
                         samples=samples, save_history=save_history, silent=silent, *args, **kwargs)

        self.combos = combos

    def evaluate_qng(self, variables):
        return evaluate_qng(self.combos, variables)

    def __call__(self, p, *args, **kwargs):
        memory = dict()
        variables = dict((self.param_keys[i], p[i]) for i in range(len(self.param_keys)))
        if self.passive_angles is not None:
            variables = {**variables, **self.passive_angles}
        out = self.evaluate_qng(variables=variables)
        for i in range(self.N):
            memory[self.param_keys[i]] = out[i]
        self.history.append(memory)
        return numpy.asarray(out, dtype=numpy.float64)


class _HessContainer(_EvalContainer):

    def __call__(self, p, *args, **kwargs):
        ddO = self.objective
        ddE_mat = numpy.zeros(shape=[self.N, self.N])
        memory = dict()
        variables = dict((self.param_keys[i], p[i]) for i in range(len(self.param_keys)))
        if self.passive_angles is not None:
            variables = {**variables, **self.passive_angles}
        for i in range(self.N):
            for j in range(i, self.N):
                key = (self.param_keys[i], self.param_keys[j])
                value = ddO[key](variables=variables, samples=self.samples, **self.backend_options)
                ddE_mat[i, j] = value
                ddE_mat[j, i] = value
                memory[key] = value
        self.history.append(memory)
        return numpy.asarray(ddE_mat, dtype=numpy.float64)  # jax types confuse optimizers
