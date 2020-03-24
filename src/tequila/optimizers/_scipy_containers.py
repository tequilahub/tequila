import numpy
from tequila import TequilaException
"""
Define Containers for SciPy usage
"""


class _EvalContainer:
    """
    Container Class to access scipy and keep the optimization history
    This class is used by the SciPy optimizer and should not be used somewhere else
    """

    def __init__(self, objective, param_keys,passive_angles=None, samples=None, save_history=True,
                 silent: bool = True):
        self.objective = objective
        self.samples = samples
        self.param_keys = param_keys
        self.N = len(param_keys)
        self.save_history = save_history
        self.silent = silent
        self.passive_angles = passive_angles
        if save_history:
            self.history = []
            self.history_angles = []

    def __call__(self, p, *args, **kwargs):
        angles = dict((self.param_keys[i], p[i]) for i in range(self.N))
        if self.passive_angles is not None:
            angles = {**angles, **self.passive_angles}

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
        if self.passive_angles is not None:
            variables = {**variables, **self.passive_angles}
        for i in range(self.N):
            dE_vec[i] = dO[self.param_keys[i]](variables=variables, samples=self.samples)
            memory[self.param_keys[i]] = dE_vec[i]
        self.history.append(memory)
        return dE_vec

class _QngContainer(_EvalContainer):



    def __init__(self, objective,metric_tensor_blocks, param_keys,passive_angles=None, samples=None, save_history=True,
                 silent: bool = True):

        if passive_angles not in [None,{}]:
            print(passive_angles)
            raise TequilaException('cannot have passive angles in a QNG yet')
        super().__init__(objective=objective,param_keys=param_keys,passive_angles=passive_angles,
                         samples=samples,save_history=save_history,silent=silent)

        self.blocks = metric_tensor_blocks

    def __call__(self,p,*args,**kwargs):
        dO = self.objective
        dE_vec = numpy.zeros(self.N)
        memory = dict()
        variables = dict((self.param_keys[i], p[i]) for i in range(len(self.param_keys)))
        for i in range(self.N):
            dE_vec[i] = dO[self.param_keys[i]](variables=variables, samples=self.samples)
        lol = numpy.zeros((self.N,self.N))
        d_v = 0
        for block in self.blocks:
            d_v_temp = 0
            for i, row in enumerate(block):
                for j, term in enumerate(row):
                    if i <= j:
                        lol[i + d_v][j + d_v] = term(variables=variables,samples=self.samples)
                    else:
                        lol[i + d_v][j + d_v] = lol[j + d_v][i + d_v]
                d_v_temp += 1
            d_v += d_v_temp
        thing=numpy.linalg.pinv(lol)
        out=numpy.dot(thing,dE_vec)
        for i in range(self.N):
            memory[self.param_keys[i]] = out[i]
        self.history.append(memory)
        return out


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
                value = ddO[key](variables=variables, samples=self.samples)
                ddE_mat[i, j] = value
                ddE_mat[j, i] = value
                memory[key] = value
        self.history.append(memory)
        return ddE_mat
