import numpy, typing, numbers
from tequila.objective import Objective
from tequila.objective.objective import assign_variable, Variable, format_variable_dictionary, format_variable_list
from .optimizer_base import Optimizer
from tequila.circuit.gradient import grad
from collections import namedtuple
from tequila.simulators.simulator_api import compile
from tequila.circuit.noise import NoiseModel

GDReturnType = namedtuple('SciPyReturnType', 'energy angles history')


class OptimizerGD(Optimizer):

    @classmethod
    def available_methods(cls):
        """:return: All tested available methods"""
        return ['adam','adagrad','adamax','nadam','basic','sgd','nesterov','rms','rms-nesterov']

    def __init__(self,
                 maxiter=100,
                 silent=True,
                 **kwargs):
        """
        Optimize a circuit to minimize a given objective using Adam
        See the Optimizer class for all other parameters to initialize
        """
        self.silent=silent
        self.maxiter=maxiter
        super().__init__(**kwargs)
        self.method_dict = {
            'adam': self.adam,
            'adagrad':self.adagrad,
            'adamax':self.adamax,
            'nadam':self.nadam,
            'basic': self.basic,
            'sgd': self.sgd,
            'nesterov': self.nesterov,
            'rms': self.rms,
            'rms-nesterov':self.rms_nesterov}

    def __call__(self, objective: Objective,
                 maxiter,
                 lr: float = .01,
                 method: str = 'sgd',
                 stop_count : int = None,
                 initial_values: typing.Dict[Variable, numbers.Real] = None,
                 variables: typing.List[Variable] = None,
                 samples: int = None,
                 backend: str = None,
                 noise: NoiseModel = None,
                 reset_history: bool = True,*args,**kwargs) -> GDReturnType:
        """
        Optimizes with a variation of gradient descent and gives back the optimized angles
        Get the optimized energies over the history
        :param objective: The tequila Objective to minimize
        :param maxiter: how many iterations to run, at maximum.
        :param method: what method to optimize via.
        :param stop_count: how many steps after which to abort if no improvement occurs.
        :param initial_values: initial values for the objective
        :param variables: which variables to optimize over. Default None: all the variables of the objective.
        :param samples: the number of samples to use. Default None: Wavefunction simulation used instead.
        :param backend: which simulation backend to use. Default None: let Tequila Pick!
        :param noise: the NoiseModel to apply to sampling. Default None. Affects chosen simulator.
        :param reset_history: reset the history before optimization starts (has no effect if self.save_history is False)
        :return: tuple of optimized energy ,optimized angles and scipy output
        """



        if self.save_history and reset_history:
            self.reset_history()

        active_angles = {}
        for v in variables:
            active_angles[v] = initial_values[v]

        passive_angles = {}
        for k, v in initial_values.items():
            if k not in active_angles.keys():
                passive_angles[k] = v

        # Transform the initial value directory into (ordered) arrays

        comp = compile(objective=objective, variables=initial_values, backend=backend,
                                     noise_model=noise,
                                     samples=samples)

        gradients=[]
        for k in active_angles.keys():
            g=grad(objective,k)
            g_comp = compile(objective=g,variables=initial_values, backend=backend,
                             noise_model=noise,samples=samples)
            gradients.append(g_comp)


        if not self.silent:
            print("ObjectiveType is {}".format(type(comp)))
            print("backend: {}".format(comp.backend))
            print("samples: {}".format(samples))
            print("{} active variables".format(len(active_angles)))

        ### prefactor. Early stopping, initialization, etc. handled here

        if maxiter is None:
            maxiter=self.maxiter
        if stop_count == None:
            stop_count = maxiter

        ### the actual algorithm acts here:

        f=self.method_dict[method.lower()]
        v = initial_values
        vec_len = len(gradients)
        best = None
        best_angles = None
        first = numpy.zeros(vec_len)
        second = numpy.zeros(vec_len)
        moments=[first,second]

        tally = 0
        for step in range(maxiter):
            e = comp(v)
            self.history.energies.append(e)
            self.history.angles.append(v)

            ### saving best performance and counting the stop tally.
            if step == 0:
                best = e
                best_angles = v
                tally = 0
            else:
                if e < best:
                    best = e
                    best_angles = v
                else:
                    tally += 1

            if not self.silent:
                string = "Iteration: {} , Energy: {}, angles: {}".format(str(step), str(e), v)
                print(string)

            ### check if its time to stop!
            if tally == stop_count:
                if not self.silent:
                    print('no improvement after {} epochs. Stopping optimization.')
                break

            new,moments=f(lr=lr,step=step,gradients=gradients,v=v,moments=moments,active_angles=active_angles,**kwargs)
            if passive_angles is not None:
                v = {**new, **passive_angles}
            else:
                v = new

        E_final,angles_final=best,best_angles
        angles_final = {**angles_final, **passive_angles}
        return GDReturnType(energy=E_final, angles=format_variable_dictionary(angles_final), history=self.history)

    def adam(self,lr,step,gradients,
             v,moments,active_angles,
             beta=0.9,beta2=0.999,epsilon=10**-7,**kwargs):


        s = moments[0]
        r = moments[1]
        t = step+1
        grads = numpy.asarray([g(v) for g in gradients])
        s = beta * s + (1 - beta) * grads
        r = beta2 * r + (1 - beta2) * numpy.square(grads)
        s_hat = s / (1 - beta ** t)
        r_hat = r / (1 - beta2 ** t)
        updates = []
        for i in range(len(grads)):
            rule = - lr * s_hat[i] / (numpy.sqrt(r_hat[i]) + epsilon)
            updates.append(rule)
        new = {}
        for i, k in enumerate(active_angles.keys()):
            new[k] = v[k] + updates[i]
        back_moment=[s,r]
        return new,back_moment

    def adagrad(self, lr, gradients,
            v, moments, active_angles,epsilon=10**-6, **kwargs):
        r=moments[1]
        grads = numpy.asarray([g(v) for g in gradients])

        r += numpy.square(grads)
        new = {}
        for i, k in enumerate(active_angles.keys()):
            new[k] = v[k] - lr  *grads[i]/numpy.sqrt(r[i]+epsilon)


        back_moments = [moments[0], r]
        return new, back_moments

    def adamax(self, lr, gradients,
             v, moments, active_angles,
             beta=0.9, beta2=0.999, epsilon=10 ** -7, **kwargs):

        s = moments[0]
        r = moments[1]
        grads = numpy.asarray([g(v) for g in gradients])
        s = beta * s + (1 - beta) * grads
        r = beta2 * r + (1 - beta2) * numpy.linalg.norm(grads,numpy.inf)
        updates = []
        for i in range(len(grads)):
            rule = - lr * s[i] / r[i]
            updates.append(rule)
        new = {}
        for i, k in enumerate(active_angles.keys()):
            new[k] = v[k] + updates[i]
        back_moment = [s, r]
        return new, back_moment

    def nadam(self,lr,step,gradients,
             v,moments,active_angles,
             beta=0.9,beta2=0.999,epsilon=10**-7,**kwargs):


        s = moments[0]
        r = moments[1]
        t = step+1
        grads = numpy.asarray([g(v) for g in gradients])
        s = beta * s + (1 - beta) * grads
        r = beta2 * r + (1 - beta2) * numpy.square(grads)
        s_hat = s / (1 - beta ** t)
        r_hat = r / (1 - beta2 ** t)
        updates = []
        for i in range(len(grads)):
            rule = - lr * (beta*s_hat[i] +(1-beta)*grads[i]/(1-beta**t)) / (numpy.sqrt(r_hat[i]) + epsilon)
            updates.append(rule)
        new = {}
        for i, k in enumerate(active_angles.keys()):
            new[k] = v[k] + updates[i]
        back_moment=[s,r]
        return new,back_moment

    def basic(self, lr, gradients,
            v, moments, active_angles, **kwargs):

        ### the sgd momentum optimizer. m is our moment tally
        grads = numpy.asarray([g(v) for g in gradients])
        new = {}
        for i, k in enumerate(active_angles.keys()):
            new[k] = v[k] - lr*grads[i]
        return new, moments

    def sgd(self,lr,gradients,
             v,moments,active_angles,
             beta=0.9,**kwargs):

        m=moments[0]

        ### the sgd momentum optimizer. m is our moment tally
        grads = numpy.asarray([g(v) for g in gradients])

        m=beta*m -lr*grads
        new={}
        for i, k in enumerate(active_angles.keys()):
            new[k] = v[k] + m[i]

        back_moments=[m,moments[1]]
        return new,back_moments

    def nesterov(self, lr, gradients,
            v, moments, active_angles,
            beta=0.9, **kwargs):


        m = moments[0]

        interim = {}
        for i, k in enumerate(active_angles.keys()):
            interim[k] = v[k] + beta * m[i]
        active_keyset=set([k for k in active_angles.keys()])
        total_keyset=set([k for k in v.keys()])
        for k in total_keyset:
            if k not in active_keyset:
                interim[k]=v[k]
        grads = numpy.asarray([g(interim) for g in gradients])

        m = beta * m - lr * grads
        new = {}
        for i, k in enumerate(active_angles.keys()):
            new[k] = v[k] + m[i]

        back_moments=[m,moments[1]]
        return new, back_moments

    def rms(self, lr, gradients,
                 v, moments, active_angles,
                 rho=0.999,epsilon=10**-6, **kwargs):

        r = moments[1]
        grads = numpy.asarray([g(v) for g in gradients])
        r = rho*r +(1-rho)*numpy.square(grads)
        new = {}
        for i, k in enumerate(active_angles.keys()):
            new[k] = v[k]-lr*grads[i]/numpy.sqrt(epsilon+r[i])

        back_moments = [moments[0], r]
        return new, back_moments

    def rms_nesterov(self, lr, gradients,
            v, moments, active_angles,beta=0.9,
            rho=0.999, epsilon=10 ** -6, **kwargs):

        m = moments[0]
        r = moments[1]

        interim = {}
        for i, k in enumerate(active_angles.keys()):
            interim[k] = v[k] + beta * m[i]
        active_keyset=set([k for k in active_angles.keys()])
        total_keyset=set([k for k in v.keys()])
        for k in total_keyset:
            if k not in active_keyset:
                interim[k]=v[k]
        grads = numpy.asarray([g(interim) for g in gradients])

        r = rho * r + (1 - rho) * numpy.square(grads)
        for i in range(len(m)):
            m[i] = beta*m[i] - lr * grads[i]/numpy.sqrt(r[i])
        new = {}
        for i, k in enumerate(active_angles.keys()):
            new[k] = v[k] +m[i]

        back_moments = [m, r]
        return new, back_moments




def minimize(objective: Objective,
             lr=0.01,
             method='sgd',
             stop_count = None,
             initial_values: typing.Dict[typing.Hashable, numbers.Real] = None,
             variables: typing.List[typing.Hashable] = None,
             samples: int = None,
             maxiter: int = 100,
             backend: str = None,
             noise: NoiseModel = None,
             silent: bool = False,
             save_history: bool = True,
             *args,
             **kwargs) -> GDReturnType:
    """

    Parameters
    ----------
    objective: Objective :
        The tequila objective to optimize
    lr: float >0:
        the learning rate. Default 0.01.
    method: string:
        which variation on Gradient Descent to use. Options include 'sgd','adam','nesterov','adagrad','rmsprop',
    stop_count: int:
        how many steps after which to cease training if no improvement occurs. Default None results in going till maxiter is complete
    initial_values: typing.Dict[typing.Hashable, numbers.Real]: (Default value = None):
        Initial values as dictionary of Hashable types (variable keys) and floating point numbers. If given None they will all be set to zero
    variables: typing.List[typing.Hashable] :
         (Default value = None)
         List of Variables to optimize
    samples: int :
         (Default value = None)
         samples/shots to take in every run of the quantum circuits (None activates full wavefunction simulation)
    maxiter: int :
         (Default value = 100)
    backend: str :
         (Default value = None)
         Simulator backend, will be automatically chosen if set to None
    noise: NoiseModel:
         (Default value =None)
         a NoiseModel to apply to all expectation values in the objective.
    stop_count: int :
         (Default value = None)
         Convergence tolerance for optimization; if no improvement after this many epochs, stop.
    silent: bool :
         (Default value = False)
         No printout if True
    save_history: bool:
        (Default value = True)
        Save the history throughout the optimization


    optional kwargs may include beta, beta2, and rho, parameters which affect (but do not need to be altered) the various
    method algorithms.
    Returns
    -------

    """

    # bring into right format
    variables = format_variable_list(variables)
    initial_values = format_variable_dictionary(initial_values)

    # set defaults
    all_variables = objective.extract_variables()
    if variables is None:
        variables = all_variables
    if initial_values is None:
        initial_values = {k: numpy.random.uniform(0, 2 * numpy.pi) for k in all_variables}
    else:
        # autocomplete initial values, warn if you did
        detected = False
        for k in all_variables:
            if k not in initial_values:
                initial_values[k] = numpy.random.uniform(0, 2 * numpy.pi)
                detected = True
        if detected and not silent:
            print("WARNING: initial_variables given but not complete: Autocomplete with random number")

    optimizer = OptimizerGD(save_history=save_history,
                               maxiter=maxiter,
                               silent=silent)
    if initial_values is not None:
        initial_values = {assign_variable(k): v for k, v in initial_values.items()}
    return optimizer(objective=objective,
                     maxiter=maxiter,
                     lr=lr,
                     method=method,
                     stop_count=stop_count,
                     backend=backend, initial_values=initial_values,
                     variables=variables, noise=noise,
                     samples=samples,*args,**kwargs)
