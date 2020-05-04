import numpy, typing, numbers, copy
from tequila.objective import Objective
from tequila.objective.objective import assign_variable, Variable, format_variable_dictionary, format_variable_list
from .optimizer_base import Optimizer
from tequila.circuit.gradient import grad
from collections import namedtuple
from tequila.simulators.simulator_api import compile
from tequila.circuit.noise import NoiseModel
from tequila.tools.qng import CallableVector, QNGVector, get_qng_combos
from tequila.utils import TequilaException

GDReturnType = namedtuple('GDReturnType', 'energy angles history moments')


class OptimizerGD(Optimizer):

    @classmethod
    def available_methods(cls):
        """:return: All tested available methods"""
        return ['adam','adagrad','adamax','nadam','sgd','momentum','nesterov','rmsprop','rmsprop-nesterov']

    def __init__(self,maxiter=100,
                 method='sgd',
                 lr=0.1,
                 beta=0.9,
                 rho=0.999,
                 epsilon= 1.0*10**(-7),
                 samples=None,
                 backend=None,
                 backend_options=None,
                 noise=None,
                 silent=True,
                 **kwargs):
        """
        Optimize a circuit to minimize a given objective using Adam
        See the Optimizer class for all other parameters to initialize
        """

        super().__init__(maxiter=maxiter,samples=samples,
                         backend=backend,backend_options=backend_options,
                         noise=noise,
                         **kwargs)
        method_dict = {
            'adam': self.adam,
            'adagrad':self.adagrad,
            'adamax':self.adamax,
            'nadam':self.nadam,
            'sgd': self.sgd,
            'momentum': self.momentum,
            'nesterov': self.nesterov,
            'rmsprop': self.rms,
            'rmsprop-nesterov': self.rms_nesterov}

        self.f = method_dict[method.lower()]
        self.silent = silent
        self.gradient_lookup={}
        self.active_key_lookup={}
        self.moments_lookup={}
        self.moments_trajectory={}
        self.step_lookup={}
        ### scaling parameters. lr is learning rate.
        ### beta rescales first moments. rho rescales second moments. epsilon is for division stability.
        self.lr = lr
        self.beta = beta
        self.rho = rho
        self.epsilon = epsilon
        assert all([k >.0 for k in [lr,beta,rho,epsilon]])


    def __call__(self, objective: Objective,
                 maxiter,
                 qng: bool = False,
                 stop_count: int = None,
                 initial_values: typing.Dict[Variable, numbers.Real] = None,
                 variables: typing.List[Variable] = None,
                 reset_history: bool = True,
                 method_options: dict = None,
                 *args,**kwargs) -> GDReturnType:
        """
        Optimizes with a variation of gradient descent and gives back the optimized angles
        Get the optimized energies over the history
        :param objective: The tequila Objective to minimize
        :param maxiter: how many iterations to run, at maximum.
        :param qng: whether or not to use the QNG to calculate gradients.
        :param stop_count: how many steps after which to abort if no improvement occurs.
        :param initial_values: initial values for the objective
        :param variables: which variables to optimize over. Default None: all the variables of the objective.
        :param reset_history: reset the history before optimization starts (has no effect if self.save_history is False)
        :return: tuple of optimized energy ,optimized angles and scipy output
        """



        if self.save_history and reset_history:
            self.reset_history()

        active_angles,passive_angles,variables=self.initialize_variables(objective,initial_values,variables)
        v={**active_angles,**passive_angles}


        comp=self.prepare(objective, initial_values, variables, qng, method_options)

        ### prefactor. Early stopping, initialization, etc. handled here

        if maxiter is None:
            maxiter = self.maxiter
        if stop_count == None:
            stop_count = maxiter

        ### the actual algorithm acts here:
        best = None
        best_angles = None
        tally = 0
        for step in range(maxiter):
            e = comp(v,samples=self.samples)
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
                    tally = 0
                else:
                    tally += 1

            if not self.silent:
                string = "Iteration: {} , Energy: {}, angles: {}".format(str(step), str(e), v)
                print(string)

            ### check if its time to stop!
            if tally == stop_count:
                if not self.silent:
                    print('no improvement after {} epochs. Stopping optimization.'.format(str(stop_count)))
                break

            ### get new parameters with self.step!
            v=self.step(comp,v)
        E_final, angles_final = best, best_angles
        return GDReturnType(energy=E_final, angles=format_variable_dictionary(angles_final), history=self.history,
                            moments=self.moments_trajectory[id(comp)])


    def prepare(self,objective,initial_values=None,variables=None,qng=False,method_options=None):
        active_angles,passive_angles,variables=self.initialize_variables(objective,initial_values,variables)
        v={**active_angles,**passive_angles}
        comp = compile(objective=objective, variables=v, backend=self.backend,
                       noise=self.noise,backend_options=self.backend_options,
                       samples=self.samples)
        ostring=id(comp)
        if method_options is not None and 'jac' in method_options and method_options['jac'].lower() in ['2-point',
                                                                                                        'numerical']:
            use_2_point = True
            if 'eps' in method_options:
                eps = method_options['eps']
            else:
                print(
                    "You demanded numerical gradients but did not pass down a stepsize.\n Better set the stepsize with 'eps' in the method_options dictionary")
                eps = 1.e-5
                print("Setting eps to ", eps)
            print("Using numerical gradients ... proceed with caution")
            print("options are: ", method_options)
        else:
            use_2_point = False

        active_angles, passive_angles, variables = self.initialize_variables(objective, initial_values, variables)
        self.active_key_lookup[ostring] = active_angles.keys()
        v = {**active_angles, **passive_angles}
        # Transform the initial value directory into (ordered) arrays

        if not qng:
            g_list = []
            for k in active_angles.keys():
                if use_2_point:
                    if 'func' in method_options:
                        func = method_options['func']
                    else:
                        func = None

                    def numerical_gradient(variables, *args, **kwargs):
                        if func is None:
                            left = copy.deepcopy(variables)
                            right = copy.deepcopy(variables)
                            left[k] += eps
                            right[k] -= eps
                            return (comp(left) - comp(right)) / (2 * eps)
                        else:
                            return func(comp, variables, k, eps)

                    g_comp = numerical_gradient
                    g_list.append(g_comp)
                else:
                    g = grad(objective, k)
                    g_comp = compile(objective=g, variables=v, backend=self.backend,
                                     noise_model=self.noise, samples=self.samples)
                    g_list.append(g_comp)

            self.gradient_lookup[ostring] = CallableVector(g_list)
        else:
            if use_2_point:
                raise Exception("Can not currently combine numerical gradients with QNG")
            self.gradient_lookup[ostring] = QNGVector(get_qng_combos(objective=objective, initial_values=v,
                                                 backend=self.backend,
                                                 noise=self.noise, samples=self.samples,
                                                 backend_options=self.backend_options))

        if not self.silent:
            print("backend: {}".format(objective.backend))
            print("samples: {}".format(self.samples))
            print("{} active variables".format(len(active_angles)))
            print("qng: {}".format(str(qng)))


        vec_len = len(active_angles)
        first = numpy.zeros(vec_len)
        second = numpy.zeros(vec_len)
        self.moments_lookup[ostring] = (first,second)
        self.moments_trajectory[ostring] = [(first,second)]
        self.step_lookup[ostring]=0
        return comp

    def step(self,objective,parameters):
        s=id(objective)
        try:
            gradients,active_keys,last_moment,adam_step=self.retrieve(s)
        except:
            raise TequilaException('Could not retrieve necessary information. Please use the prepare function before optimizing!')
        new, moments, grads=self.f(step=adam_step,gradients=gradients,active_keys=active_keys,moments=last_moment,v=parameters)
        back={**parameters}
        for k in new.keys():
            back[k]=new[k]
        save_grad={}
        self.moments_lookup[s]=moments
        self.moments_trajectory[s].append(moments)
        if self.save_history:
            for i, k in enumerate(active_keys):
                save_grad[k] = grads[i]
            self.history.gradients.append(save_grad)
        self.step_lookup[s]+=1
        return back

    def reset_stepper(self):
        self.moments_trajectory={}
        self.moments_lookup={}
        self.step_lookup={}
        self.gradient_lookup={}
    
    def retrieve(self,s):
        return self.gradient_lookup[s],self.active_key_lookup[s],self.moments_lookup[s],self.step_lookup[s]

    def adam(self,gradients,step,
             v,moments,active_keys,
             **kwargs):
        t=step+1
        s = moments[0]
        r = moments[1]
        grads = gradients(v,samples=self.samples)
        s = self.beta * s + (1 - self.beta) * grads
        r = self.rho * r + (1 - self.rho) * numpy.square(grads)
        s_hat = s / (1 - self.beta ** t)
        r_hat = r / (1 - self.rho ** t)
        updates = []
        for i in range(len(grads)):
            rule = - self.lr * s_hat[i] / (numpy.sqrt(r_hat[i]) + self.epsilon)
            updates.append(rule)
        new = {}
        for i, k in enumerate(active_keys):
            new[k] = v[k] + updates[i]
        back_moment = [s, r]
        return new, back_moment, grads

    def adagrad(self,gradients,
            v, moments, active_keys, **kwargs):
        r=moments[1]
        grads = gradients(v,self.samples)

        r += numpy.square(grads)
        new = {}
        for i, k in enumerate(active_keys):
            new[k] = v[k] - self.lr * grads[i] / numpy.sqrt(r[i] + self.epsilon)

        back_moments = [moments[0], r]
        return new, back_moments, grads

    def adamax(self, gradients,
             v, moments, active_keys, **kwargs):

        s = moments[0]
        r = moments[1]
        grads = gradients(v,samples=self.samples)
        s = self.beta * s + (1 - self.beta) * grads
        r = self.rho * r + (1 - self.rho) * numpy.linalg.norm(grads, numpy.inf)
        updates = []
        for i in range(len(grads)):
            rule = - self.lr * s[i] / r[i]
            updates.append(rule)
        new = {}
        for i, k in enumerate(active_keys):
            new[k] = v[k] + updates[i]
        back_moment = [s, r]
        return new, back_moment, grads

    def nadam(self,step,gradients,
             v,moments,active_keys,
             **kwargs):

        s = moments[0]
        r = moments[1]
        t = step+1
        grads = gradients(v,samples=self.samples)
        s = self.beta * s + (1 - self.beta) * grads
        r = self.rho * r + (1 - self.rho) * numpy.square(grads)
        s_hat = s / (1 - self.beta ** t)
        r_hat = r / (1 - self.rho ** t)
        updates = []
        for i in range(len(grads)):
            rule = - self.lr * (self.beta * s_hat[i] + (1 - self.beta) * grads[i] / (1 - self.beta ** t)) / (numpy.sqrt(r_hat[i]) + self.epsilon)
            updates.append(rule)
        new = {}
        for i, k in enumerate(active_keys):
            new[k] = v[k] + updates[i]
        back_moment = [s, r]
        return new, back_moment, grads

    def sgd(self, gradients,
            v, moments, active_keys, **kwargs):

        ### the sgd optimizer without momentum.
        grads = gradients(v,samples=self.samples)
        new = {}
        for i, k in enumerate(active_keys):
            new[k] = v[k] - self.lr * grads[i]
        return new, moments, grads

    def momentum(self,gradients,
             v,moments,active_keys,**kwargs):

        m = moments[0]
        ### the sgd momentum optimizer. m is our moment tally
        grads = gradients(v,samples=self.samples)

        m = self.beta * m - self.lr * grads
        new = {}
        for i, k in enumerate(active_keys):
            new[k] = v[k] + m[i]

        back_moments = [m, moments[1]]
        return new, back_moments, grads

    def nesterov(self, gradients,
            v, moments, active_keys, **kwargs):

        m = moments[0]

        interim = {}
        for i, k in enumerate(active_keys):
            interim[k] = v[k] + self.beta * m[i]
        active_keyset = set([k for k in active_keys])
        total_keyset = set([k for k in v.keys()])
        for k in total_keyset:
            if k not in active_keyset:
                interim[k] = v[k]
        grads = gradients(interim,samples=self.samples)

        m = self.beta * m - self.lr * grads
        new = {}
        for i, k in enumerate(active_keys):
            new[k] = v[k] + m[i]

        back_moments = [m, moments[1]]
        return new, back_moments, grads

    def rms(self, gradients,
                 v, moments, active_keys,
                 **kwargs):

        r = moments[1]
        grads = gradients(v,samples=self.samples)
        r = self.rho * r + (1 - self.rho) * numpy.square(grads)
        new = {}
        for i, k in enumerate(active_keys):
            new[k] = v[k] - self.lr * grads[i] / numpy.sqrt(self.epsilon + r[i])

        back_moments = [moments[0], r]
        return new, back_moments, grads

    def rms_nesterov(self,gradients,
            v, moments, active_keys,
            **kwargs):

        m = moments[0]
        r = moments[1]

        interim = {}
        for i, k in enumerate(active_keys):
            interim[k] = v[k] + self.beta * m[i]
        active_keyset = set([k for k in active_keys])
        total_keyset = set([k for k in v.keys()])
        for k in total_keyset:
            if k not in active_keyset:
                interim[k] = v[k]
        grads = gradients(interim,samples=self.samples)

        r = self.rho * r + (1 - self.rho) * numpy.square(grads)
        for i in range(len(m)):
            m[i] = self.beta * m[i] - self.lr * grads[i] / numpy.sqrt(r[i])
        new = {}
        for i, k in enumerate(active_keys):
            new[k] = v[k] + m[i]

        back_moments = [m, r]
        return new, back_moments, grads


def minimize(objective: Objective,
             lr=0.1,
             method='sgd',
             qng: bool = False,
             stop_count=None,
             initial_values: typing.Dict[typing.Hashable, numbers.Real] = None,
             variables: typing.List[typing.Hashable] = None,
             samples: int = None,
             maxiter: int = 100,
             backend: str = None,
             backend_options: typing.Dict = None,
             noise: NoiseModel = None,
             silent: bool = False,
             save_history: bool = True,
             method_options: dict = None,
             beta: float = 0.9,
             rho: float = 0.999,
             epsilon: float = 1.*10**(-7),
             *args,
             **kwargs) -> GDReturnType:
    """

    Parameters
    ----------
    objective: Objective :
        The tequila objective to optimize
    lr: float >0:
        the learning rate. Default 0.01.
    beta: float >0:
        scaling factor for first moments. default 0.9
    rho: float >0:
        scaling factor for second moments. default 0.999
    epsilon: float>0:
        small float for stability of division. default 10^-7

    method: string:
        which variation on Gradient Descent to use. Options include 'sgd','adam','nesterov','adagrad','rmsprop', etc.
    qng: bool:
        whether or not the gradient calculated should be the quantum natural gradient or not. defaults to False.
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
    backend_options: dict:
        (Default value = None)
        extra options, to be passed to the backend
    noise: NoiseModel:
         (Default value = None)
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

    optimizer = OptimizerGD(save_history=save_history,
                            method=method,
                            lr=lr,
                            beta=beta,
                            rho=rho,
                            epsilon=epsilon,
                            samples=samples,backend=backend,
                            noise=noise,backend_options=backend_options,
                            maxiter=maxiter,
                            silent=silent)
    return optimizer(objective=objective,
                     maxiter=maxiter,
                     qng=qng,
                     stop_count=stop_count, initial_values=initial_values,
                     variables=variables,
                     method_options=method_options,*args,**kwargs)
