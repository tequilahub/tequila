import numpy, typing, numbers
from tequila.objective import Objective
from tequila.objective.objective import assign_variable, Variable, format_variable_dictionary, format_variable_list
from .optimizer_base import Optimizer
from tequila.circuit.gradient import grad
from collections import namedtuple
from tequila.simulators.simulator_api import compile
from tequila.circuit.noise import NoiseModel
from tequila.tools.qng import get_qng_combos

AdamReturnType = namedtuple('SciPyReturnType', 'energy angles history')


class OptimizerAdam(Optimizer):

    def __init__(self,
                 maxiter=100,
                 silent=True,
                 *args,
                 **kwargs):
        """
        Optimize a circuit to minimize a given objective using Adam
        See the Optimizer class for all other parameters to initialize
        """
        self.silent=silent
        self.maxiter=maxiter
        super().__init__(**kwargs)


    def __call__(self, objective: Objective,
                 maxiter=None,
                 lr : float = .01,
                 beta1: float= 0.9,
                 beta2: float= 0.999,
                 epsilon: float = 10.**-7,
                 stop_count : int = None,
                 initial_values: typing.Dict[Variable, numbers.Real] = None,
                 variables: typing.List[Variable] = None,
                 samples: int = None,
                 backend: str = None,
                 noise: NoiseModel = None,
                 reset_history: bool = True) -> AdamReturnType:
        """
        Optimizes with Adam and gives back the optimized angles
        Get the optimized energies over the history
        :param objective: The tequila Objective to minimize
        :param lr: the learning rate. Default given by Deep Learning (bengio, goodfellow).
        :param beta1: the decay parameter for the first moment. Default given by Deep Learning (bengio, goodfellow).
        :param beta2: the decay parameter for the second moment. Default given by Deep Learning (bengio, goodfellow).
        :param epsilon: a small numerical constant for stability of division. Default given by Deep Learning (bengio, goodfellow).
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
        vec_len=len(gradients)
        energies=[]
        angles=[]
        s=numpy.zeros(vec_len)
        r=numpy.zeros(vec_len)
        t=0.
        v=initial_values
        if maxiter is None:
            maxiter=self.maxiter
        if stop_count == None:
            stop_count = maxiter
        best=None
        best_angles=None
        tally=0

        ### the actual algorithm acts here:
        for step in range(maxiter):
            e=comp(v)
            energies.append(e)
            angles.append(v)

            ### saving best performance and counting the stop tally.
            if step == 0:
                best=e
                best_angles=v
                tally=0
            else:
                if e<best:
                    best=e
                    best_angles=v
                else:
                    tally +=1

            if not self.silent:
                string = "Iteration: {} , Energy: {}, angles: {}".format(str(step),str(e),v)
                print(string)

            ### check if its time to stop!
            if tally == stop_count:
                if not self.silent:
                    print('no improvement after {} epochs. Stopping optimization.')
                E_final = best
                angles_final = best_angles
                angles_final = {**angles_final, **passive_angles}
                self.history.energies = energies
                self.history.angles = angles
                return AdamReturnType(energy=E_final, angles=format_variable_dictionary(angles_final),
                                      history=self.history)



            ### the actual, base adam optimizer. s is a first moment, r is a second moment estimator.
            grads = numpy.asarray([g(v) for g in gradients])
            t += 1
            s = beta1*s +(1-beta1)*grads
            r = beta2*r + (1-beta2)*numpy.square(grads)
            s_hat = s/(1-beta1**t)
            r_hat = r/(1-beta2**t)
            updates=[]
            for i in range(vec_len):
                rule= - lr * s_hat[i]/(numpy.sqrt(r_hat[i])+epsilon)
                updates.append(rule)
            new={}
            for i,k in enumerate(active_angles.keys()):
                new[k] = v[k] +updates[i]

            if passive_angles is not None:
                v = {**new, **passive_angles}



        E_final = best
        angles_final = best_angles
        angles_final = {**angles_final, **passive_angles}
        self.history.energies=energies
        self.history.angles=angles
        return AdamReturnType(energy=E_final, angles=format_variable_dictionary(angles_final), history=self.history)


def minimize(objective: Objective,
             lr=0.001,
             beta1=0.9,
             beta2=0.99,
             epsilon=10.**(-7.),
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
             **kwargs) -> AdamReturnType:
    """

    Parameters
    ----------
    objective: Objective :
        The tequila objective to optimize
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
    method: str :
         (Default value = "BFGS")
         Optimization method (see scipy documentation, or 'available methods')
    stop_count: int :
         (Default value = None)
         Convergence tolerance for optimization; if no improvement after this many epochs, stop.
    silent: bool :
         (Default value = False)
         No printout if True
    save_history: bool:
        (Default value = True)
        Save the history throughout the optimization

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

    optimizer = OptimizerAdam(save_history=save_history,
                               maxiter=maxiter,
                               silent=silent)
    if initial_values is not None:
        initial_values = {assign_variable(k): v for k, v in initial_values.items()}
    return optimizer(objective=objective,
                     maxiter=maxiter,
                     lr=lr,
                     beta1=beta1,
                     beta2=beta2,
                     epsilon=epsilon,
                     stop_count=stop_count,
                     backend=backend, initial_values=initial_values,
                     variables=variables, noise=noise,
                     samples=samples)
