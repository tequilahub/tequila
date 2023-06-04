"""
Base class for Optimizers.
"""
import typing, numbers, copy, warnings

from tequila.utils.exceptions import TequilaException, TequilaWarning
from tequila.simulators.simulator_api import compile, pick_backend
from tequila.objective import Objective
from tequila.circuit.gradient import grad
from dataclasses import dataclass, field
from tequila.objective.objective import assign_variable, Variable, format_variable_dictionary, format_variable_list
import numpy
from random import choices


class TequilaOptimizerException(TequilaException):
    pass


@dataclass
class OptimizerHistory:
    """
    A class representing the history of optimizers over time. Has a variety of convenience functions attached to it.
    """

    @property
    def iterations(self):
        if self.energies is None:
            return 0
        else:
            return len(self.energies)

    # history of all true iterations (epochs)
    energies: typing.List[numbers.Real] = field(default_factory=list)
    gradients: typing.List[typing.Dict[str, numbers.Real]] = field(default_factory=list)
    angles: typing.List[typing.Dict[str, numbers.Number]] = field(default_factory=list)

    # history of all function evaluations
    energy_calls: typing.List[numbers.Real] = field(default_factory=list)
    gradient_calls: typing.List[typing.Dict[str, numbers.Real]] = field(default_factory=list)
    angles_calls: typing.List[typing.Dict[str, numbers.Number]] = field(default_factory=list)
    
    # backward comp.
    @property
    def energies_calls(self):
        return self.energy_calls
    @property
    def energies_evaluations(self):
        return self.energy_calls
    
    def __add__(self, other):
        """
        magic method for convenient combination of history objects.
        """
        result = OptimizerHistory()
        result.energies = self.energies + other.energies
        result.gradients = self.gradients + other.gradients
        result.angles = self.angles + other.angles
        return result

    def __iadd__(self, other):
        """
        magic method for convenient in place combination of history objects.
        """
        self.energies += other.energies
        self.gradients += other.gradients
        self.angles += other.angles
        return self

    def extract_energies(self, *args, **kwargs) -> typing.Dict[numbers.Integral, numbers.Real]:
        """
        convenience function to get the energies back as a dictionary.
        """
        return {i: e for i, e in enumerate(self.energies)}

    def extract_gradients(self, key: str) -> typing.Dict[numbers.Integral, numbers.Real]:
        """
        convenience function to get the gradients of some variable out of the history.
        Parameters
        ----------
        key: str:
            the name of the variable whose gradients are sought

        Returns
        -------
        dict:
            a dictionary, representing the gradient of variable 'key' over time.
        """
        gradients = {}
        for i, d in enumerate(self.gradients):
            if key in d:
                gradients[i] = d[assign_variable(key)]
        return gradients

    def extract_angles(self, key: str) -> typing.Dict[numbers.Integral, numbers.Real]:
        """
        convenience function to get the value of some variable out of the history.

        Parameters
        ----------
        key: str:
            name of the variable whose values are sought

        Returns
        -------
        dict:
            a dictionary, representing the value of variable 'key' over time.
        """
        angles = {}
        for i, d in enumerate(self.angles):
            if key in d:
                angles[i] = d[assign_variable(key)]
        return angles

    def plot(self,
             property: typing.Union[str, typing.List[str]] = 'energies',
             key: str = None,
             filename=None,
             baselines: typing.Dict[str, float] = None,
             *args, **kwargs):

        """
        Convenience function to plot the progress of the optimizer over time.
        Parameters
        ----------
        property: (list of) str: Default = 'energies'
            which property (eg angles, energies, gradients) to plot.
            Default: plot energies over time.
        key: str, optional:
            if property is 'angles' or 'gradients', key allows you to plot just an individual variables' property.
            Default: plot everything
        filename, optional:
            if give, plot to this file; else, plot to terminal.
            Default: plot to terminal.
        baselines: dict, optional:
            dictionary of plotting axis baseline information.
            Default: use whatever matplotlib auto-generates.

        args:
            args.
        kwargs:
            kwargs.

        Returns
        -------
        None
        """

        from matplotlib import pyplot as plt
        from matplotlib.ticker import MaxNLocator
        fig = plt.figure()
        fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        import pickle

        if baselines is not None:
            for k, v in baselines.items():
                plt.axhline(y=v, label=k)

        if hasattr(property, "lower"):
            properties = [property.lower()]
        else:
            properties = property

        labels = None
        if 'labels' in kwargs:
            labels = kwargs['labels']
        elif 'label' in kwargs:
            labels = kwargs['label']

        if hasattr(labels, "lower"):
            labels = [labels] * len(properties)

        for k, v in kwargs.items():
            if hasattr(plt, k):
                f = getattr(plt, k)
                if callable(f):
                    f(v)
                else:
                    f = v

        if key is None:
            keys = [[k for k in self.angles[-1].keys()]] * len(properties)
        elif isinstance(key, typing.Hashable):
            keys = [[assign_variable(key)]] * len(properties)
        else:
            key = [assign_variable(k) for k in key]
            keys = [key] * len(properties)

        for i, p in enumerate(properties):
            try:
                label = labels[i]
            except:
                label = p

            if p == "energies":
                data = getattr(self, "extract_" + p)()
                plt.plot(list(data.keys()), list(data.values()), label=str(label), marker='o', linestyle='--')
            else:
                for k in keys[i]:
                    data = getattr(self, "extract_" + p)(key=k)
                    plt.plot(list(data.keys()), list(data.values()), label=str(label) + " " + str(k), marker='o',
                             linestyle='--')

        loc = 'best'
        if 'loc' in kwargs:
            loc = kwargs['loc']
        plt.legend(loc=loc)
        if filename is None:
            plt.show()
        else:
            pickle.dump(fig, open(filename + ".pickle", "wb"))
            plt.savefig(fname=filename + ".pdf", **kwargs)

@dataclass
class OptimizerResults:

    energy: float = None
    history: OptimizerHistory = None
    variables: dict = None


    @property
    def angles(self):
        # allow backwards compatibility
        return self.variables


class Optimizer:

    """
    The base optimizer class, from which other optimizers inherit.


    Attributes
    ----------

    backend:
        The quantum backend to use (None means autopick)
    maxiter:
        Maximum number of iterations to perform.
    silent:
        whether or not to print during call or on init.
    samples:
        number of samples to call objectives with during call.
    print_level:
        Allow customization of printout in derived classes, is set to 0 if silent==True.
    save_history:
        whether or not to save history.
    history:
        a history object, saving information during optimization.
    noise:
        what noise (e.g, a NoiseModel) to apply to simulations during optimization.
    device:
        the device that sampling (real or emulated) should be performed on.


    Methods
    -------
    reset_history:
        reset the optimizer history.
    initialize_variables:
        convenience: format variables of an objective and segregrate actives from passives.
    compile_objective:
        convenience: compile an objective.
    compile_gradient:
        convenience: build and compile (i.e render callable) the gradient of an objective.
    compile_hessian:
        convenience: build and compile (i.e render callable) the hessian of an objective.

    """
    def __init__(self, backend: str = None,
                 maxiter: int = None,
                 samples: int = None,
                 device: str= None,
                 noise=None,
                 save_history: bool = True,
                 silent: typing.Union[bool, int] = False,
                 print_level: int = 99, *args, **kwargs):

        """
        initialize an optimizer.

        Parameters
        ----------
        backend: str, optional:
            a quantum backend to use. None means autopick.
        maxiter: int, optional:
            maximum number of iterations to performed.
            Note: overwrites attribute of same name to 100, not None, if default.
        samples: int, optional:
            number of samples to simulate measurement of objectives with.
            Default: none, i.e full wavefunction simulation.
        device: optional:
            changeable type. The device on which to perform (or, simulate performing) actual quantum computation.
            Default None will use the basic, un-restricted simulators of backend.
        noise: optional:
            NoiseModel object or str 'device', being either a custom noisemodel or the instruction to use that of
            the emulated device.
            Default value none means: simulate without any noise.
        save_history: bool: Default = True:
            whether or not to save history during optimization. Defaults to true.
        silent: bool: Default = False:
            whether or not to be verbose during iterations of optimization.
            False indicates verbosity.
        print_level: int: Default = 99:
            The degree of verbosity during print. Meaningless on in base.
        args
        kwargs
        """
        if backend is None:
            self.backend = pick_backend(backend, samples=samples, noise=noise,device=device)
        else:
            self.backend = backend

        if maxiter is None:
            self.maxiter = 100
        else:
            self.maxiter = maxiter

        if silent is None:
            self.silent = False
        else:
            self.silent = silent

        if print_level is None:
            self.print_level = 99
        else:
            self.print_level = print_level

        if silent:
            self.print_level = 0

        self.samples = samples
        self.save_history = save_history
        if save_history:
            self.history = OptimizerHistory()
        else:
            self.history = None

        self.noise = noise
        self.device = device
        self.args = args
        self.kwargs = kwargs

    def reset_history(self):
        """
        replace self.history with a blank history.

        Returns
        -------
        None
        """
        self.history = OptimizerHistory()

    def __call__(self, objective: Objective,
                 variables: typing.List[Variable],
                 initial_values: typing.Dict[Variable, numbers.Real] = None,
                 *args,
                 **kwargs) -> OptimizerResults:
        """
        Optimize some objective with the optimizer.

        Parameters
        ----------
        objective: Objective:
            The objective to optimize.
        variables: list:
            which variables to optimize over.
        initial_values: dict, optional:
            a starting point at which to begin optimization; a dict of variable, number pairs.
        args
        kwargs

        Returns
        -------
        OptimizerResults instance with "energy" "history" and "variables" as attributes
        see inheritors for more details.
        """
        raise TequilaOptimizerException("Tried to call BaseClass of Optimizer")

    def initialize_variables(self, objective, initial_values, variables):
        """
        Convenience function to format the variables of some objective recieved in calls to optimzers.

        Parameters
        ----------
        objective: Objective:
            the objective being optimized.
        initial_values: dict or string:
            initial values for the variables of objective, as a dictionary.
            if string: can be `zero` or `random`
            if callable: custom function that initializes when keys are passed
            if None: random initialization between 0 and 2pi (not recommended)
        variables: list:
            the variables being optimized over.

        Returns
        -------
        tuple:
            active_angles, a dict of those variables being optimized.
            passive_angles, a dict of those variables NOT being optimized.
            variables: formatted list of the variables being optimized.
        """
        # bring into right format
        variables = format_variable_list(variables)
        all_variables = objective.extract_variables()
        if variables is None:
            variables = all_variables
        if initial_values is None:
            initial_values = {k: numpy.random.uniform(0, 2 * numpy.pi) for k in all_variables}
        elif hasattr(initial_values, "lower"):
            if initial_values.lower() == "zero":
                initial_values = {k:0.0 for k in all_variables}
            elif "zero" in initial_values.lower():
                scale=0.1
                if "scale" in initial_values.lower():
                    # pass as: near_zero_scale=0.1_...
                    scale = float(initial_values.split("scale")[1].split("_")[0].split("=")[1])
                initial_values = {k: numpy.random.normal(loc=0.0, scale=scale) for k in all_variables}
            elif initial_values.lower() == "random":
                initial_values = {k: numpy.random.uniform(0.0, 4*numpy.pi) for k in all_variables}
            elif "random" in initial_values.lower():
                scale=2*numpy.pi
                loc=0.0
                if "scale" in initial_values.lower():
                    scale = float(initial_values.split("scale")[1].split("_")[0].split("=")[1])
                if "loc" in initial_values.lower():
                    loc = float(initial_values.split("loc")[1].split("_")[0].split("=")[1])
                initial_values = {k: numpy.random.normal(loc=loc, scale=scale) for k in all_variables}
            else:
                raise TequilaOptimizerException("unknown initialization instruction: {}".format(initial_values))
        elif callable(initial_values):
            initial_values = {k: initial_values(k) for k in all_variables}
        elif isinstance(initial_values, numbers.Number):
            initial_values = {k: initial_values for k in all_variables}
        else:
            # autocomplete initial values, warn if you did
            detected = False
            for k in all_variables:
                if k not in initial_values:
                    initial_values[k] = 0.0
                    detected = True
            if detected and not self.silent:
                warnings.warn("initial_variables given but not complete: Autocompleted with zeroes", TequilaWarning)
        initial_values = format_variable_dictionary(initial_values)

        active_angles = {}
        for v in variables:
            active_angles[v] = initial_values[v]

        passive_angles = {}
        for k, v in initial_values.items():
            if k not in active_angles.keys():
                passive_angles[k] = v
        return active_angles, passive_angles, variables

    def compile_objective(self, objective: Objective, *args, **kwargs):
        """
        convenience function to wrap over compile; for use by inheritors.
        Parameters
        ----------
        objective: Objective:
            an objective to compile.
        args
        kwargs

        Returns
        -------
        Objective:
            a compiled Objective. Types vary.
        """
        return compile(objective=objective,
                       samples=self.samples,
                       backend=self.backend,
                       device=self.device,
                       noise=self.noise,
                       *args, **kwargs)

    def compile_gradient(self, objective: Objective,
                         variables: typing.List[Variable],
                         gradient=None,
                         *args, **kwargs) -> typing.Tuple[
        typing.Dict, typing.Dict]:
        """
        convenience function to compile gradient objects and relavant types. For use by inheritors.

        Parameters
        ----------
        objective: Objective:
            the objective whose gradient is to be calculated.
        variables: list:
            the variables to take gradients with resepct to.
        gradient, optional:
            special argument to change what structure is used to calculate the gradient, like numerical, or QNG.
            Default: use regular, analytic gradients.
        args
        kwargs

        Returns
        -------
        tuple:
            both the uncompiled and compiled gradients of objective, w.r.t variables.
        """
        if gradient is None:
            dO = {k: grad(objective=objective, variable=k, *args, **kwargs) for k in variables}
            compiled_grad = {k: self.compile_objective(objective=dO[k], *args, **kwargs) for k in variables}

        elif isinstance(gradient, dict) or hasattr(gradient, "items"):
            if all([isinstance(x, Objective) for x in gradient.values()]):
                dO = gradient
                compiled_grad = {k: self.compile_objective(objective=dO[k], *args, **kwargs) for k in variables}
            elif 'method' in gradient and gradient['method'] == 'standard_spsa':
                dO = None
                compiled = self.compile_objective(objective=objective)
                compiled_grad = _SPSAGrad(objective=compiled, variables=variables, **gradient)
            else:
                dO = None
                compiled = self.compile_objective(objective=objective)
                compiled_grad = {k: _NumGrad(objective=compiled, variable=k, **gradient) for k in variables}
        else:
            raise TequilaOptimizerException(
                "unknown gradient instruction of type {} : {}".format(type(gradient), gradient))

        return dO, compiled_grad

    def compile_hessian(self,
                        variables: typing.List[Variable],
                        grad_obj: typing.Dict[Variable, Objective],
                        comp_grad_obj: typing.Dict[Variable, Objective],
                        hessian: dict = None,
                        *args,
                        **kwargs) -> tuple:
        """
        convenience function to compile hessians for optimizers which require it.
        Parameters
        ----------
        variables:
            the variables of the hessian.
        grad_obj:
            the gradient object, to be differentiated once more
        comp_grad_obj:
            the compiled gradient object, used for further compilation of the hessian.
        hessian: optional:
            extra information to modulate compilation of the hessian.
        args
        kwargs

        Returns
        -------
        tuple:
            uncompiled and compiled hessian objects, in that order
        """
        dO = grad_obj
        cdO = comp_grad_obj

        if hessian is None:
            if dO is None:
                raise TequilaOptimizerException("Can not combine analytical Hessian with numerical Gradient\n"
                                                "hessian instruction was: {}".format(hessian))

            compiled_hessian = {}
            ddO = {}
            for k in variables:
                dOk = dO[k]
                for l in variables:
                    ddO[(k, l)] = grad(objective=dOk, variable=l)
                    compiled_hessian[(k, l)] = self.compile_objective(ddO[(k, l)])
                    ddO[(l, k)] = ddO[(k, l)]
                    compiled_hessian[(l, k)] = compiled_hessian[(k, l)]

        elif isinstance(hessian, dict):
            if all([isinstance(x, Objective) for x in hessian.values()]):
                ddO = hessian
                compiled_hessian = {k: self.compile_objective(objective=ddO[k], *args, **kwargs) for k in
                                    hessian.keys()}
            else:
                ddO = None
                compiled_hessian = {}
                for k in variables:
                    for l in variables:
                        compiled_hessian[(k, l)] = _NumGrad(objective=cdO[k], variable=l, **hessian)
                        compiled_hessian[(l, k)] = _NumGrad(objective=cdO[l], variable=k, **hessian)
        else:
            raise TequilaOptimizerException("unknown hessian instruction: {}".format(hessian))

        return ddO, compiled_hessian

    def __repr__(self):
        infostring = "Optimizer: {} \n".format(str(type(self)))
        infostring += "{:15} : {}\n".format("backend", self.backend)
        infostring += "{:15} : {}\n".format("device", self.device)
        infostring += "{:15} : {}\n".format("samples", self.samples)
        infostring += "{:15} : {}\n".format("save_history", self.save_history)
        infostring += "{:15} : {}\n".format("noise", self.noise)
        return infostring


class _NumGrad:
    """ Numerical Gradient object.

    Should not be used outside of optimizers.
    Can't interact with other tequila structures.

    Attributes
    ----------

    objective:
        the objective whose gradient is to be approximated.
    variable:
        the variable with respect to which the gradient is taken.
    stepsize:
        the size of the small constant for shifting.
    method: how to approximate the gradient.


    Methods
    -------
    symmetric_two_point_stencil:
        get gradient by point + shift, point - shift
    forward_two_point_stencil:
        get gradient by point + shift, point.
    backward_two_point_stencil:
        get gradient by point, point -shift
    count_expectaionvalues:
        convenience; call the count_expectationvalues method of objective

    """

    def __init__(self, objective, variable, stepsize, method=None):
        """

        Parameters
        ----------
        objective: Objective:
            the objective whose gradient is to be approximated.
        variable:
            the variable the gradient of objective with respect to which is taken.
        stepsize:
            the small shift by which to displace variable around a point.
        method:
            the method by which to approximate the gradient.
        """
        self.objective = objective
        self.variable = variable
        self.stepsize = stepsize
        if method is None or method == "2-point":
            self.method = self.symmetric_two_point_stencil
        elif method is None or method == "2-point-forward":
            self.method = self.forward_two_point_stencil
        elif method is None or method == "2-point-backward":
            self.method = self.backward_two_point_stencil
        else:
            self.method = method

    @staticmethod
    def symmetric_two_point_stencil(obj, vars, key, step, *args, **kwargs):
        """
        calculate objective gradient by symmetric shifts about a point.
        Parameters
        ----------
        obj: Objective:
            objective to call.
        vars:
            variables to feed to the objective.
        key:
            which variable to shift, i.e, which variable's gradient is being called.
        step:
            the size of the shift; a small float.
        args
        kwargs

        Returns
        -------
        float:
            the approximated gradient of obj w.r.t var at point vars as a float.

        """
        left = copy.deepcopy(vars)
        left[key] += step / 2
        right = copy.deepcopy(vars)
        right[key] -= step / 2
        return 1.0 / step * (obj(left, *args, **kwargs) - obj(right, *args, **kwargs))

    @staticmethod
    def forward_two_point_stencil(obj, vars, key, step, *args, **kwargs):
        """
        calculate objective gradient by asymmetric upward shfit relative to some point.
        Parameters
        ----------
        obj: Objective:
            objective to call.
        vars:
            variables to feed to the objective.
        key:
            which variable to shift, i.e, which variable's gradient is being called.
        step:
            the size of the shift; a small float.
        args
        kwargs

        Returns
        -------
        float:
            the approximated gradient of obj w.r.t var at point vars as a float.

        """

        left = copy.deepcopy(vars)
        left[key] += step
        right = copy.deepcopy(vars)
        return 1.0 / step * (obj(left, *args, **kwargs) - obj(right, *args, **kwargs))

    @staticmethod
    def backward_two_point_stencil(obj, vars, key, step, *args, **kwargs):
        """
        calculate objective gradient by asymmetric downward shfit relative to some point.
        Parameters
        ----------
        obj: Objective:
            objective to call.
        vars:
            variables to feed to the objective.
        key:
            which variable to shift, i.e, which variable's gradient is being called.
        step:
            the size of the shift; a small float.
        args
        kwargs

        Returns
        -------
        the approximated gradient of obj w.r.t var at point vars as a float.

        """

        left = copy.deepcopy(vars)
        right = copy.deepcopy(vars)
        right[key] -= step
        return 1.0 / step * (obj(left, *args, **kwargs) - obj(right, *args, **kwargs))

    def __call__(self, variables, *args, **kwargs):
        """
        convenience function to call self.method, e.g one of the staticmethods of this class.

        Parameters
        ----------
        variables:
            the variables constitutive of the point at which numerical gradients of self.objective are to be taken
        args
        kwargs

        Returns
        -------
        type:
            generally, float, the result of the numerical gradient.
        """
        return self.method(self.objective, variables, self.variable, self.stepsize, *args, **kwargs)

    def count_expectationvalues(self, *args, **kwargs):
        """
        how many expectationvalues are in self.objective?
        Parameters
        ----------
        args
        kwargs

        Returns
        -------
        int:
            how many expectationvalues are in self.objective
        """
        return self.objective.count_expectationvalues(*args, **kwargs)

class _SPSAGrad(_NumGrad):
    """ Simultaneous Perturbation Stochastic Approximation Gradient object.

    Should not be used outside of optimizers.
    Can't interact with other tequila structures.

    Attributes
    ----------

    objective:
        the objective whose gradient is to be approximated.
    variables:
        the variables with respect to which the gradient is taken.
    stepsize:
        the size of the small constant for shifting.

    """

    def __init__(self, objective, variables, stepsize, gamma=None,method=None):
        """

        Parameters
        ----------
        objective: Objective:
            the objective whose gradient is to be approximated.
        variables:
            the variables the gradient of objective with respect to which is taken.
        stepsize:
            the small shift by which to displace variable around a point.
        nextIndex:
            Integer indicating the next index of the list stepsize to use
            if(nextIndex == -1) stepsize is a float
        """
        self.objective = objective
        self.variables = variables
        self.gamma = gamma

        if isinstance(stepsize, list):
            self.nextIndex = 0
        elif gamma != None:
            self.nextIndex = "adjust"
        else:
            self.nextIndex = -1
        self.stepsize = stepsize
        if method is None or method == "standard_spsa":
            self.method = self.standard_spsa
        else:
            self.method = method

    @staticmethod
    def standard_spsa(obj, vars, keys, step, *args, **kwargs):
        """
        calculate objective gradient using standar spsa.
        Parameters
        ----------
        obj: Objective:
            objective to call.
        vars:
            variables to feed to the objective.
        key:
            which variables to shift, i.e, which variable's gradient is being called.
        step:
            the size of the shift; a small float.
        args
        kwargs

        Returns
        -------
        the approximated gradient of obj w.r.t var at point vars as a float.

        """
        dim = len(keys)
        perturbation_vector = choices([-1,1],k = dim)
        left = copy.deepcopy(vars)
        right = copy.deepcopy(vars)
        for i, key in enumerate(keys):
            left[key] += perturbation_vector[i] * step
            right[key] -= perturbation_vector[i] * step
        numerator = obj(left, *args, **kwargs) - obj(right, *args, **kwargs)
        gradient = list()
        for i in range(dim):
            gradientComponent = numerator / (2 * step * perturbation_vector[i])
            gradient.append(gradientComponent)
        return gradient

    def __call__(self, variables, iteration=1, *args, **kwargs):
        """
        convenience function to call self.method, e.g one of the staticmethods of this class.

        Parameters
        ----------
        variables:
            the variables constitutive of the point at which numerical gradients of self.objective are to be taken
        args
        kwargs

        Returns
        -------
        type:
            generally, float, the result of the numerical gradient.
        """
        if(self.nextIndex != -1 and self.nextIndex != "adjust"):
            stepsize = self.stepsize[self.nextIndex]
            if(self.nextIndex != len(self.stepsize) - 1):
                self.nextIndex += 1
        elif(self.nextIndex == -1):
            stepsize = self.stepsize
        else:
            stepsize = self.stepsize / (iteration ** self.gamma)
   
        return self.method(self.objective, variables, self.variables, stepsize, *args, **kwargs)

    def calibrated_lr(self, lr, initial_value, max_iter, *args, **kwargs):
        """
        Calculates a calibrated learning rate for spsa
        Parameters
        ----------
        lr:
            learning rate (a variable in spsa related papers)
        initial_value:
            the initial values of the variables used in the optimization
        max_iter:
            number of iteration used for the calibration
        args
        kwargs

        Returns
        -------
        type:
            float: the learning rate calibrated
        """
        dim = len(initial_value)
        delta = 0
        if(self.nextIndex != -1 and self.nextIndex != "adjust"):
            stepsize = self.stepsize[0]
        else:
            stepsize = self.stepsize
 
        for i in range(max_iter):
            perturbation_vector = choices([-1,1],k = dim)
            left = copy.deepcopy(initial_value)
            right = copy.deepcopy(initial_value)
            for j, v in enumerate(initial_value):
                left[v] += perturbation_vector[j] * stepsize
                right[v] -= perturbation_vector[j] * stepsize
            numeratorLeft = self.objective(left, *args, **kwargs) 
            numeratorRight = self.objective(right, *args, **kwargs)
            delta += numpy.absolute(numeratorRight - numeratorLeft) / max_iter
        return lr * 2 * stepsize / delta 
