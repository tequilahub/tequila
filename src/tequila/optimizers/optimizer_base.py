"""
BaseClss for Optimizers
Suggestion, feel free to propose new things/changes
"""
import typing, numbers, copy

from tequila.utils.exceptions import TequilaException
from tequila.simulators.simulator_api import compile, pick_backend
from tequila.objective import Objective
from tequila.circuit.gradient import grad
from dataclasses import dataclass, field
from tequila.objective.objective import assign_variable, Variable, format_variable_dictionary, format_variable_list
import numpy


class TequilaOptimizerException(TequilaException):
    pass


@dataclass
class OptimizerHistory:

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
    energies_calls: typing.List[numbers.Real] = field(default_factory=list)
    gradients_calls: typing.List[typing.Dict[str, numbers.Real]] = field(default_factory=list)
    angles_calls: typing.List[typing.Dict[str, numbers.Number]] = field(default_factory=list)

    def __add__(self, other):
        result = OptimizerHistory()
        result.energies = self.energies + other.energies
        result.gradients = self.gradients + other.gradients
        result.angles = self.angles + other.angles
        return result

    def __iadd__(self, other):
        self.energies += other.energies
        self.gradients += other.gradients
        self.angles += other.angles
        return self

    def extract_energies(self, *args, **kwargs) -> typing.Dict[numbers.Integral, numbers.Real]:
        return {i: e for i, e in enumerate(self.energies)}

    def extract_gradients(self, key: str) -> typing.Dict[numbers.Integral, numbers.Real]:
        """
        :param key: the key specifiying which gradient shall be extracted
        :return: dictionary with dictionary_key=iteration, dictionary_value=gradient[key]
        """
        gradients = {}
        for i, d in enumerate(self.gradients):
            if key in d:
                gradients[i] = d[assign_variable(key)]
        return gradients

    def extract_angles(self, key: str) -> typing.Dict[numbers.Integral, numbers.Real]:
        """
        :param key: the key specifiying which angle shall be extracted
        :return: dictionary with dictionary_key=iteration, dictionary_value=angle[key]
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
        Convenience function to plot the progress of the optimizer
        :param filename: if given plot to file, otherwise plot to terminal
        :param property: the property to plot, given as string
        :param key: for properties like angles and gradients you can specifiy which one you want to plot
        if set to none all keys are plotted. You can pass down single keys or lists of keys. DO NOT use tuples of keys or any other hashable list types
        give key as list if you want to plot multiple properties with different keys
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


class Optimizer:
    """
    Base Class for Tequila Optimizers
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
        :param backend: The quantum backend to use (None means autopick)
        :param maxiter: Maximum number of iterations
        :param samples: Number of Samples for the Quantum Backend takes (None means full wavefunction simulation)

        :param print_level: Allow customization in derived classes, is set to 0 if silent==True
        :param save_history: Save the optimization history in self.history
        :silent: Silence printout
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

        if self.silent:
            self.print_level = 0

        self.samples = samples
        self.save_history = save_history
        if save_history:
            self.history = OptimizerHistory()
        else:
            self.history = None

        self.noise = noise
        self.device = device

    def reset_history(self):
        self.history = OptimizerHistory()

    def __call__(self, objective: Objective,
                 variables: typing.List[Variable],
                 initial_values: typing.Dict[Variable, numbers.Real] = None,
                 *args,
                 **kwargs) -> typing.Tuple[
        numbers.Number, typing.Dict[str, numbers.Number]]:
        """
        Will try to solve and give back optimized parameters
        :param objective: tequila Objective object
        :param parameters: initial parameters, if none the optimizers uses what was set in the objective
        :return: tuple of optimial energy and optimal parameters
        """
        raise TequilaOptimizerException("Tried to call BaseClass of Optimizer")

    def initialize_variables(self, objective, initial_values, variables):
        # bring into right format
        variables = format_variable_list(variables)
        initial_values = format_variable_dictionary(initial_values)
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
            if detected and not self.silent:
                print("WARNING: initial_variables given but not complete: Autocomplete with random number")

        active_angles = {}
        for v in variables:
            active_angles[v] = initial_values[v]

        passive_angles = {}
        for k, v in initial_values.items():
            if k not in active_angles.keys():
                passive_angles[k] = v
        return active_angles, passive_angles, variables

    def compile_objective(self, objective: Objective, *args, **kwargs):

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

        if gradient is None:
            dO = {k: grad(objective=objective, variable=k, *args, **kwargs) for k in variables}
            compiled_grad = {k: self.compile_objective(objective=dO[k], *args, **kwargs) for k in variables}

        elif isinstance(gradient, dict):
            if all([isinstance(x, Objective) for x in gradient.values()]):
                dO = gradient
                compiled_grad = {k: self.compile_objective(objective=dO[k], *args, **kwargs) for k in variables}
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
        infostring += "{:15} : {}\n".format("samples", self.samples)
        infostring += "{:15} : {}\n".format("save_history", self.save_history)
        infostring += "{:15} : {}\n".format("noise", self.noise)
        return infostring


class _NumGrad:
    """
    Numerical Gradient
    Should not be used outside of optimizers
    Can't interact with the current tequila structures
    """

    def __init__(self, objective, variable, stepsize, method=None):
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
        left = copy.deepcopy(vars)
        left[key] += step / 2
        right = copy.deepcopy(vars)
        right[key] -= step / 2
        return 1.0 / step * (obj(left, *args, **kwargs) - obj(right, *args, **kwargs))

    @staticmethod
    def forward_two_point_stencil(obj, vars, key, step, *args, **kwargs):
        left = copy.deepcopy(vars)
        left[key] += step
        right = copy.deepcopy(vars)
        return 1.0 / step * (obj(left, *args, **kwargs) - obj(right, *args, **kwargs))

    @staticmethod
    def backward_two_point_stencil(obj, vars, key, step, *args, **kwargs):
        left = copy.deepcopy(vars)
        right = copy.deepcopy(vars)
        right[key] -= step
        return 1.0 / step * (obj(left, *args, **kwargs) - obj(right, *args, **kwargs))

    def __call__(self, variables, *args, **kwargs):
        return self.method(self.objective, variables, self.variable, self.stepsize, *args, **kwargs)

    def count_expectationvalues(self, *args, **kwargs):
        return self.objective.count_expectationvalues(*args, **kwargs)
