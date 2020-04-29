"""
BaseCalss for Optimizers
Suggestion, feel free to propose new things/changes
"""
import typing, numbers

from tequila.utils.exceptions import TequilaException
from tequila.simulators.simulator_api import compile
from tequila.objective.objective import assign_variable, Variable
from tequila.objective import Objective
from tequila.circuit.gradient import grad
from dataclasses import dataclass, field


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
                 backend_options: dict = None,
                 maxiter: int = None,
                 samples: int = None,
                 noise_model=None,
                 save_history: bool = True,
                 silent: typing.Union[bool, int] = False,
                 print_level: int = 99, *args, **kwargs):
        """
        :param backend: The quantum backend to use (None means autopick)
        :param backend_options: backend specific options can also be passed as keywords with `backend_optionname=...`
        :param maxiter: Maximum number of iterations
        :param samples: Number of Samples for the Quantum Backend takes (None means full wavefunction simulation)
        :param print_level: Allow customization in derived classes, is set to 0 if silent==True
        :param save_history: Save the optimization history in self.history
        :silent: Silence printout
        """

        if isinstance(backend, type):
            self.backend = backend()
        else:
            self.backend = backend

        self.backend_options = {}
        if backend_options is not None:
            self.backend_options = backend_options

        for k, v in kwargs:
            # detect if backend specific options where passed
            # as keyworkds
            # like e.g. `qiskit_backend=...'
            if self.backend.lower() in k:
                self.backend_options[k] = v

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

        self.noise_model = noise_model

    def reset_history(self):
        self.history = OptimizerHistory()

    def __call__(self, objective: Objective,
                 variabeles: typing.List[Variable],
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

    def update_parameters(self, parameters: typing.Dict[str, float], *args, **kwargs) -> typing.Dict[str, float]:
        """
        :param parameters: the parameters which will be updated
        :return: updated parameters
        """
        raise TequilaOptimizerException("Tried to call BaseClass of Optimizer")

    def compile_objective(self, objective: Objective, *args, **kwargs):
        return compile(objective=objective,
                       samples=self.samples,
                       backend=self.backend,
                       backend_options=self.backend_options,
                       noise_model=self.noise_model,
                       *args, **kwargs)

    def compile_gradient(self, objective: Objective, variables, gradient=None, *args, **kwargs) -> dict:
        if isinstance(gradient, dict):
            return {k: self.compile_objective(v) for k, v in gradient.items()}

        compiled_grad = {}
        for k in variables:
            dO = grad(objective=objective, variable=k, *args, **kwargs)
            compiled_grad[k] = self.compile_objective(objective=dO, *args, **kwargs)
        return compiled_grad

    def compile_hessian(self, objective: Objective, variables, *args, **kwargs) -> dict:
        compiled_hessian = {}
        for k in variables:
            dOk = grad(objective=objective, variable=k, *args, **kwargs)
            for l in variables:
                dOkl = grad(objective=dOk, variable=l)
                compiled = self.compile_objective(dOkl)
                compiled_hessian[(k, l)] = compiled
                compiled_hessian[(l, k)] = compiled
        return compiled_hessian

    def __repr__(self):
        infostring = "Optimizer: {} \n".format(str(type(self)))
        infostring += "{:30} : {:30}".format("backend", self.backend)
        infostring += "{:30} : {:30}".format("samples", self.samples)
        infostring += "{:30} : {:30}".format("save_history", self.save_history)
        infostring += "{:30} : {:30}".format("noise_model", self.noise_model)
