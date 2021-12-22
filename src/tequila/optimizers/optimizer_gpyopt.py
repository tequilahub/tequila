from tequila.objective.objective import Objective
from tequila.optimizers.optimizer_base import Optimizer, OptimizerResults, dataclass
import typing
import numbers
from tequila.objective.objective import Variable
import warnings

warnings.simplefilter("ignore")
import GPyOpt
from GPyOpt.methods import BayesianOptimization
import numpy as np
from tequila.utils import to_float


@dataclass
class GPyOptResults(OptimizerResults):
    gpyopt_instance: GPyOpt.methods.BayesianOptimization = None


def array_to_objective_dict(objective, array, passives=None) -> typing.Dict[Variable, float]:
    """
    reformats a numpy array of parameters to a dictionary in so that objective might use it as variables.

    Parameters
    ----------
    objective: Objective:
        an Objective, whose parameters the array is meant to represent
    array: numpy.ndarray
        a numpy array of parameters.
    passives: optional:
        a dictionary of passive parameters to suppled the suggested parameters of array.
        Default means no passives involved.

    Returns
    -------
    dict:
        dictinary of formatted parameters for use by objective
    """
    op = objective.extract_variables()
    if passives is not None:
        for i, thing in enumerate(op):
            if thing in passives.keys():
                op.remove(thing)
    back = {v: array[:, i] for i, v in enumerate(op)}
    if passives is not None:
        for k, v in passives.items():
            back[k] = v
    back = {k: to_float(v) for k, v in back.items()}
    return back


class OptimizerGPyOpt(Optimizer):
    """
    Wrapper around the optimization package GPyOpt. See: https://github.com/SheffieldML/GPyOpt and Optimizer.

    Methods
    -------
    get_domain:
        initialize a domain for use by the gpyopt optimizer; a list of dicts about parameters to optimize over.
    get_object:
        return a GPyOpt BayesianOptimization object from prepared information.
    construct_function:
        return a tequila Objective as a callable function of a single numpy array.
    redictify:
        transform an array of parameters into a dictionary.
    """

    @classmethod
    def available_methods(cls):
        return ['gpyopt-lbfgs', 'gpyopt-direct', 'gpyopt-cma']

    def __init__(self, maxiter=100, backend=None,
                 samples=None, noise=None, device=None,
                 save_history=True, silent=False):

        """

        Parameters
        ----------
        maxiter: int: Default = 100:
            maximum number of iterations to performed.
        backend: str, optional:
            a quantum backend to use. None means autopick.
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
            whether or not to save this history of the optimization.
        silent: bool: Default = False:
            suppresses printouts if true.
        """
        super().__init__(backend=backend, maxiter=maxiter, samples=samples, save_history=save_history, device=device,
                         noise=noise, silent=silent)

    def get_domain(self, objective: Objective, passive_angles: dict = None) -> typing.List[typing.Dict]:
        """
        return a 'domain' object, for use by GPyOpt.

        This function constructs a list of dictionaries about each variable in objective to optimize over:
        we enforce the domain of 0 to 2 pi, the period of a rotation, since some domain MUST be specified.

        Parameters
        ----------
        objective: Objective:
            the Objective to extract variables from to build the domain.
        passive_angles: dict, optional:
            a dictionary of which angles are passive, in Objective.
            Default: there are none; optimize all angles.



        Returns
        -------
        list of dicts
            the domain object for use by gpyopt.

        """
        op = objective.extract_variables()
        if passive_angles is not None:
            for i, thing in enumerate(op):
                if thing in passive_angles.keys():
                    op.remove(thing)
        return [{'name': v, 'type': 'continuous', 'domain': (0, 2 * np.pi)} for v in op]

    def get_object(self, func, domain, method) -> GPyOpt.methods.BayesianOptimization:
        """
        get a GPyOpt BayesianOptimization object to run optimization with.

        Parameters
        ----------
        func: callable:
            the function to optimize.
        domain: list:
            the domain of optimization; a list of dicts.
        method: str:
            what optimization method to use.

        Returns
        -------
        a BayesianOptimization object.
        """
        return BayesianOptimization(f=func, domain=domain, acquisition=method)

    def construct_function(self, objective, passive_angles=None) -> typing.Callable:
        """
        return an objective as a callable function of a numpy array.

        Parameters
        ----------
        objective: Objective:
            an objective.
        passive_angles: dict, optional:
            the passive angles of objective.
        Returns
        -------
        callable.
        """
        return lambda arr: objective(backend=self.backend,
                                     variables=array_to_objective_dict(objective, arr, passive_angles),
                                     samples=self.samples,
                                     noise=self.noise)

    def redictify(self, arr, objective, passive_angles=None) -> typing.Dict:
        """
        turn an array back into a dictionary of parameters corresponding to the variables of an objective.

        Parameters
        ----------
        arr:
            a numpy array.
        objective: Objective:
            an objective.
        passive_angles: dict, optional:
            supplements array with the passive angles of objective.
        Returns
        -------
        a dictionary of parameters.
        """
        op = objective.extract_variables()
        if passive_angles is not None:
            for i, thing in enumerate(op):
                if thing in passive_angles.keys():
                    op.remove(thing)
        back = {v: arr[i] for i, v in enumerate(op)}
        if passive_angles is not None:
            for k, v in passive_angles.items():
                back[k] = v
        return back

    def __call__(self, objective: Objective,
                 initial_values: typing.Dict[Variable, numbers.Real] = None,
                 variables: typing.List[typing.Hashable] = None,
                 method: str = 'lbfgs', *args, **kwargs) -> GPyOptResults:

        """
        perform optimization of an objective via GPyOpt.

        Parameters
        ----------
        objective: Objective:
            the objective to optimize.
        initial_values: dict, optional:
            a starting point for optimization.
            Default: generate at random.
        variables: list, optional:
            which variables to optimize over.
            If None: optimize over all variables.
        method: str: Default = 'lbfgs'
            what method to use for the acquisition function of the bayesian optimization.
            Default: use lbfgs.
        args
        kwargs

        Returns
        -------
        GPyOptResults.
            Results of the optimization.
        """
        objective = objective.contract()
        active_angles, passive_angles, variables = self.initialize_variables(objective, initial_values, variables)
        dom = self.get_domain(objective, passive_angles)

        O = self.compile_objective(objective=objective)

        if not self.silent:
            print(self)
            print("{:15} : {}".format("method", method))
            print("{:15} : {} expectationvalues".format("Objective", O.count_expectationvalues()))

        f = self.construct_function(O, passive_angles)
        opt = self.get_object(f, dom, method)

        method_options = {"max_iter": self.maxiter, "verbosity": not self.silent, "eps": 1.e-4}

        if "method_options" in kwargs:
            tmp = {**method_options, **kwargs["method_options"]}

        opt.run_optimization(**method_options)
        if self.save_history:
            self.history.energies = opt.get_evaluations()[1].flatten()
            self.history.angles = [self.redictify(v, objective, passive_angles) for v in opt.get_evaluations()[0]]
        return GPyOptResults(energy=opt.fx_opt, variables=self.redictify(opt.x_opt, objective, passive_angles),
                             history=self.history, gpyopt_instance=opt)


def minimize(objective: Objective,
             maxiter: int,
             variables: typing.List = None,
             initial_values: typing.Dict = None,
             samples: int = None,
             backend: str = None,
             noise=None,
             device: str = None,
             method: str = 'lbfgs',
             silent: bool = False,
             *args,
             **kwargs
             ) -> GPyOptResults:
    """
    Minimize an objective using GPyOpt.
    Parameters
    ----------
    objective: Objective :
        The tequila objective to optimize
    initial_values: typing.Dict[typing.Hashable, numbers.Real], optional:
        Initial values as dictionary of Hashable types (variable keys) and floating point numbers.
        generates FIXED variables! if not provided, all variables will be optimized.
    variables: typing.List[typing.Hashable], optional:
         List of Variables to optimize. If None, all variables optimized.
    samples: int, optional:
         samples/shots to take in every run of the quantum circuits (None activates full wavefunction simulation)
    maxiter: int :
         how many iterations of GPyOpt to run. Note: GPyOpt will override this as it sees fit.
    backend: str, optional:
         Simulator backend, will be automatically chosen if set to None
    noise: NoiseModel, optional:
        a noise model to apply to the circuits of Objective.
    device: optional:
        the device from which to (potentially, simulatedly) sample all quantum circuits employed in optimization.
    method: str: Default = 'lbfgs':
         method of acquisition. Allowed arguments are 'lbfgs', 'DIRECT', and 'CMA'

    Returns
    -------
    GPyOptResults:
        the results of an optimization.
    """

    optimizer = OptimizerGPyOpt(samples=samples, backend=backend, maxiter=maxiter,
                                device=device,
                                noise=noise, silent=silent)
    return optimizer(objective=objective, initial_values=initial_values,
                     variables=variables,
                     method=method
                     )
