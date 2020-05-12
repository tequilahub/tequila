from tequila.objective.objective import Objective
from tequila.optimizers.optimizer_base import Optimizer

import typing
import numbers
from tequila.objective.objective import Variable
import warnings

warnings.simplefilter("ignore")
import GPyOpt
from GPyOpt.methods import BayesianOptimization

import numpy as np
from tequila.simulators.simulator_api import compile, pick_backend
from collections import namedtuple

GPyOptReturnType = namedtuple('GPyOptReturnType', 'energy angles history opt')


def array_to_objective_dict(objective, array, passives=None) -> typing.Dict[Variable, float]:
    op = objective.extract_variables()
    if passives is not None:
        for i, thing in enumerate(op):
            if thing in passives.keys():
                op.remove(thing)
    back = {v: array[:, i] for i, v in enumerate(op)}
    if passives is not None:
        for k, v in passives.items():
            back[k] = v
    return back


class OptimizerGpyOpt(Optimizer):

    @classmethod
    def available_methods(cls):
        return ['lbfgs', 'direct', 'cma']

    def __init__(self, maxiter=100, backend=None, save_history=True, minimize=True,
                 samples=None, noise=None, backend_options=None, silent=False):
        self._minimize = minimize
        super().__init__(backend=backend, maxiter=maxiter, samples=samples, save_history=save_history,
                         noise=noise, backend_options=backend_options, silent=silent)

    def get_domain(self, objective, passive_angles=None) -> typing.List[typing.Dict]:
        op = objective.extract_variables()
        if passive_angles is not None:
            for i, thing in enumerate(op):
                if thing in passive_angles.keys():
                    op.remove(thing)
        return [{'name': v, 'type': 'continuous', 'domain': (0, 2 * np.pi)} for v in op]

    def get_object(self, func, domain, method) -> GPyOpt.methods.BayesianOptimization:
        return BayesianOptimization(f=func, domain=domain, acquisition=method)

    def construct_function(self, objective, passive_angles=None) -> typing.Callable:
        return lambda arr: objective(backend=self.backend,
                                     variables=array_to_objective_dict(objective, arr, passive_angles),
                                     samples=self.samples,
                                     noise=self.noise)

    def redictify(self, arr, objective, passive_angles=None) -> typing.Dict:
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
                 method: str = 'lbfgs', *args, **kwargs) -> GPyOptReturnType:

        active_angles, passive_angles, variables = self.initialize_variables(objective, initial_values, variables)
        dom = self.get_domain(objective, passive_angles)

        O = compile(objective=objective, variables=initial_values, backend=self.backend,
                    noise=self.noise, samples=self.samples,
                    backend_options=self.backend_options)

        if not self.silent:
            print(self)
            print("{:15} : {}".format("method", method))
            print("{:15} : {} expectationvalues".format("Objective", O.count_expectationvalues()))

        f = self.construct_function(O, passive_angles)
        opt = self.get_object(f, dom, method)
        opt.run_optimization(self.maxiter, verbosity=not self.silent)
        if self.save_history:
            self.history.energies = opt.get_evaluations()[1].flatten()
            self.history.angles = [self.redictify(v, objective, passive_angles) for v in opt.get_evaluations()[0]]
        return GPyOptReturnType(energy=opt.fx_opt, angles=self.redictify(opt.x_opt, objective, passive_angles),
                                history=self.history, opt=opt)


def minimize(objective: Objective,
             maxiter: int,
             variables: typing.List = None,
             initial_values: typing.Dict = None,
             samples: int = None,
             backend: str = None,
             backend_options: dict = None,
             noise=None,
             method: str = 'lbfgs',
             silent: bool = False,
             *args,
             **kwargs
             ) -> GPyOptReturnType:
    """

    Parameters
    ----------
    objective: Objective :
        The tequila objective to optimize
    initial_values: typing.Dict[typing.Hashable, numbers.Real]: (Default value = None):
        Initial values as dictionary of Hashable types (variable keys) and floating point numbers. generates FIXED variables! if not provided,
        all variables will be optimized.
    variables: typing.List[typing.Hashable] :
         (Default value = None)
         List of Variables to optimize. If None, all variables optimized, and the passives command is over-ruled.
    samples: int :
         (Default value = None)
         samples/shots to take in every run of the quantum circuits (None activates full wavefunction simulation)
    maxiter: int :
         how many iterations of GPyOpt to run. Note: GPyOpt will override this as it sees fit.
    backend: str :
         (Default value = None)
         Simulator backend, will be automatically chosen if set to None
    noise: NoiseModel :
         (Default value = None)
        a noise model to apply to the circuits of Objective.
    method: str:
         (Default value = 'lbfgs')
         method of acquisition. Allowed arguments are 'lbfgs', 'DIRECT', and 'CMA'

    Returns
    -------

    """

    optimizer = OptimizerGpyOpt(samples=samples, backend=backend, maxiter=maxiter,
                                backend_options=backend_options,
                                noise=noise, silent=silent)
    return optimizer(objective=objective, initial_values=initial_values,
                     variables=variables,
                     method=method
                     )
