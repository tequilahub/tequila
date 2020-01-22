import scipy, numpy, typing, numbers
from tequila.objective import Objective
from tequila.objective.objective import assign_variable, Variable
from .optimizer_base import Optimizer
from tequila.circuit.gradient import grad
from ._scipy_containers import _EvalContainer, _GradContainer
from collections import namedtuple
from tequila.simulators import compile_objective, simulate_objective
import copy

SciPyReturnType = namedtuple('SciPyReturnType', 'energy angles history scipy_output')


class OptimizerSciPy(Optimizer):
    gradient_free_methods = ['NELDER-MEAD', 'COBYLA', 'POWELL', 'SLSQP']
    gradient_based_methods = ['L-BFGS-B', 'BFGS', 'CG', 'DOGLEG', 'TNC']

    @classmethod
    def available_methods(cls):
        """
        :return: All tested available methods
        """
        return cls.gradient_free_methods + cls.gradient_based_methods

    def __init__(self, method: str = "L-BFGS-B",
                 tol: numbers.Real = None,
                 method_options=None,
                 method_bounds=None,
                 method_constraints=None,
                 silent: bool = True,
                 use_gradient: bool = None,
                 **kwargs):
        """
        Optimize a circuit to minimize a given objective using scipy
        See the Optimizer class for all other parameters to initialize
        :param method: The scipy method passed as string
        :param use_gradient: do gradient based optimization
        :param tol: See scipy documentation for the method you picked
        :param method_options: See scipy documentation for the method you picked
        :param method_bounds: See scipy documentation for the method you picked
        :param method_constraints: See scipy documentation for the method you picked
        :param silent: if False the optimizer print out all evaluated energies
        :param use_gradient: select if gradients shall be used. Can be done automatically for most methods
        """
        super().__init__(**kwargs)
        self.method = method.upper()
        self.tol = tol
        self.method_options = method_options

        if method_bounds is not None:
            method_bounds = {assign_variable(k): v for k, v in method_bounds.items()}
        self.method_bounds = method_bounds
        self.silent = silent
        if use_gradient is None:
            if method.upper() in self.gradient_based_methods:
                self.use_gradient = True
            elif method.upper() in self.gradient_free_methods:
                self.use_gradient = False
            else:
                self.use_gradient = True
        else:
            self.use_gradient = use_gradient

        if method_options is None:
            self.method_options = {'maxiter': self.maxiter}
        else:
            self.method_options = method_options
            if 'maxiter' not in method_options:
                self.method_options['maxiter'] = self.maxiter
        self.method_options['disp'] = not silent

        if method_constraints is None:
            self.method_constraints = ()
        else:
            self.method_constraints = method_constraints

    def __call__(self, objective: Objective,
                 initial_values: typing.Dict[Variable, numbers.Real] = None,
                 reset_history: bool = True) -> SciPyReturnType:
        """
        Optimizes with scipi and gives back the optimized angles
        Get the optimized energies over the history
        :param objective: The tequila Objective to minimize
        :param initial_values: initial values for the objective
        :param return_scipy_output: chose if the full scipy output shall be returned
        :param reset_history: reset the history before optimization starts (has no effect if self.save_history is False)
        :return: tuple of optimized energy ,optimized angles and scipy output
        """

        infostring = "Starting {method} optimization\n".format(method=self.method)
        infostring += "Objective: {} expectationvalues\n".format(objective.count_expectationvalues())

        if self.save_history and reset_history:
            self.reset_history()

        # Extract initial values
        angles = initial_values
        if initial_values is None:
            variables = objective.extract_variables()
            angles = {v: 0.0 for v in variables}
        else:
            # convenience which allows the user to just pass down the keys of Variables
            angles = {assign_variable(k): v for k, v in initial_values.items()}

        # do the compilation here to avoid costly recompilation during the optimization
        compiled_objective = compile_objective(objective=objective, variables=angles, backend=self.simulator,
                                               samples=self.samples)

        grad_exval = []
        compiled_grad_objectives = dict()
        if self.use_gradient:
            for k in angles.keys():
                dO = grad(objective=objective, variable=k)
                grad_exval.append(dO.count_expectationvalues())
                compiled_grad_objectives[k] = compile_objective(objective=dO, variables=angles, samples=self.samples, backend=self.simulator)

            infostring += "Gradients: {} expectationvalues (min={}, max={})\n".format(sum(grad_exval), min(grad_exval), max(grad_exval))

        # Transform the initial value directory into (ordered) arrays
        param_keys, param_values = zip(*angles.items())
        param_values = numpy.array(param_values)

        # Make E, grad E
        dE = None
        Es = []
        E = _EvalContainer(objective=compiled_objective,
                           param_keys=param_keys,
                           samples=self.samples,
                           save_history=self.save_history,
                           silent=self.silent)
        if self.use_gradient:
            dE = _GradContainer(objective=compiled_grad_objectives,
                                param_keys=param_keys,
                                samples=self.samples,
                                save_history=self.save_history,
                                silent=self.silent)

        bounds = None
        if self.method_bounds is not None:
            names, bounds = zip(*self.method_bounds.items())
            assert (names == param_keys)

        if not self.silent:
            print(infostring)

        # get the number of real scipy iterations for better histories
        real_iterations = []
        res = scipy.optimize.minimize(E, param_values, jac=dE,
                                      args=(Es,),
                                      method=self.method, tol=self.tol,
                                      bounds=bounds,
                                      constraints=self.method_constraints,
                                      options=self.method_options,
                                      callback=lambda x: real_iterations.append(len(E.history) - 1))

        # failsafe since callback is not implemented everywhere
        if len(real_iterations) == 0:
            real_iterations = range(len(E.history))
        if self.save_history:
            self.history.energies = [E.history[i] for i in real_iterations]
            self.history.energy_evaluations = E.history
            self.history.angles = [E.history_angles[i] for i in real_iterations]
            self.history.angles_evaluations = E.history_angles
            if self.use_gradient:
                self.history.gradients = [dE.history[i] for i in real_iterations]
                self.history.gradients_evaluations = dE.history

        E_final = res.fun
        angles_final = dict((param_keys[i], res.x[i]) for i in range(len(param_keys)))

        return SciPyReturnType(energy=E_final, angles=angles_final, history=self.history, scipy_output=res)


def available_methods():
    """
    Convenience
    :return: Available methods of the scipy optimizer (lists all gradient free and gradient based methods)
    """
    return OptimizerSciPy.available_methods()


def minimize(objective: Objective,
             initial_values: typing.Dict[str, numbers.Real] = None,
             samples: int = None,
             maxiter: int = 100,
             backend: str = None,
             method: str = "L-BFGS-B",
             tol: float = 1.e-3,
             method_options: dict = None,
             method_bounds: typing.Dict[str, numbers.Real] = None,
             method_constraints=None,
             save_history: bool = True,
             silent: bool = False) -> SciPyReturnType:
    """
    Call this if you don't like objects
    :param objective: The tequila Objective to minimize
    :param initial_values: initial values for the objective
    :param samples: Number of samples to measure in each simulators run (None means full wavefunction simulation)
    :param maxiter: maximum number of iterations (can also be set over method_options)
    Note that some SciPy optimizers also accept 'maxfun' which is the maximum number of function evaluation
    You might consider massing down that keyword in the method_options dictionary
    :param backend: The quantum simulator you want to use (None -> automatically assigned)
    :param method: The scipy method passed as string
    :param tol: See scipy documentation for the method you picked
    :param method_options: See scipy documentation for the method you picked
    :param method_bounds: See scipy documentation for the method you picked
    Give in the same format as parameters/initial_values: Dict[hashable_type, float]
    :param return_dictionary: return results as dictionary instead of tuples
    :param method_constraints: See scipy documentation for the method you picked
    :param silent: If False the optimizer prints out evaluated energies
    :return: Named Tuple with: Optimized Energy, optimized angles, history (if return_history is True, scipy_output (if return_scipy_output is True)
    """
    optimizer = OptimizerSciPy(save_history=save_history,
                               samples=samples,
                               maxiter=maxiter,
                               method=method,
                               method_options=method_options,
                               method_bounds=method_bounds,
                               method_constraints=method_constraints,
                               silent=silent,
                               simulator=backend,
                               tol=tol)
    if initial_values is not None:
        initial_values = {assign_variable(k): v for k,v in initial_values.items()}
    return optimizer(objective=objective, initial_values=initial_values)
