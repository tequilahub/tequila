import scipy, numpy, typing, numbers
from tequila.objective import Objective
from tequila.objective.objective import assign_variable, Variable, format_variable_dictionary, format_variable_list
from .optimizer_base import Optimizer
from tequila.circuit.gradient import grad
from ._scipy_containers import _EvalContainer, _GradContainer, _HessContainer
from collections import namedtuple
from tequila.simulators import compile_objective, simulate_objective
import copy

SciPyReturnType = namedtuple('SciPyReturnType', 'energy angles history scipy_output')


class OptimizerSciPy(Optimizer):
    gradient_free_methods = ['NELDER-MEAD', 'COBYLA', 'POWELL', 'SLSQP']
    gradient_based_methods = ['L-BFGS-B', 'BFGS', 'CG', 'TNC']
    hessian_based_methods = ["TRUST-KRYLOV", "NEWTON-CG", "DOGLEG", "TRUST-NCG", "TRUST-EXACT", "TRUST-CONSTR"]
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
                 use_hessian: bool = None,
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

        # set defaults
        default_use_gradient = False
        default_use_hessian = False
        if method.upper() in self.gradient_based_methods + self.hessian_based_methods:
            default_use_gradient = True
            if method.upper() in self.hessian_based_methods:
                default_use_hessian = True

        # overwrite if use_gradient/use_hessian are explicitly given (proceed on own risk)
        # if you chose a method which requires gradients/hessian scipy will evaluate numerically
        # if possible
        if use_gradient is None:
            self.use_gradient = default_use_gradient
        else:
            self.use_gradient = use_gradient
            #failsafe
            if use_gradient is False and method.upper() in self.gradient_based_methods + self.hessian_based_methods:
                self.use_gradient = '2-point'
        if use_hessian is None:
            self.use_hessian = default_use_hessian
        else:
            self.use_hessian = use_hessian
            # failsafe
            if use_hessian is False and method.upper() in self.hessian_based_methods:
                self.use_hessian = '2-point'

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
                 gradient: typing.Dict[Variable, Objective] = None,
                 initial_values: typing.Dict[Variable, numbers.Real] = None,
                 variables: typing.List[Variable] = None,
                 samples: int = None,
                 backend: str = None,
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
        infostring += "use_gradient : {}\n".format(self.use_gradient)
        infostring += "use_hessian  : {}\n".format(self.use_hessian)
        infostring += "Objective: {} expectationvalues\n".format(objective.count_expectationvalues())

        if self.save_history and reset_history:
            self.reset_history()

        if variables is None:
            variables = objective.extract_variables()

        # Extract initial values
        angles = initial_values
        if initial_values is None:
            angles = {v: 0.0 for v in objective.extract_variables()}

        active_angles = {}
        for v in variables:
            active_angles[v] = angles[v]

        passive_angles = {}
        for k, v in angles.items():
            if k not in active_angles.keys():
                passive_angles[k] = v

        # do the compilation here to avoid costly recompilation during the optimization
        compiled_objective = compile_objective(objective=objective, variables=angles, backend=backend,
                                               samples=samples)

        grad_exval = []
        compiled_grad_objectives = None
        if self.use_gradient is True:
            compiled_grad_objectives = dict()
            if gradient is None:
                gradient = {assign_variable(k): grad(objective=objective, variable=k) for k in active_angles.keys()}
            else:
                gradient = {assign_variable(k): v for k, v in gradient.items()}

            for k in active_angles.keys():
                if k not in gradient:
                    raise Exception("No gradient for variable {}".format(k))
                grad_exval.append(gradient[k].count_expectationvalues())
                compiled_grad_objectives[k] = compile_objective(objective=gradient[k], variables=angles,
                                                                samples=samples, backend=backend)

            infostring += "Gradients: {} expectationvalues (min={}, max={})\n".format(sum(grad_exval), min(grad_exval),
                                                                                      max(grad_exval))

        compiled_hess_objective = None
        if self.use_hessian is True:
            compiled_hess_objective = dict()
            for i, k in enumerate(active_angles.keys()):
                for j, l in enumerate(active_angles.keys()):
                    if j > i: continue
                    hess = grad(gradient[k], l)
                    compiled_hess = compile_objective(objective=hess, variables=angles, samples=samples,
                                                       backend=backend)
                    compiled_hess_objective[(k, l)] = compiled_hess
                    compiled_hess_objective[(l, k)] = compiled_hess

        # Transform the initial value directory into (ordered) arrays
        param_keys, param_values = zip(*active_angles.items())
        param_values = numpy.array(param_values)

        # Make E, grad
        if self.method in self.gradient_based_methods + self.hessian_based_methods:
            dE = self.use_gradient
        else:
            dE = None
        if self.method in self.hessian_based_methods:
            ddE = self.use_hessian
        else:
            ddE = None
        Es = []
        E = _EvalContainer(objective=compiled_objective,
                           param_keys=param_keys,
                           samples=samples,
                           passive_angles=passive_angles,
                           save_history=self.save_history,
                           silent=self.silent)
        if self.use_gradient is True:
            dE = _GradContainer(objective=compiled_grad_objectives,
                                param_keys=param_keys,
                                samples=samples,
                                passive_angles=passive_angles,
                                save_history=self.save_history,
                                silent=self.silent)
        if self.use_hessian is True:
            ddE = _HessContainer(objective=compiled_hess_objective,
                                 param_keys=param_keys,
                                 samples=samples,
                                 passive_angles=passive_angles,
                                 save_history=self.save_history,
                                 silent=self.silent)

        bounds = None
        if self.method_bounds is not None:
            names, bounds = zip(*self.method_bounds.items())
            assert (names == param_keys)

        if not self.silent:
            print("ObjectiveType is {}".format(type(compiled_objective)))
            print(infostring)
            print("backend: {}".format(compiled_objective.backend))
            print("samples: {}".format(samples))
            print("{} active variables".format(len(active_angles)))

        # get the number of real scipy iterations for better histories
        real_iterations = [0]

        res = scipy.optimize.minimize(E, param_values, jac=dE, hess=ddE,
                                      args=(Es,),
                                      method=self.method, tol=self.tol,
                                      bounds=bounds,
                                      constraints=self.method_constraints,
                                      options=self.method_options,
                                      callback=lambda x, *args: real_iterations.append(len(E.history) - 1))

        # failsafe since callback is not implemented everywhere
        if len(real_iterations) == 0:
            real_iterations = range(len(E.history))
        if self.save_history:
            self.history.energies = [E.history[i] for i in real_iterations]
            self.history.energy_evaluations = E.history
            self.history.angles = [E.history_angles[i] for i in real_iterations]
            self.history.angles_evaluations = E.history_angles
            if self.use_gradient is True:
                # can currently only save gradients if explicitly evaluated
                # and will fail for hessian based approaches
                # need better callback functions
                if self.method not in self.hessian_based_methods:
                    self.history.gradients = [dE.history[i] for i in real_iterations]
                self.history.gradients_evaluations = dE.history
            if self.use_hessian is True:
                # hessians are not evaluated in the same frequencies as energies
                # therefore we can not store the "real" iterations currently
                self.history.hessians_evaluations = ddE.history

        E_final = res.fun
        angles_final = dict((param_keys[i], res.x[i]) for i in range(len(param_keys)))
        angles_final = {**angles_final, **passive_angles}

        return SciPyReturnType(energy=E_final, angles=angles_final, history=self.history, scipy_output=res)


def available_methods():
    """
    Convenience
    :return: Available methods of the scipy optimizer (lists all gradient free and gradient based methods)
    """
    return OptimizerSciPy.available_methods()


def minimize(objective: Objective,
             gradient: typing.Dict[Variable, Objective] = None,
             initial_values: typing.Dict[typing.Hashable, numbers.Real] = None,
             variables: typing.List[typing.Hashable] = None,
             samples: int = None,
             maxiter: int = 100,
             backend: str = None,
             method: str = "BFGS",
             tol: float = 1.e-3,
             method_options: dict = None,
             method_bounds: typing.Dict[typing.Hashable, numbers.Real] = None,
             method_constraints=None,
             save_history: bool = True,
             silent: bool = False,
             use_gradient: bool = None,
             use_hessian: bool = None) -> SciPyReturnType:
    """
    Call this if you don't like objects
    :param objective: The tequila Objective to minimize
    :param gradient: The gradient of the Objective as other Objective, if None it will be created automatically;
    Only pass the gradient manually if there is a more efficient way
    :param initial_values: initial values for the objective
    :param variables: list of variables to optimize, None means all will be optimized
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

    # bring into right format
    variables = format_variable_list(variables)
    initial_values = format_variable_dictionary(initial_values)
    gradient = format_variable_dictionary(gradient)
    method_bounds = format_variable_dictionary(method_bounds)

    optimizer = OptimizerSciPy(save_history=save_history,
                               maxiter=maxiter,
                               method=method,
                               method_options=method_options,
                               method_bounds=method_bounds,
                               method_constraints=method_constraints,
                               silent=silent,
                               simulator=backend,
                               tol=tol,
                               use_gradient = use_gradient,
                               use_hessian = use_hessian)
    if initial_values is not None:
        initial_values = {assign_variable(k): v for k, v in initial_values.items()}
    return optimizer(objective=objective, gradient=gradient, initial_values=initial_values, variables=variables,
                     samples=samples)
