import scipy, numpy, typing, numbers
from tequila.objective import Objective
from tequila.objective.objective import assign_variable, Variable, format_variable_dictionary, format_variable_list
from .optimizer_base import Optimizer
from tequila.circuit.gradient import grad
from ._scipy_containers import _EvalContainer, _GradContainer, _HessContainer  #_QngContainer
from collections import namedtuple
from tequila.simulators.simulator_api import compile
from tequila.utils.exceptions import TequilaException
from tequila.circuit.noise import NoiseModel
#from tequila.tools.qng import qng_metric_tensor_blocks

class TequilaScipyException(TequilaException):
    """ """
    pass


SciPyReturnType = namedtuple('SciPyReturnType', 'energy angles history scipy_output')


class OptimizerSciPy(Optimizer):
    """ """
    gradient_free_methods = ['NELDER-MEAD', 'COBYLA', 'POWELL', 'SLSQP']
    gradient_based_methods = ['L-BFGS-B', 'BFGS', 'CG', 'TNC']
    hessian_based_methods = ["TRUST-KRYLOV", "NEWTON-CG", "DOGLEG", "TRUST-NCG", "TRUST-EXACT", "TRUST-CONSTR"]

    @classmethod
    def available_methods(cls):
        """:return: All tested available methods"""
        return cls.gradient_free_methods + cls.gradient_based_methods + cls.hessian_based_methods

    def __init__(self, method: str = "L-BFGS-B",
                 tol: numbers.Real = None,
                 method_options=None,
                 method_bounds=None,
                 method_constraints=None,
                 silent: bool = True,
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
        if hasattr(method, "upper"):
            self.method = method.upper()
        else:
            self.method = method
        self.tol = tol
        self.method_options = method_options

        if method_bounds is not None:
            method_bounds = {assign_variable(k): v for k, v in method_bounds.items()}
        self.method_bounds = method_bounds
        self.silent = silent

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
                 initial_values: typing.Dict[Variable, numbers.Real],
                 variables: typing.List[Variable],
                 gradient: typing.Dict[Variable, Objective] = None,
                 #qng: bool = False,
                 hessian: typing.Dict[typing.Tuple[Variable, Variable], Objective] = None,
                 samples: int = None,
                 backend: str = None,
                 noise: NoiseModel=None,
                 reset_history: bool = True) -> SciPyReturnType:
        """
        Optimizes with scipy and gives back the optimized angles
        Get the optimized energies over the history
        :param objective: The tequila Objective to minimize
        :param initial_valuesxx: initial values for the objective
        :param return_scipy_output: chose if the full scipy output shall be returned
        :param reset_history: reset the history before optimization starts (has no effect if self.save_history is False)
        :return: tuple of optimized energy ,optimized angles and scipy output
        """

        infostring = "Starting {method} optimization\n".format(method=self.method)
        infostring += "Objective: {} expectationvalues\n".format(objective.count_expectationvalues())

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
        param_keys, param_values = zip(*active_angles.items())
        param_values = numpy.array(param_values)

        bounds = None
        if self.method_bounds is not None:
            bounds = {k: None for k in active_angles}
            for k,v in self.method_bounds.items():
                if k in bounds:
                    bounds[k] = v
            infostring += "bounds : {}\n".format(self.method_bounds)
            names, bounds = zip(*bounds.items())
            assert (names == param_keys) # make sure the bounds are not shuffled

        # do the compilation here to avoid costly recompilation during the optimization
        compiled_objective = compile(objective=objective, variables=initial_values, backend=backend,
                                               noise_model=noise,
                                               samples=samples)

        E = _EvalContainer(objective=compiled_objective,
                           param_keys=param_keys,
                           samples=samples,
                           passive_angles=passive_angles,
                           save_history=self.save_history,
                           silent=self.silent)



        # compile gradients
        if self.method in self.gradient_based_methods + self.hessian_based_methods and not isinstance(gradient, str):
            compiled_grad_objectives = dict()
            if gradient is None:
                gradient = {assign_variable(k): grad(objective=objective, variable=k) for k in active_angles.keys()}
            else:
                gradient = {assign_variable(k): v for k, v in gradient.items()}

            grad_exval = []
            for k in active_angles.keys():
                if k not in gradient:
                    raise Exception("No gradient for variable {}".format(k))
                grad_exval.append(gradient[k].count_expectationvalues())
                compiled_grad_objectives[k] = compile(objective=gradient[k], variables=initial_values,
                                                           samples=samples,noise_model=noise, backend=backend)
            '''
            if qng:
                metric_tensor_blocks=qng_metric_tensor_blocks(objective,initial_values,samples=samples,noise_model=noise,
                                                backend=backend)
                dE = _QngContainer(objective=compiled_grad_objectives,
                                   metric_tensor_blocks=metric_tensor_blocks,
                                param_keys=param_keys,
                                samples=samples,
                                passive_angles=passive_angles,
                                save_history=self.save_history,
                                silent=self.silent)
            else:
            '''
            dE = _GradContainer(objective=compiled_grad_objectives,
                                param_keys=param_keys,
                                samples=samples,
                                passive_angles=passive_angles,
                                save_history=self.save_history,
                                silent=self.silent)

            infostring += "Gradients: {} expectationvalues (min={}, max={})\n".format(sum(grad_exval), min(grad_exval),
                                                                                      max(grad_exval))
        else:
            # use numerical gradient
            dE = gradient
            infostring += "Gradients: {}\n".format(gradient)

        # compile hessian

        if self.method in self.hessian_based_methods and not isinstance(hessian, str):

            if isinstance(gradient, str):
                raise TequilaScipyException("Can not use numerical gradients for Hessian based methods")
            #if qng is True:
                #raise TequilaScipyException('Quantum Natural Hessian not yet well-defined, sorry!')
            compiled_hess_objectives = dict()
            hess_exval = []
            for i, k in enumerate(active_angles.keys()):
                for j, l in enumerate(active_angles.keys()):
                    if j > i: continue
                    hess = grad(gradient[k], l)
                    compiled_hess = compile(objective=hess, variables=initial_values, samples=samples,
                                                      noise_model=noise,
                                                      backend=backend)
                    compiled_hess_objectives[(k, l)] = compiled_hess
                    compiled_hess_objectives[(l, k)] = compiled_hess
                    hess_exval.append(compiled_hess.count_expectationvalues())

            ddE = _HessContainer(objective=compiled_hess_objectives,
                                 param_keys=param_keys,
                                 samples=samples,
                                 passive_angles=passive_angles,
                                 save_history=self.save_history,
                                 silent=self.silent)

            infostring += "Hessian: {} expectationvalues (min={}, max={})\n".format(sum(hess_exval), min(hess_exval),
                                                                                    max(hess_exval))

        else:
            infostring += "Hessian: {}\n".format(hessian)
            if self.method is not "TRUST-CONSTR" and hessian is not None:
                raise TequilaScipyException("numerical hessians only for trust-constr method")
            ddE = hessian

        if not self.silent:
            print("ObjectiveType is {}".format(type(compiled_objective)))
            print(infostring)
            print("backend: {}".format(compiled_objective.backend))
            print("samples: {}".format(samples))
            print("{} active variables".format(len(active_angles)))

        # get the number of real scipy iterations for better histories
        real_iterations = []

        Es = []
        callback = lambda x, *args: real_iterations.append(len(E.history) - 1)
        res = scipy.optimize.minimize(E, x0=param_values, jac=dE, hess=ddE,
                                      args=(Es,),
                                      method=self.method, tol=self.tol,
                                      bounds=bounds,
                                      constraints=self.method_constraints,
                                      options=self.method_options,
                                      callback=callback)

        # failsafe since callback is not implemented everywhere
        if len(real_iterations) == 0:
            real_iterations = range(len(E.history))
        else:
            real_iterations = [0] + real_iterations
        if self.save_history:
            self.history.energies = [E.history[i] for i in real_iterations]
            self.history.energy_evaluations = E.history
            self.history.angles = [E.history_angles[i] for i in real_iterations]
            self.history.angles_evaluations = E.history_angles
            if dE is not None and not isinstance(dE, str):
                # can currently only save gradients if explicitly evaluated
                # and will fail for hessian based approaches
                # need better callback functions
                try:
                    if self.method not in self.hessian_based_methods:
                        self.history.gradients = [dE.history[i] for i in real_iterations]
                except:
                    print("WARNING: History could not assign the stored gradients")
                self.history.gradients_evaluations = dE.history
            if ddE is not None and not isinstance(ddE, str):
                # hessians are not evaluated in the same frequencies as energies
                # therefore we can not store the "real" iterations currently
                self.history.hessians_evaluations = ddE.history

        E_final = res.fun
        angles_final = dict((param_keys[i], res.x[i]) for i in range(len(param_keys)))
        angles_final = {**angles_final, **passive_angles}

        return SciPyReturnType(energy=E_final, angles=format_variable_dictionary(angles_final), history=self.history, scipy_output=res)


def available_methods(energy=True, gradient=True, hessian=True) -> typing.List[str]:
    """Convenience
    :return: Available methods of the scipy optimizer

    Parameters
    ----------
    energy :
        (Default value = True)
    gradient :
        (Default value = True)
    hessian :
        (Default value = True)

    Returns
    -------

    
    """
    methods = []
    if energy:
        methods += OptimizerSciPy.gradient_free_methods
    if gradient:
        methods += OptimizerSciPy.gradient_based_methods
    if hessian:
        methods += OptimizerSciPy.hessian_based_methods
    return methods


def minimize(objective: Objective,
             gradient: typing.Union[str, typing.Dict[Variable, Objective]] = None,
             hessian: typing.Union[str, typing.Dict[typing.Tuple[Variable, Variable], Objective]] = None,
             #qng: bool =None,
             initial_values: typing.Dict[typing.Hashable, numbers.Real] = None,
             variables: typing.List[typing.Hashable] = None,
             samples: int = None,
             maxiter: int = 100,
             backend: str = None,
             noise: NoiseModel =None,
             method: str = "BFGS",
             tol: float = 1.e-3,
             method_options: dict = None,
             method_bounds: typing.Dict[typing.Hashable, numbers.Real] = None,
             method_constraints=None,
             silent: bool = False,
             save_history: bool = True,
             *args,
             **kwargs) -> SciPyReturnType:
    """

    Parameters
    ----------
    objective: Objective :
        The tequila objective to optimize
    gradient: typing.Union[str, typing.Dict[Variable, Objective], None] : (Default value = None) :
        '2-point', 'cs' or '3-point' for numerical gradient evaluation (does not work in combination with all optimizers),
        dictionary of variables and tequila objective to define own gradient,
        None for automatic construction (default)
    hessian: typing.Union[str, typing.Dict[Variable, Objective], None] : (Default value = None) :
        '2-point', 'cs' or '3-point' for numerical gradient evaluation (does not work in combination with all optimizers),
        dictionary (keys:tuple of variables, values:tequila objective) to define own gradient,
        None for automatic construction (default)
    qng: bool : (Default value = False) :
        whether or not, in the event that a gradient-based method is to be used, the qng, rather than the standard gradient,
        should be employed. NOTE: throws an error for anything but a single expectationvalue with no passive angles.
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
    tol: float :
         (Default value = 1.e-3)
         Convergence tolerance for optimization (see scipy documentation)
    method_options: dict :
         (Default value = None)
         Dictionary of options
         (see scipy documentation)
    method_bounds: typing.Dict[typing.Hashable, typing.Tuple[float, float]]:
        (Default value = None)
        bounds for the variables (see scipy documentation)
    method_constraints :
         (Default value = None)
         (see scipy documentation
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
    if isinstance(gradient, dict) or hasattr(gradient, "items"):
        gradient = format_variable_dictionary(gradient)
    if isinstance(hessian, dict) or hasattr(hessian, "items"):
        hessian = {(assign_variable(k[0]), assign_variable([k[1]])): v for k, v in hessian.items()}
    method_bounds = format_variable_dictionary(method_bounds)

    # set defaults
    all_variables = objective.extract_variables()
    if variables is None:
        variables = all_variables
    if initial_values is None:
        initial_values = {k: numpy.random.uniform(0,2*numpy.pi) for k in all_variables}
    else:
        # autocomplete initial values, warn if you did
        detected = False
        for k in all_variables:
            if k not in initial_values:
                initial_values[k] = numpy.random.uniform(0,2*numpy.pi)
                detected = True
        if detected and not silent:
            print("WARNING: initial_variables given but not complete: Autocomplete with random number")

    optimizer = OptimizerSciPy(save_history=save_history,
                               maxiter=maxiter,
                               method=method,
                               method_options=method_options,
                               method_bounds=method_bounds,
                               method_constraints=method_constraints,
                               silent=silent,
                               simulator=backend,
                               tol=tol)
    if initial_values is not None:
        initial_values = {assign_variable(k): v for k, v in initial_values.items()}
    return optimizer(objective=objective,backend=backend, gradient=gradient,hessian=hessian, initial_values=initial_values,
                     variables=variables,noise=noise,
                     samples=samples)
