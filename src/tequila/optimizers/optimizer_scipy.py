import scipy, numpy, typing, numbers
from tequila.objective import Objective
from tequila.objective.objective import assign_variable, Variable, format_variable_dictionary, format_variable_list
from .optimizer_base import Optimizer, OptimizerResults
from ._containers import _EvalContainer, _GradContainer, _HessContainer, _QngContainer
from tequila.utils.exceptions import TequilaException
from tequila.circuit.noise import NoiseModel
from tequila.tools.qng import get_qng_combos

from dataclasses import dataclass

class TequilaScipyException(TequilaException):
    """ """
    pass

@dataclass
class SciPyResults(OptimizerResults):

    scipy_result: scipy.optimize.OptimizeResult = None



class OptimizerSciPy(Optimizer):
    """
    Class wrapping over the scipy optimizer for use by Tequila.

    Attributes
    ----------
    method:
        The scipy optimization method passed as string.
    tol:
        See scipy documentation for the method you picked
    method_options:
        See scipy documentation for the method you picked
    method_bounds:
        See scipy documentation for the method you picked
    method_constraints:
        See scipy documentation for the method you picked
    silent:
        if False, the optimizer prints out all evaluated energies
    """
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
                 **kwargs):
        """
        Parameters
        ----------
        method: str: Default = 'L-BFGS-B':
            The scipy optimization method passed as string.
        tol: float, optional:
            See scipy documentation for the method you picked
        method_options: optional:
            See scipy documentation for the method you picked
        method_bounds: optional:
            See scipy documentation for the method you picked
        method_constraints: optional:
            See scipy documentation for the method you picked
        silent: bool:
            if False the optimizer prints out all evaluated energies
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

        if method_options is None:
            self.method_options = {'maxiter': self.maxiter}
        else:
            self.method_options = method_options
            if 'maxiter' not in method_options:
                self.method_options['maxiter'] = self.maxiter

        self.method_options['disp'] = self.print_level > 0

        if method_constraints is None:
            self.method_constraints = ()
        else:
            self.method_constraints = method_constraints

    def __call__(self, objective: Objective,
                 variables: typing.List[Variable] = None,
                 initial_values: typing.Dict[Variable, numbers.Real] = None,
                 gradient: typing.Dict[Variable, Objective] = None,
                 hessian: typing.Dict[typing.Tuple[Variable, Variable], Objective] = None,
                 reset_history: bool = True,
                 *args,
                 **kwargs) -> SciPyResults:

        """
        Perform optimization using scipy optimizers.

        Parameters
        ----------
        objective: Objective:
            the objective to optimize.
        variables: list, optional:
            the variables of objective to optimize. If None: optimize all.
        initial_values: dict, optional:
            a starting point from which to begin optimization. Will be generated if None.
        gradient: optional:
            Information or object used to calculate the gradient of objective. Defaults to None: get analytically.
        hessian: optional:
            Information or object used to calculate the hessian of objective. Defaults to None: get analytically.
        reset_history: bool: Default = True:
            whether or not to reset all history before optimizing.
        args
        kwargs

        Returns
        -------
        ScipyReturnType:
            the results of optimization.
        """
        objective = objective.contract()
        infostring = "{:15} : {}\n".format("Method", self.method)
        infostring += "{:15} : {} expectationvalues\n".format("Objective", objective.count_expectationvalues())

        if gradient is not None:
            infostring += "{:15} : {}\n".format("grad instr", gradient)
        if hessian is not None:
            infostring += "{:15} : {}\n".format("hess_instr", hessian)

        if self.save_history and reset_history:
            self.reset_history()

        active_angles, passive_angles, variables = self.initialize_variables(objective, initial_values, variables)

        # Transform the initial value directory into (ordered) arrays
        param_keys, param_values = zip(*active_angles.items())
        param_values = numpy.array(param_values)

        # process and initialize scipy bounds
        bounds = None
        if self.method_bounds is not None:
            bounds = {k: None for k in active_angles}
            for k, v in self.method_bounds.items():
                if k in bounds:
                    bounds[k] = v
            infostring += "{:15} : {}\n".format("bounds", self.method_bounds)
            names, bounds = zip(*bounds.items())
            assert (names == param_keys)  # make sure the bounds are not shuffled

        # do the compilation here to avoid costly recompilation during the optimization
        compiled_objective = self.compile_objective(objective=objective, *args, **kwargs)
        E = _EvalContainer(objective=compiled_objective,
                           param_keys=param_keys,
                           samples=self.samples,
                           passive_angles=passive_angles,
                           save_history=self.save_history,
                           print_level=self.print_level)

        compile_gradient = self.method in (self.gradient_based_methods + self.hessian_based_methods)
        compile_hessian = self.method in self.hessian_based_methods

        dE = None
        ddE = None
        # detect if numerical gradients shall be used
        # switch off compiling if so
        if isinstance(gradient, str):
            if gradient.lower() == 'qng':
                compile_gradient = False
                if compile_hessian:
                    raise TequilaException('Sorry, QNG and hessian not yet tested together.')

                combos = get_qng_combos(objective, initial_values=initial_values, backend=self.backend,
                                        samples=self.samples, noise=self.noise)
                dE = _QngContainer(combos=combos, param_keys=param_keys, passive_angles=passive_angles)
                infostring += "{:15} : QNG {}\n".format("gradient", dE)
            else:
                dE = gradient
                compile_gradient = False
                if compile_hessian:
                    compile_hessian = False
                    if hessian is None:
                        hessian = gradient
                infostring += "{:15} : scipy numerical {}\n".format("gradient", dE)
                infostring += "{:15} : scipy numerical {}\n".format("hessian", ddE)

        if isinstance(gradient,dict) and "method" in gradient:
            if gradient['method'] == 'qng':
                func = gradient['function']
                compile_gradient = False
                if compile_hessian:
                    raise TequilaException('Sorry, QNG and hessian not yet tested together.')

                combos = get_qng_combos(objective,func=func, initial_values=initial_values, backend=self.backend,
                                        samples=self.samples, noise=self.noise)
                dE = _QngContainer(combos=combos, param_keys=param_keys, passive_angles=passive_angles)
                infostring += "{:15} : QNG {}\n".format("gradient", dE)

        if isinstance(hessian, str):
            ddE = hessian
            compile_hessian = False

        if compile_gradient:
            grad_obj, comp_grad_obj = self.compile_gradient(objective=objective, variables=variables, gradient=gradient, *args, **kwargs)
            expvals = sum([o.count_expectationvalues() for o in comp_grad_obj.values()])
            infostring += "{:15} : {} expectationvalues\n".format("gradient", expvals)
            dE = _GradContainer(objective=comp_grad_obj,
                                param_keys=param_keys,
                                samples=self.samples,
                                passive_angles=passive_angles,
                                save_history=self.save_history,
                                print_level=self.print_level)
        if compile_hessian:
            hess_obj, comp_hess_obj = self.compile_hessian(variables=variables,
                                                           hessian=hessian,
                                                           grad_obj=grad_obj,
                                                           comp_grad_obj=comp_grad_obj, *args, **kwargs)
            expvals = sum([o.count_expectationvalues() for o in comp_hess_obj.values()])
            infostring += "{:15} : {} expectationvalues\n".format("hessian", expvals)
            ddE = _HessContainer(objective=comp_hess_obj,
                                 param_keys=param_keys,
                                 samples=self.samples,
                                 passive_angles=passive_angles,
                                 save_history=self.save_history,
                                 print_level=self.print_level)
        if self.print_level > 0:
            print(self)
            print(infostring)
            print("{:15} : {}\n".format("active variables", len(active_angles)))

        Es = []

        optimizer_instance = self
        class SciPyCallback:
            energies = []
            gradients = []
            hessians = []
            angles = []
            real_iterations = 0

            def __call__(self, *args, **kwargs):
                self.energies.append(E.history[-1])
                self.angles.append(E.history_angles[-1])
                if dE is not None and not isinstance(dE, str):
                    self.gradients.append(dE.history[-1])
                if ddE is not None and not isinstance(ddE, str):
                    self.hessians.append(ddE.history[-1])
                self.real_iterations += 1
                if 'callback' in optimizer_instance.kwargs:
                    optimizer_instance.kwargs['callback'](E.history_angles[-1])

        callback = SciPyCallback()
        res = scipy.optimize.minimize(E, x0=param_values, jac=dE, hess=ddE,
                                      args=(Es,),
                                      method=self.method, tol=self.tol,
                                      bounds=bounds,
                                      constraints=self.method_constraints,
                                      options=self.method_options,
                                      callback=callback)

        # failsafe since callback is not implemented everywhere
        if callback.real_iterations == 0:
            real_iterations = range(len(E.history))

        if self.save_history:
            self.history.energies = callback.energies
            self.history.energy_calls = E.history
            self.history.angles = callback.angles
            self.history.angles_calls = E.history_angles
            self.history.gradients = callback.gradients
            self.history.hessians = callback.hessians
            if dE is not None and not isinstance(dE, str):
                self.history.gradient_calls = dE.history
            if ddE is not None and not isinstance(ddE, str):
                self.history.hessian_calls = ddE.history

            # some methods like "cobyla" do not support callback functions
            if len(self.history.energies) == 0:
                self.history.energies = E.history
                self.history.angles = E.history_angles

        # some scipy methods always give back the last value and not the minimum (e.g. cobyla)
        ea = sorted(zip(E.history, E.history_angles), key=lambda x: x[0])
        E_final = ea[0][0]
        angles_final = ea[0][1] #dict((param_keys[i], res.x[i]) for i in range(len(param_keys)))
        angles_final = {**angles_final, **passive_angles}

        return SciPyResults(energy=E_final, history=self.history, variables=format_variable_dictionary(angles_final), scipy_result=res)


def available_methods(energy=True, gradient=True, hessian=True) -> typing.List[str]:
    """Convenience
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
    Available methods of the scipy optimizer, a list of strings.
    
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
             initial_values: typing.Dict[typing.Hashable, numbers.Real] = None,
             variables: typing.List[typing.Hashable] = None,
             samples: int = None,
             maxiter: int = 100,
             backend: str = None,
             backend_options: dict = None,
             noise: NoiseModel = None,
             device: str = None,
             method: str = "BFGS",
             tol: float = 1.e-3,
             method_options: dict = None,
             method_bounds: typing.Dict[typing.Hashable, numbers.Real] = None,
             method_constraints=None,
             silent: bool = False,
             save_history: bool = True,
             *args,
             **kwargs) -> SciPyResults:
    """

    Parameters
    ----------
    objective: Objective :
        The tequila objective to optimize
    gradient: typing.Union[str, typing.Dict[Variable, Objective], None] : Default value = None):
        '2-point', 'cs' or '3-point' for numerical gradient evaluation (does not work in combination with all optimizers),
        dictionary of variables and tequila objective to define own gradient,
        None for automatic construction (default)
        Other options include 'qng' to use the quantum natural gradient.
    hessian: typing.Union[str, typing.Dict[Variable, Objective], None], optional:
        '2-point', 'cs' or '3-point' for numerical gradient evaluation (does not work in combination with all optimizers),
        dictionary (keys:tuple of variables, values:tequila objective) to define own gradient,
        None for automatic construction (default)
    initial_values: typing.Dict[typing.Hashable, numbers.Real], optional:
        Initial values as dictionary of Hashable types (variable keys) and floating point numbers. If given None they will all be set to zero
    variables: typing.List[typing.Hashable], optional:
         List of Variables to optimize
    samples: int, optional:
         samples/shots to take in every run of the quantum circuits (None activates full wavefunction simulation)
    maxiter: int : (Default value = 100):
         max iters to use.
    backend: str, optional:
         Simulator backend, will be automatically chosen if set to None
    backend_options: dict, optional:
         Additional options for the backend
         Will be unpacked and passed to the compiled objective in every call
    noise: NoiseModel, optional:
         a NoiseModel to apply to all expectation values in the objective.
    method: str : (Default = "BFGS"):
         Optimization method (see scipy documentation, or 'available methods')
    tol: float : (Default = 1.e-3):
         Convergence tolerance for optimization (see scipy documentation)
    method_options: dict, optional:
         Dictionary of options
         (see scipy documentation)
    method_bounds: typing.Dict[typing.Hashable, typing.Tuple[float, float]], optional:
        bounds for the variables (see scipy documentation)
    method_constraints: optional:
         (see scipy documentation
    silent: bool :
         No printout if True
    save_history: bool:
        Save the history throughout the optimization

    Returns
    -------
    SciPyReturnType:
        the results of optimization
    """
    if isinstance(gradient, dict) or hasattr(gradient, "items"):
        if all([isinstance(x, Objective) for x in gradient.values()]):
            gradient = format_variable_dictionary(gradient)
    if isinstance(hessian, dict) or hasattr(hessian, "items"):
        if all([isinstance(x, Objective) for x in hessian.values()]):
            hessian = {(assign_variable(k[0]), assign_variable([k[1]])): v for k, v in hessian.items()}
    method_bounds = format_variable_dictionary(method_bounds)

    # set defaults

    optimizer = OptimizerSciPy(save_history=save_history,
                               maxiter=maxiter,
                               method=method,
                               method_options=method_options,
                               method_bounds=method_bounds,
                               method_constraints=method_constraints,
                               silent=silent,
                               backend=backend,
                               backend_options=backend_options,
                               device=device,
                               samples=samples,
                               noise=noise,
                               tol=tol,
                               *args,
                               **kwargs)
    return optimizer(objective=objective,
                     gradient=gradient,
                     hessian=hessian,
                     initial_values=initial_values,
                     variables=variables, *args, **kwargs)
