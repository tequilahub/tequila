import scipy, numpy, typing, numbers
from tequila.objective import Objective
from tequila.objective.objective import assign_variable, Variable, format_variable_dictionary
from .optimizer_base import Optimizer
from ._containers import _EvalContainer, _GradContainer, _HessContainer, _QngContainer
from collections import namedtuple
from tequila.utils.exceptions import TequilaException
from tequila.circuit.noise import NoiseModel
from tequila.tools.qng import get_qng_combos


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
                 variables: typing.List[Variable] = None,
                 initial_values: typing.Dict[Variable, numbers.Real] = None,
                 gradient: typing.Dict[Variable, Objective] = None,
                 hessian: typing.Dict[typing.Tuple[Variable, Variable], Objective] = None,
                 reset_history: bool = True,
                 *args,
                 **kwargs) -> SciPyReturnType:
        """
        Optimizes with scipy and gives back the optimized angles
        Get the optimized energies over the history
        :param objective: The tequila Objective to minimize
        :param initial_values: initial values for the objective
        :param return_scipy_output: chose if the full scipy output shall be returned
        :param reset_history: reset the history before optimization starts (has no effect if self.save_history is False)
        :return: tuple of optimized energy ,optimized angles and scipy output
        """

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
        compiled_objective = self.compile_objective(objective=objective)
        for arg in compiled_objective.args:
            if hasattr(arg,'U'):
                if arg.U.device is not None:
                    # don't retrieve computer 100 times; pyquil errors out if this happens!
                    self.device = arg.U.device
                    break

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
                                        samples=self.samples, noise=self.noise,device=self.device)
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

        if isinstance(hessian, str):
            ddE = hessian
            compile_hessian = False

        if compile_gradient:
            grad_obj, comp_grad_obj = self.compile_gradient(objective=objective, variables=variables, gradient=gradient)
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
                                                           comp_grad_obj=comp_grad_obj)
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
            self.history.energy_evaluations = E.history
            self.history.angles = callback.angles
            self.history.angles_evaluations = E.history_angles
            self.history.gradients = callback.gradients
            self.history.hessians = callback.hessians
            if dE is not None and not isinstance(dE, str):
                self.history.gradients_evaluations = dE.history
            if ddE is not None and not isinstance(ddE, str):
                self.history.hessians_evaluations = ddE.history

            # some methods like "cobyla" do not support callback functions
            if len(self.history.energies) == 0:
                self.history.energies = E.history
                self.history.angles = E.history_angles



        E_final = res.fun
        angles_final = dict((param_keys[i], res.x[i]) for i in range(len(param_keys)))
        angles_final = {**angles_final, **passive_angles}

        return SciPyReturnType(energy=E_final, angles=format_variable_dictionary(angles_final), history=self.history,
                               scipy_output=res)


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
             initial_values: typing.Dict[typing.Hashable, numbers.Real] = None,
             variables: typing.List[typing.Hashable] = None,
             samples: int = None,
             maxiter: int = 100,
             backend: str = None,
             device: str = None,
             noise: NoiseModel = None,
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
        Other options include 'qng' to use the quantum natural gradient.
    hessian: typing.Union[str, typing.Dict[Variable, Objective], None] : (Default value = None) :
        '2-point', 'cs' or '3-point' for numerical gradient evaluation (does not work in combination with all optimizers),
        dictionary (keys:tuple of variables, values:tequila objective) to define own gradient,
        None for automatic construction (default)
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
    device: str:
        (Default value = None)
        the device from which to (potentially, simulatedly) sample all quantum circuits employed in optimization.
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
                               device=device,
                               method_options=method_options,
                               method_bounds=method_bounds,
                               method_constraints=method_constraints,
                               silent=silent,
                               backend=backend,
                               samples=samples,
                               noise_model=noise,
                               tol=tol,
                               *args,
                               **kwargs)
    if initial_values is not None:
        initial_values = {assign_variable(k): v for k, v in initial_values.items()}
    return optimizer(objective=objective,
                     gradient=gradient,
                     hessian=hessian,
                     initial_values=initial_values,
                     variables=variables, *args, **kwargs)
