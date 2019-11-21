import scipy, numpy, typing, numbers
from openvqe.objective import Objective
from .optimizer_base import Optimizer
from openvqe.circuit.gradient import grad
from ._scipy_containers import _EvalContainer, _GradContainer
from collections import namedtuple
import copy

SciPyReturnType = namedtuple('SciPyReturnType', 'energy angles history scipy_output')

class OptimizerSciPy(Optimizer):
    gradient_free_methods = ['Nelder-Mead', 'COBYLA', 'Powell', 'SLSQP']
    gradient_based_methods = ['L-BFGS-B', 'BFGS', 'CG', 'dogleg', 'TNC']

    @classmethod
    def available_methods(cls):
        """
        :return: All tested available methods
        """
        return cls.gradient_free_methods + cls.gradient_based_methods

    def __init__(self, method: str = "L-BFGS-B", tol: numbers.Real = None, method_options=None, method_bounds=None,
                 method_constraints=None, silent :bool = True, use_gradient: bool = None, **kwargs):
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
        self.method_bounds = method_bounds
        self.silent = silent
        if use_gradient is None:
            if method in self.gradient_based_methods:
                self.use_gradient = True
            elif method in self.gradient_free_methods:
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

        if method_constraints is None:
            self.method_constraints = ()
        else:
            self.method_constraints = method_constraints

    def __get_eval_function(self, simulator) -> _EvalContainer:
        if self.samples is None:
            return simulator.simulate_objective
        else:
            return lambda objective : simulator.measure_objective(objective=objective, samples=self.samples)

    def __call__(self, objective: Objective,
                 initial_values: typing.Dict[str, numbers.Number] = None,
                 reset_history: bool = True) -> SciPyReturnType:
        """
        Optimizes with scipi and gives back the optimized angles
        Get the optimized energies over the history
        :param objective: The openvqe Objective to minimize
        :param initial_values: initial values for the objective
        :param return_scipy_output: chose if the full scipy output shall be returned
        :param reset_history: reset the history before optimization starts (has no effect if self.save_history is False)
        :return: tuple of optimized energy ,optimized angles and scipy output
        """

        if self.save_history and reset_history:
            self.reset_history()

        # Need that for now to avoid compiler issues with gradients
        if self.use_gradient:
            copy_objective = copy.deepcopy(objective)

        simulator = self.initialize_simulator(self.samples)

        # Generate the function that evaluates <O>
        sim_eval = self.__get_eval_function(simulator=simulator)

        # Extract initial values
        angles = initial_values
        if angles is None:
            angles = objective.extract_parameters()

        # Transform the initial value directory into (ordered) arrays
        param_keys, param_values = zip(*angles.items())
        param_values = numpy.array(param_values)

        # Make E, grad E
        dE = None
        Es = []
        E = _EvalContainer(objective=objective,
                           param_keys=param_keys,
                           eval=sim_eval,
                           save_history=self.save_history,
                           silent=self.silent)
        if self.use_gradient:
            dO = grad(objective)
            dE = _GradContainer(objective=dO, param_keys=param_keys, eval=sim_eval,
                                save_history=self.save_history, silent=self.silent)

        bounds = None
        if self.method_bounds is not None:
            names, bounds = zip(*self.method_bounds.items())
            assert (names == param_keys)

        res = scipy.optimize.minimize(E, param_values, jac=dE,
                                      args=(Es,),
                                      method=self.method, tol=self.tol,
                                      bounds=bounds,
                                      constraints=self.method_constraints,
                                      options=self.method_options)

        if self.save_history:
            self.history.energies = E.history
            self.history.angles = E.history_angles
            if self.use_gradient:
                self.history.gradients = dE.history

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
             simulator: type = None,
             method: str = "L-BFGS-B",
             tol: float = None,
             method_options: dict = None,
             method_bounds: typing.Dict[str, numbers.Real] = None,
             method_constraints=None,
             save_history: bool = True,
             silent: bool = True) -> SciPyReturnType:
    """
    Call this if you don't like objects
    :param objective: The openvqe Objective to minimize
    :param initial_values: initial values for the objective
    :param samples: Number of samples to measure in each simulators run (None means full wavefunction simulation)
    :param maxiter: maximum number of iterations (can also be set over method_options)
    Note that some SciPy optimizers also accept 'maxfun' which is the maximum number of function evaluation
    You might consider massing down that keyword in the method_options dictionary
    :param simulator: The simulators you want to use (None -> automatically assigned)
    :param method: The scipy method passed as string
    :param tol: See scipy documentation for the method you picked
    :param method_options: See scipy documentation for the method you picked
    :param method_bounds: See scipy documentation for the method you picked
    Give in the same format as parameters/initial_values: Dict[str, float]
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
                               simulator=simulator,
                               tol=tol)

    return optimizer(objective=objective, initial_values=initial_values)



