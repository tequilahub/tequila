from openvqe import typing, numbers, copy
from openvqe.objective import Objective
from openvqe.simulator import pick_simulator
from openvqe.circuit.gradient import grad
from openvqe import scipy
from openvqe import numpy as np
from .optimizer_base import Optimizer


class EvalContainer:

    def __init__(self, objective, params, eval):
        self.objective = objective
        self.params = params
        self.eval = eval
        self.N = len(params)
        self.memory = []
        self.memory_angles = []

    def E_function(self, p, cache=None):
        angles = dict((self.params[i], p[i]) for i in range(self.N))
        self.objective.update_parameters(angles)
        E = self.eval(self.objective)
        self.memory.append(E)
        self.memory_angles.append(angles)
        cache += [E]
        return E

    def __call__(self, angle, cache=None):
        return self.E_function(p=angle, cache=cache)

class GradContainer:

    def __init__(self, objective, params, eval):
        self.objective = objective
        self.params = params
        self.N = len(params)
        self.eval = eval
        self.memory = []

    def dE_function(self, p, cache=None):
        dO = grad(self.objective, self.params)
        dE_vec = np.zeros(self.N)
        memory = dict()
        for i in range(self.N):
            dO[self.params[i]].update_parameters(dict((self.params[i], p[i]) for i in range(len(self.params))))
            dE_vec[i] = self.eval(dO[self.params[i]])
            memory[self.params[i]] = dE_vec[i]
        self.memory.append(memory)
        return dE_vec

    def __call__(self, angle, cache=None):
        return self.dE_function(p=angle, cache=cache)





class OptimizerSciPy(Optimizer):

    def __init__(self, method: str = "BFGS", tol: numbers.Real = 1.e-3, method_options=None, method_bounds=None,
                 method_constraints=None, use_gradient: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.method = method.upper()
        self.tol = tol
        self.method_options = method_options
        self.method_bounds = method_bounds
        self.use_gradient = use_gradient
        if method_constraints is None:
            self.method_constraints = ()
        else:
            self.method_constraints = method_constraints

    def __get_eval_function(self, simulator) -> EvalContainer:
        if self.samples is None:
            return simulator.simulate_objective
        else:
            return simulator.measure_objective

    def __call__(self, objective: Objective, initial_values: typing.Dict[str, numbers.Number] = None) -> typing.Dict[
        str, numbers.Number]:
        simulator = self.initialize_simulator(self.samples)

        # Generate the function that evaluates <O>
        sim_eval = self.__get_eval_function(simulator=simulator)

        # Extract initial values
        angles = initial_values
        if angles is None:
            angles = objective.extract_parameters()

        # Transform the initial value directory into (ordered) arrays
        params, x0 = zip(*angles.items())
        x0 = np.array(x0)

        # Make E, grad E
        dE = None
        Es = []
        E = EvalContainer(objective=objective, params=params, eval=sim_eval)
        if self.use_gradient:
            dE = GradContainer(objective=objective, params=params, eval=sim_eval)

        res = scipy.optimize.minimize(E, x0, jac=dE,
                                      args=(Es,),
                                      method=self.method, tol=self.tol,
                                      bounds=self.method_bounds,
                                      constraints=self.method_constraints,
                                      options=self.method_options)

        if self.save_history:
            self.history.energies = E.memory
            self.history.angles = E.memory_angles
            if self.use_gradient:
                self.history.gradients = dE.memory
            # self.history['angles'].append()

        angles_final = dict((params[i], res.x[i]) for i in range(len(params)))
        return angles_final


def minimize(objective: Objective,
             initial_values: typing.Dict[str, numbers.Number] = None,
             use_gradient: bool = True,
             simulator=None, samples: int = None,
             method: str = "BFGS", tol: numbers.Real = 1e-3,
             method_options=None,
             method_bounds=None,
             method_constraints=(),
             return_all: bool = True) -> typing.Tuple[
    numbers.Number, typing.Dict[str, numbers.Number], typing.Union[scipy.optimize.OptimizeResult, None]]:
    """
    Optimize a circuit to minimize a given objective using scipy
    :param objective: The openvqe Objective to minimize
    :param initial_values: initial values for the objective
    :param use_gradient: do gradient based optimization
    :param simulator: choose a specific simulator as backend, if None it will be automatically picked
    :param samples: number of individual measurment samples for each step in the optimization. None means full wavefunction simulation
    :param method: The scipy method passed as string
    :param tol: See scipy documentation for the method you picked
    :param method_options: See scipy documentation for the method you picked
    :param method_bounds: See scipy documentation for the method you picked
    :param method_constraints: See scipy documentation for the method you picked
    :param return_all: return the whole scipy OptimizeResult object
    :return: tuple(final value of the Objective, optimized parameters, scipy OptimizeResult object or None)
    """

    # Start the simulator
    if simulator is None:
        sim = pick_simulator(samples=samples)
    else:
        sim = simulator
    if isinstance(sim, type):
        sim = sim()  # Make sure the simulator is initialized

    # Generate the function that evaluates <O>
    if samples is None:
        sim_eval = lambda O: sim.simulate_objective(objective=O)
    else:
        sim_eval = lambda O: sim.measure_objective(objective=O, samples=samples)

    # Extract initial values
    angles = initial_values
    if angles is None:
        angles = objective.extract_parameters()

    # Transform the initial value directory into (ordered) arrays
    params, x0 = zip(*angles.items())
    x0 = np.array(x0)

    # Make E, grad E
    Es = []
    E = wrap_energy_function(objective, sim_eval, params)
    if use_gradient:
        dE = wrap_energy_gradient_function(objective, sim_eval, params)
    else:
        dE = None

    if minimize:
        res = scipy.optimize.minimize(E, x0, jac=dE,
                                      args=(Es,),
                                      method=method, tol=tol,
                                      bounds=method_bounds,
                                      constraints=method_constraints,
                                      options=method_options)

    # Format output
    res.parameters = params[:]  # add the ordered parameter list to res
    res.evals = Es[:]  # function evaluations
    angles_final = dict((params[i], res.x[i]) for i in range(len(params)))

    if return_all:
        return res.fun, angles_final, res
    else:
        return res.fun, angles_final


def wrap_energy_function(objective: Objective, simulator_eval, params) -> typing.Callable:
    """
    Helper Function for minimize/maximize
    Generates the scalar function E(theta_1, theta_2 ...).
    """
    N = len(params)

    def E_function(p, cache=None):
        objective.update_parameters(dict((params[i], p[i])
                                         for i in range(N)))
        E = simulator_eval(objective)
        cache += [E]
        return E

    return E_function


def wrap_energy_gradient_function(objective: Objective, simulator_eval, params) -> typing.Callable:
    """
    Helper Function for minimize/maximize
    Generates the vector function \nabla E(theta_1, theta_2 ...).
    """

    N = len(params)
    dO = grad(objective, params)

    def dE_function(p, cache=None):
        dE_vec = np.zeros(N)
        for i in range(N):
            dO[params[i]].update_parameters(dict((params[i], p[i])
                                                 for i in range(len(params))))
            dE_vec[i] = simulator_eval(dO[params[i]])
        return dE_vec

    return dE_function
