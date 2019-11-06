from openvqe import OpenVQEException
from openvqe import typing
from openvqe.circuit.variable import Variable
from openvqe.objective import Objective
from openvqe.simulator import pick_simulator
from openvqe.circuit.gradient import grad
from openvqe import scipy
from openvqe import numpy as np

def minimize(objective: Objective,
             initial_values=None,
             use_gradient=True,
             simulator=None,samples=None,
             method="BFGS", tol=1e-3,
             method_options=None,
             method_bounds=None,
             method_constraints=(),
             return_all=True):
    """Optimize a circuit to minimize a given objective using scipy."""

    # Start the simulator
    if simulator is None:
        sim = pick_simulator(samples=samples)
    else:
        sim = simulator
    if isinstance(sim, type):
        sim = sim()             # Make sure the simulator is initialized

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
    E = make_energy_function(objective, sim_eval, params)
    if use_gradient:
        dE = make_energy_gradient(objective, sim_eval, params)
    else:
        dE = None

    res= scipy.optimize.minimize(E, x0, jac=dE,
                                 args=(Es,),
                                 method=method, tol=tol,
                                 bounds=method_bounds,
                                 constraints=method_constraints,
                                 options=method_options)
    
    # Format output
    res.parameters = params[:]  # add the ordered parameter list to res
    res.Ovals = Es[:]           # Add values of O
    angles_final = dict((params[i], res.x[i]) for i in range(len(params)))

    if return_all:
        return res.fun, angles_final, res
    else:
        return res.fun, angles_final


def make_energy_function(objective: Objective, simulator_eval, params):
    """
    Generates the scalar function E(theta_1, theta_2 ...).
    """
    N = len(params)

    def E(p, cache):
        objective.update_parameters(dict((params[i], p[i])
                                    for i in range(N)))
        E=simulator_eval(objective)
        cache += [E]
        return E

    return E

def make_energy_gradient(objective: Objective, simulator_eval, params):
    """
    Generates the vector function \nabla E(theta_1, theta_2 ...).
    """
    N = len(params)
    dO = grad(objective, params)

    def dE(p, cache=None):
        dE_vec = np.zeros(N)
        for i in range(N):
            dO[params[i]].update_parameters(dict((params[i], p[i])
                                                    for i in range(len(params))))
            dE_vec[i] = simulator_eval(dO[params[i]])
        return dE_vec

    return dE
