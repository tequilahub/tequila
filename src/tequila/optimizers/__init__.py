from tequila.optimizers.optimizer_base import OptimizerHistory, Optimizer, TequilaOptimizerException, OptimizerResults
from tequila.optimizers.optimizer_scipy import OptimizerSciPy
from tequila.optimizers.optimizer_gd import OptimizerGD
from tequila.optimizers.optimizer_scipy import minimize as minimize_scipy
from tequila.optimizers.optimizer_gd import minimize as minimize_gd
from tequila.simulators.simulator_api import simulate
from dataclasses import dataclass

import typing, numbers, numpy


@dataclass
class _Optimizers:
    minimize: typing.Callable = None
    cls: type = None
    methods: list = None


SUPPORTED_OPTIMIZERS = ['scipy', 'phoenics', 'gpyopt', 'gd']
INSTALLED_OPTIMIZERS = {}
INSTALLED_OPTIMIZERS['scipy'] = _Optimizers(cls=OptimizerSciPy,
                                            minimize=minimize_scipy,
                                            methods=OptimizerSciPy.available_methods())
INSTALLED_OPTIMIZERS['gd'] = _Optimizers(cls=OptimizerGD,
                                         minimize=minimize_gd,
                                         methods=OptimizerGD.available_methods())

has_gpyopt = False
try:
    from tequila.optimizers.optimizer_gpyopt import OptimizerGPyOpt
    from tequila.optimizers.optimizer_gpyopt import minimize as minimize_gpyopt

    INSTALLED_OPTIMIZERS['gpyopt'] = _Optimizers(cls=OptimizerGPyOpt,
                                                 minimize=minimize_gpyopt,
                                                 methods=OptimizerGPyOpt.available_methods())
    has_gpyopt = True
except ImportError:
    has_gpyopt = False

has_phoenics = False
try:
    from tequila.optimizers.optimizer_phoenics import OptimizerPhoenics
    from tequila.optimizers.optimizer_phoenics import minimize as minimize_phoenics

    INSTALLED_OPTIMIZERS['phoenics'] = _Optimizers(cls=OptimizerPhoenics,
                                                   minimize=minimize_phoenics,
                                                   methods=OptimizerPhoenics.available_methods())
    has_phoenics = True
except ImportError:
    has_phoenics = False


def show_available_optimizers(module=None):
    """
    Returns
    -------
        A list of available optimization methods
        The list depends on optimization packages installed in your system
    """
    if module is None:
        print("available methods for optimizer modules found on your system:")
    else:
        print("available methods for optimizer module {}".format(module))
        if module not in INSTALLED_OPTIMIZERS:
            print("module {} not found!".format(module))
            module = None

    print("{:20} | {}".format("method", "optimizer module"))
    print("--------------------------")
    for k, v in INSTALLED_OPTIMIZERS.items():
        if module is not None and module != k:
            continue
        for method in v.methods:
            print("{:20} | {}".format(method, k))

    if module is None:
        print("Supported optimizer modules: ", SUPPORTED_OPTIMIZERS)
        print("Installed optimizer modules: ", list(INSTALLED_OPTIMIZERS.keys()))


def minimize(objective,
             method: str = "bfgs",
             variables: list = None,
             initial_values: typing.Union[dict, numbers.Number, typing.Callable] = 0.0,
             maxiter: int = None,
             *args,
             **kwargs):
    """

    Parameters
    ----------
    method: str:
       The optimization method (e.g. bfgs, cobyla, nelder-mead, ...)
       see 'tq.optimizers.show_available_methods()' for an overview
    objective: tq.Objective:
       The abstract tequila objective to be optimized
    variables: list of names:
       The variables which shall be optimized given as list
       Can be passed as list of names or list of tq variables
    initial_values: dict:
       Initial values for the optimization, passed as dictionary
       with the variable names as keys.
       Alternatively `zero`, `random` or a single number are accepted
    maxiter:
       maximum number of iterations
    kwargs:
       further keyword arguments for the actual minimization functions
       can also be called directly as tq.minimize_modulename
       e.g. tq.minimize_scipy
       See their documentation for more details

       example: gradient keyword:
       gradient (Default Value: None):
       instructions for gradient compilation
       can be a dictionary of tequila objectives representing the gradients
       or a string/dictionary giving instructions for numerical gradients
       examples are
            gradient = '2-point'
            gradient = {'method':'2-point', 'stepsize': 1.e-4}
            gradient = {'method':Callable, 'stepsize': 1.e-4}
            see optimizer_base.py for method examples

        gradient = None: analytical gradients are compiled


    Returns
    -------

    """

    ovtmp=objective.extract_variables()
    fast_return=False
    if ovtmp is None or len(ovtmp) == 0:
        return OptimizerResults(energy=float(simulate(objective, *args, **kwargs)), variables={}, history=OptimizerHistory())

    for k, v in INSTALLED_OPTIMIZERS.items():
        if method.lower() in v.methods or method.upper() in v.methods:
            return v.minimize(
                objective=objective,
                method=method,
                variables=variables,
                initial_values=initial_values,
                maxiter=maxiter,
                *args, **kwargs)

    raise TequilaOptimizerException(
        "Could not find optimization method {} in tequila optimizers. You might miss dependencies")
