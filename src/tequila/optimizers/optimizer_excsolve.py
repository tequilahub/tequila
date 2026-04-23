import scipy
import numpy
import typing
import numbers
from excitationsolve import ExcitationSolveScipy
from tequila.objective import Objective
from tequila.objective.objective import assign_variable, Variable, format_variable_dictionary, format_variable_list
from .optimizer_base import Optimizer, OptimizerResults
from ._containers import _EvalContainer, _GradContainer, _HessContainer, _QngContainer
from .optimizer_scipy import SciPyResults
from tequila.utils.exceptions import TequilaException
from tequila.circuit.noise import NoiseModel
from tequila.tools.qng import get_qng_combos

from dataclasses import dataclass


class OptimizerExcitationSolve(Optimizer):
    r"""The ExcitationSolve optimizer as a SciPy optimizer that can be given to the scipy.optimize.minimize function.

    Usage:
    ```python
        excsolve_obj = ExcitationSolveScipy(maxiter=100, tol=1e-10, save_parameters=True)
        optimizer = excsolve_obj.minimize
        res = scipy.optimize.minimize(cost, params, method=optimizer)
        energies = excsolve_obj.energies
        counts = excsolve_obj.nfevs
    ```

    Note that this optimizer never needs to evaluate the ansatz circuit
    at the (current) optimal parameters, unless the optimal parameters fall onto
    the sample points used to reconstruct the energy function.
    Therefore, when used with a qiskit VQE object, the energies transmitted
    to a VQE callback function, do not seem to improve or converge. Nevertheless,
    the determined optimal energy and parameters are still returned.

    Args:
        maxiter (int): Maximum number of VQE iterations (maximum number of times to optimize all parameters)
        tol: Threshold of energy difference after subsequent VQE iterations defining convergence
        num_samples (int, optional): Number of different parameter values at which to sample
            the energy to reconstruct the energy function in one parameter.
            Must be greater or equal to 5. Defaults to 5.
        hf_energy (float | None, optional): The Hartree-Fock energy, i.e. the energy of the
            system where all parameters in the circuit are zero. If none, this will be
            calculated by evaluating the energy of the ansatz with all parameters set to zero.
            If this energy is known from a prior classical calculation, e.g. a Hartree-Fock
            calculation, one energy evaluation is saved. Defaults to None.
        save_parameters (bool, optional): If True, params member variable contains
            all optimal parameter values after each optimization step,
            i.e. after optimizing each single parameter. Defaults to False.
        param_scaling (float, optional): Factor used for rescaling the parameters. This ExcitationSolve optimizer
                                            expects the parameters to be 2\pi periodic. For example, in Qiskit
                                            the excitation parameters result in excitation operators being \pi periodic.
                                            Therefore, we use a factor of 0.5 for qiskit, resulting in a Period of 2\pi.
    """

    @classmethod
    def available_methods(cls):
        """:return: All tested available methods"""
        return ["excitationsolve"]

    def __init__(
        self, maxiter, tol=1e-12, num_samples=5, hf_energy=None, save_parameters=False, param_scaling=0.5, **kwargs
    ):
        if maxiter is None:
            maxiter = 10

        super().__init__(**kwargs)

        self.opt = ExcitationSolveScipy(
            maxiter=maxiter,
            tol=tol,
            num_samples=num_samples,
            hf_energy=hf_energy,
            save_parameters=save_parameters,
            param_scaling=param_scaling,
        )

    def __call__(
        self,
        objective: Objective,
        variables: typing.List[Variable],
        initial_values: typing.Dict[Variable, numbers.Real] = None,
        *args,
        **kwargs,
    ) -> SciPyResults:
        objective = objective.contract()
        infostring = "{:15} : {}\n".format("Method", "ExcitationSolve")
        infostring += "{:15} : {} expectationvalues\n".format("Objective", objective.count_expectationvalues())

        # if self.save_history and reset_history:
        #     self.reset_history()

        active_angles, passive_angles, variables = self.initialize_variables(objective, initial_values, variables)

        # Transform the initial value directory into (ordered) arrays
        param_keys, param_values = zip(*active_angles.items())
        param_values = numpy.array(param_values)

        # do the compilation here to avoid costly recompilation during the optimization
        compiled_objective = self.compile_objective(objective=objective, *args, **kwargs)
        E = _EvalContainer(
            objective=compiled_objective,
            param_keys=param_keys,
            samples=self.samples,
            passive_angles=passive_angles,
            save_history=self.save_history,
            print_level=self.print_level,
        )

        res = self.opt.minimize(E, param_values, *args, **kwargs)

        if self.save_history:
            self.history.energies = self.opt.energies
            self.history.angles = self.opt.params
            # self.history.gradients = self.opt.energies_shiftste

        return SciPyResults(energy=res.fun, history=self.history, variables=res.x, scipy_result=res)


def minimize(
    objective: Objective,
    variables: typing.List[Variable],
    initial_values: typing.Dict[Variable, numbers.Real] = None,
    method: str = "excitationsolve",
    maxiter: int = 10,
    *args,
    **kwargs,
):
    optimize = OptimizerExcitationSolve(
        maxiter=maxiter,
        save_parameters=True,
        *args,
        **kwargs,
    )
    return optimize(
        objective=objective,
        variables=variables,
        initial_values=initial_values,
        *args,
        **kwargs,
    )
