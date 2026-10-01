# Generalized Adaptive Solvers
# as described in Kottmann, Anand, Aspuru-Guzik: https://doi.org/10.1039/D0SC06627C

from tequila import (
    QCircuit,
    QubitHamiltonian,
    gates,
    paulis,
    grad,
    simulate,
    TequilaWarning,
    TequilaException,
    minimize,
    ExpectationValue,
)
import numpy
import dataclasses
import warnings
from itertools import combinations

@dataclasses.dataclass
class AdaptParameters:
    optimizer_args: dict = dataclasses.field(
        default_factory=lambda: {"method": "bfgs", "silent": True, "method_options": {"gtol": 1.0e-5}}
    )
    compile_args: dict = dataclasses.field(default_factory=lambda: {})
    maxiter: int = 100
    batch_size = 1
    energy_convergence: float = None
    gradient_convergence: float = 1.0e-3
    max_gradient_convergence: float = 5.0e-4
    degeneracy_threshold: float = 5.0e-4
    silent: bool = False

    def __post_init__(self):
        # avoid stacking of same operator-types in a row
        if "method_options" in self.optimizer_args:
            if "gtol" in self.optimizer_args["method_options"]:
                gtol = self.optimizer_args["method_options"]["gtol"]
                if gtol > self.max_gradient_convergence:
                    warnings.warn(
                        "you specified screening threshold max_gradient_convergence={} but optimizer theshold gtol={}. This will lead to accumulation of the same operator, will set max_gradient_convergence={}".format(
                            self.max_gradient_convergence, gtol, gtol * 2
                        ),
                        TequilaWarning,
                    )
                    self.max_gradient_convergence = gtol * 2.0

    def __str__(self):
        info = ""
        for k, v in self.__dict__.items():
            info += "{:30} : {}\n".format(k, v)
        return info


class AdaptPoolBase:
    """
    Standard class for operator pools in Adapt
    The pool is a list of generators (tequila QubitHamiltonians)
    """

    generators: list = None

    __n: int = 0  # for iterator, don't touch

    def __init__(self, generators, trotter_steps=1):
        self.generators = generators
        self.trotter_steps = 1

    def make_unitary(self, k, label) -> QCircuit:
        return gates.Trotterized(generators=[self.generators[k]], angles=[(str(k), label)], steps=self.trotter_steps)

    def __iter__(self):
        self.__n = 0
        return self

    def __next__(self):
        if self.__n < len(self.generators):
            result = self.__n
            self.__n += 1
            return result
        else:
            raise StopIteration

    def __str__(self):
        return "{} with {} Generators".format(type(self).__name__, len(self.generators))


class ObjectiveFactoryBase:
    """
    Default class to create the objective in the Adapt solver
    This just creates the single ExpectationValue <H>_{Upre + U + Upost}
    and U will be the circuit that is adaptively constructed
    """

    Upre: QCircuit = QCircuit()
    Upost: QCircuit = QCircuit()
    H: QubitHamiltonian = None

    def __init__(self, H=None, Upre=None, Upost=None, *args, **kwargs):
        if H is None:
            raise TequilaException("No Hamiltonian was given to Adapt!")

        self.H = H
        if Upre is not None:
            self.Upre = Upre
        else:
            self.Upre = QCircuit()
        if Upost is not None:
            self.Upost = Upost
        else:
            self.Upost = QCircuit()

    def __call__(self, U, screening=False, *args, **kwargs):
        return ExpectationValue(H=self.H, U=self.Upre + U + self.Upost, *args, **kwargs)

    def grad_objective(self, *args, **kwargs):
        return self(*args, **kwargs)

    def __str__(self):
        return "{}".format(type(self).__name__)


class Adapt:
    operator_pool: AdaptPoolBase = None
    objective_factory = None
    parameters: AdaptParameters = AdaptParameters()

    def make_objective(self, U, variables=None, *args, **kwargs):
        return self.objective_factory(U=U, variables=variables, *args, **{**self.parameters.compile_args, **kwargs})

    def __init__(self, operator_pool, H=None, objective_factory=None, *args, **kwargs):
        """
        For the Default Adaptive Solver kwargs can contain Upre and Upost as described in:
        See out online tutorial for more information: https://github.com/tequilahub/tequila-tutorials
        Better code-documentation will be there at some point ....
        """
        self.operator_pool = operator_pool
        if objective_factory is None:
            self.objective_factory = ObjectiveFactoryBase(H, *args, **kwargs)
        else:
            self.objective_factory = objective_factory

        filtered = {k: v for k, v in kwargs.items() if k in self.parameters.__dict__}
        self.parameters = AdaptParameters(*args, **filtered)
        if (
            self.parameters.silent
            and self.parameters.optimizer_args is not None
            and "silent" not in self.parameters.optimizer_args
        ):
            self.parameters.optimizer_args["silent"] = True

    def __call__(self, static_variables=None, mp_pool=None, label=None, variables=None, *args, **kwargs):
        if not self.parameters.silent:
            print("Starting Adaptive Solver")
            print(self)

        # count resources
        screening_cycles = 0
        objective_expval_evaluations = 0
        gradient_expval_evaluations = 0
        histories = []

        if static_variables is None:
            static_variables = {}

        if variables is None:
            variables = {**static_variables}
        else:
            variables = {**variables, **static_variables}

        U = QCircuit()
        if "U" in kwargs:
            U = kwargs["U"]
        elif hasattr(self.operator_pool, "initialize_circuit"):
            U = self.operator_pool.initialize_circuit()

        initial_objective = self.make_objective(U, variables=variables)
        for k in initial_objective.extract_variables():
            if k not in variables:
                warnings.warn(
                    "variable {} of initial objective not given, setting to 0.0 and activate optimization".format(k),
                    TequilaWarning,
                )
                variables[k] = 0.0

        if len(initial_objective.extract_variables()) > 0:
            active_variables = [k for k in variables if k not in static_variables]
            if len(active_variables) > 0:
                if not self.parameters.silent:
                    print("initial optimization")
                margs = {"initial_values": variables}
                margs = {**margs, **self.parameters.compile_args, **self.parameters.optimizer_args}
                result = minimize(objective=initial_objective, variables=active_variables, **margs)

                variables = result.variables

        energy = simulate(initial_objective, variables=variables)
        for iter in range(self.parameters.maxiter):
            current_label = (iter, 0)
            if label is not None:
                current_label = (iter, label)

            gradients = self.screen_gradients(U=U, variables=variables, mp_pool=mp_pool)

            grad_values = numpy.asarray(list(gradients.values()))
            max_grad = max(grad_values)
            grad_norm = numpy.linalg.norm(grad_values)

            if grad_norm < self.parameters.gradient_convergence:
                if not self.parameters.silent:
                    print("pool gradient norm is {:+2.8f}, convergence criterion met".format(grad_norm))
                break
            if numpy.abs(max_grad) < self.parameters.max_gradient_convergence:
                if not self.parameters.silent:
                    print(
                        "max pool gradient is {:+2.8f}, convergence criterion |max(grad)|<{} met".format(
                            max_grad, self.parameters.max_gradient_convergence
                        )
                    )
                break

            batch_size = self.parameters.batch_size

            # detect degeneracies
            degeneracies = [
                k
                for k in range(batch_size, len(grad_values))
                if numpy.isclose(grad_values[batch_size - 1], grad_values[k], rtol=self.parameters.degeneracy_threshold)
            ]

            if len(degeneracies) > 0:
                batch_size += len(degeneracies)
                if not self.parameters.silent:
                    print(
                        "detected degeneracies: increasing batch size temporarily from {} to {}".format(
                            self.parameters.batch_size, batch_size
                        )
                    )

            count = 0

            op_names = []
            for k, v in gradients.items():
                Ux = self.operator_pool.make_unitary(k, label=current_label)
                U += Ux
                op_names.append(Ux.extract_variables())
                count += 1
                if count >= batch_size:
                    break

            variables = {**variables, **{k: 0.0 for k in U.extract_variables() if k not in variables}}
            active_variables = [k for k in variables if k not in static_variables]

            objective = self.make_objective(U, variables=variables)
            margs = {"initial_values": variables}
            margs = {**margs, **self.parameters.compile_args, **self.parameters.optimizer_args}
            result = minimize(objective=objective, variables=active_variables, **margs)

            niter = len(result.history.energies)
            diff = energy - result.energy
            energy = result.energy
            variables = result.variables

            if not self.parameters.silent:
                print("-------------------------------------")
                print("Finished iteration {}".format(iter))
                print("added ", op_names)
                print("current energy : {:+2.8f}".format(energy))
                print("difference     : {:+2.8f}".format(diff))
                print("grad_norm      : {:+2.8f}".format(grad_norm))
                print("max_grad       : {:+2.8f}".format(max_grad))
                print("ops in circuis : {}".format(len(U.gates)))
                print("optimizer      : {}".format(self.parameters.optimizer_args["method"]))
                print("opt-iterations : {}".format(niter))

            screening_cycles += 1
            mini_iter = len(result.history.extract_energies())
            gradient_expval = sum([v.count_expectationvalues() for k, v in grad(objective).items()])
            objective_expval_evaluations += mini_iter * objective.count_expectationvalues()
            gradient_expval_evaluations += mini_iter * gradient_expval
            histories.append(result.history)

            if self.parameters.energy_convergence is not None and numpy.abs(diff) < self.parameters.energy_convergence:
                if not self.parameters.silent:
                    print("energy difference is {:+2.8f}, convergence criterion met".format(diff))
                break

            if iter == self.parameters.maxiter - 1:
                if not self.parameters.silent:
                    print("reached maximum number of iterations")
                break

        @dataclasses.dataclass
        class AdaptReturn:
            U: QCircuit = None
            objective_factory: ObjectiveFactoryBase = None
            variables: dict = None
            energy: float = None
            histories: list = None
            screening_cycles: int = None
            objective_expval_evaluations: int = None
            gradient_expval_evaluations: int = None

        return AdaptReturn(
            U=U,
            variables=variables,
            objective_factory=self.objective_factory,
            energy=energy,
            histories=histories,
            screening_cycles=screening_cycles,
            objective_expval_evaluations=objective_expval_evaluations,
            gradient_expval_evaluations=gradient_expval_evaluations,
        )

    def screen_gradients(self, U, variables, mp_pool=None):
        args = []
        for k in self.operator_pool:
            arg = {}
            arg["k"] = k
            arg["variables"] = variables
            arg["U"] = U
            args.append(arg)

        if mp_pool is None:
            dEs = [self.do_screening(arg) for arg in args]
        else:
            if not self.parameters.silent:
                print("screen with {} workers".format(mp_pool._processes))
            dEs = mp_pool.map(self.do_screening, args)
        dEs = dict(sorted(dEs, reverse=True, key=lambda x: numpy.fabs(x[1])))
        return dEs

    def do_screening(self, arg):
        Ux = self.operator_pool.make_unitary(k=arg["k"], label="tmp")
        Utmp = arg["U"] + Ux
        variables = {**arg["variables"]}
        objective = self.make_objective(Utmp, screening=True, variables=variables)

        dEs = []
        for k in Ux.extract_variables():
            variables[k] = 0.0
            dEs.append(grad(objective, k))

        gradients = [
            numpy.abs(simulate(objective=dE, variables=variables, **self.parameters.compile_args)) for dE in dEs
        ]

        return arg["k"], sum(gradients)

    def __str__(self):
        result = str(self.parameters)
        result += str("{:30} : {}\n".format("operator pool: ", self.operator_pool))
        result += str("{:30} : {}\n".format("objective factory : ", self.objective_factory))
        return result


def MolecularPool(molecule, indices: str, *args, **kwargs):
    try:
        from sunrise.ADAPT.adapt import MolecularPool as _MolecularPool
    except ImportError:
        raise TequilaException("Project Sunrise not installed. Tequila Chemistry module has been outsourced there, keeping Tequila commands.\n Please install project-sunrise (pip install project-sunrise) to keep using these features.")
    return _MolecularPool(molecule, indices, *args, **kwargs)


def PseudoSingletMolecularPool(*args, **kwargs):
    try:
        from sunrise.ADAPT.adapt import PseudoSingletMolecularPool as _PseudoSingletMolecularPool
    except ImportError:
        raise TequilaException("Project Sunrise not installed. Tequila Chemistry module has been outsourced there, keeping Tequila commands.\n Please install project-sunrise (pip install project-sunrise) to keep using these features.")
    return _PseudoSingletMolecularPool(*args, **kwargs)


def ObjectiveFactorySequentialExcitedState(H, circuits: list, factors: list, *args, **kwargs):
    try:
        from sunrise.ADAPT.adapt import ObjectiveFactorySequentialExcitedState as _ObjectiveFactorySequentialExcitedState
    except ImportError:
        raise TequilaException("Project Sunrise not installed. Tequila Chemistry module has been outsourced there, keeping Tequila commands.\n Please install project-sunrise (pip install project-sunrise) to keep using these features.")
    return _ObjectiveFactorySequentialExcitedState(H, circuits, factors, *args, **kwargs)


def run_molecular_adapt(molecule, operator_pool: str = None, Upre=None, Upost=None, backend=None, *args, **kwargs):
    try:
        from sunrise.ADAPT.adapt import run_molecular_adapt as _run_molecular_adapt
    except ImportError:
        raise TequilaException("Project Sunrise not installed. Tequila Chemistry module has been outsourced there, keeping Tequila commands.\n Please install project-sunrise (pip install project-sunrise) to keep using these features.")
    return _run_molecular_adapt(molecule, operator_pool, Upre, Upost, backend, *args, **kwargs)
