from collections import namedtuple
import typing
from numbers import Real as RealNumber
from typing import Dict, Union, Hashable

from tequila.objective import Objective, Variable, assign_variable, format_variable_dictionary
from tequila.utils.exceptions import TequilaException
from tequila.utils import to_float
from tequila.simulators.simulator_base import BackendCircuit, BackendExpectationValue

SUPPORTED_BACKENDS = ["qulacs", "qiskit", "cirq", "pyquil", "symbolic"]
SUPPORTED_NOISE_BACKENDS = ["qiskit",'pyquil']
BackendTypes = namedtuple('BackendTypes', 'CircType ExpValueType')
INSTALLED_SIMULATORS = {}
INSTALLED_SAMPLERS = {}
HAS_QULACS = True
INSTALLED_NOISE_SAMPLERS = {}
if typing.TYPE_CHECKING:
    from tequila.objective import Objective, Variable
    from tequila.circuit.gates import QCircuit
    import numbers.Real as RealNumber
    from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction

"""
Check which simulators are installed
We are distinguishing two classes of simulators: Samplers and full wavefunction simuators
"""

try:
    import qulacs
    from tequila.simulators.simulator_qulacs import BackendCircuitQulacs, BackendExpectationValueQulacs

    HAS_QULACS = True
    INSTALLED_SIMULATORS["qulacs"] = BackendTypes(CircType=BackendCircuitQulacs,
                                                  ExpValueType=BackendExpectationValueQulacs)

except ImportError:
    HAS_QULACS = False

HAS_PYQUIL = True
from shutil import which

HAS_QVM = which("qvm") is not None
if HAS_QVM:
    try:
        from tequila.simulators.simulator_pyquil import BackendCircuitPyquil, BackendExpectationValuePyquil

        HAS_PYQUIL = True
        INSTALLED_SIMULATORS["pyquil"] = BackendTypes(BackendCircuitPyquil, BackendExpectationValuePyquil)
        INSTALLED_SAMPLERS["pyquil"] = BackendTypes(BackendCircuitPyquil, BackendExpectationValuePyquil)
        INSTALLED_NOISE_SAMPLERS["pyquil"] = BackendTypes(BackendCircuitPyquil, BackendExpectationValuePyquil)
    except ImportError:
        HAS_PYQUIL = False
else:
    HAS_PYQUIL = False

HAS_QISKIT = True
try:
    from tequila.simulators.simulator_qiskit import BackendCircuitQiskit, BackendExpectationValueQiskit

    HAS_QISKIT = True
    INSTALLED_SIMULATORS["qiskit"] = BackendTypes(BackendCircuitQiskit, BackendExpectationValueQiskit)
    INSTALLED_SAMPLERS["qiskit"] = BackendTypes(BackendCircuitQiskit, BackendExpectationValueQiskit)
    INSTALLED_NOISE_SAMPLERS["qiskit"] = BackendTypes(BackendCircuitQiskit, BackendExpectationValueQiskit)
except ImportError:
    HAS_QISKIT = False

HAS_CIRQ = True
try:
    from tequila.simulators.simulator_cirq import BackendCircuitCirq, BackendExpectationValueCirq
    HAS_CIRQ = True
    INSTALLED_SIMULATORS["cirq"] = BackendTypes(CircType=BackendCircuitCirq, ExpValueType=BackendExpectationValueCirq)
    INSTALLED_SAMPLERS["cirq"] = BackendTypes(CircType=BackendCircuitCirq, ExpValueType=BackendExpectationValueCirq)
    INSTALLED_NOISE_SAMPLERS["cirq"] = BackendTypes(CircType=BackendCircuitCirq, ExpValueType=BackendExpectationValueCirq)
except ImportError:
    HAS_CIRQ = False

from tequila.simulators.simulator_symbolic import BackendCircuitSymbolic, BackendExpectationValueSymbolic

INSTALLED_SIMULATORS["symbolic"] = BackendTypes(CircType=BackendCircuitSymbolic,
                                                ExpValueType=BackendExpectationValueSymbolic)
HAS_SYMBOLIC = True


def show_available_simulators():
    """ """
    print("Supported Backends:\n")
    for k in SUPPORTED_BACKENDS:
        print(k)
    print("Installed Wavefunction Simulators:\n")
    for k in INSTALLED_SIMULATORS.keys():
        print(k)
    print("\nInstalled Wavefunction Samplers:\n")
    for k in INSTALLED_SAMPLERS.keys():
        print(k)


def pick_backend(backend: str = None, samples: int = None, noise: bool=False, exclude_symbolic: bool = True) -> str:
    """
    verifies if the backend is installed and picks one automatically if set to None
    :param backend: the demanded backend
    :param samples: if not None the simulator needs to be able to sample wavefunctions
    :param noise: if true,
    :param exclude_symbolic: only for random choice
    :return: An installed backend as string
    """

    if len(INSTALLED_SIMULATORS) == 0:
        raise TequilaException("No simulators installed on your system")

    if backend is None:
        if noise is False:
            for f in SUPPORTED_BACKENDS:
                if samples is None:
                    if f in INSTALLED_SIMULATORS:
                        return f
                else:
                    if f in INSTALLED_SAMPLERS:
                        return f
        else:
            for f in SUPPORTED_NOISE_BACKENDS:
                if samples is None:
                    raise TequilaException("Noise requires sampling; please provide a positive, integer value for samples")
                else:
                    if f in INSTALLED_NOISE_SAMPLERS:
                        return f
    if hasattr(backend, "lower"):
        backend = backend.lower()

    if backend == "random":
        from numpy import random as random
        import time
        state = random.RandomState(int(str(time.clock()).split('.')[-1]) % 2 ** 32)
        if samples is None:
            backend = state.choice(list(INSTALLED_SIMULATORS.keys()), 1)[0]
        else:
            backend = state.choice(list(INSTALLED_SAMPLERS.keys()), 1)[0]

        if exclude_symbolic:
            while (backend == "symbolic"):
                backend = state.choice(list(INSTALLED_SIMULATORS.keys()), 1)[0]
        return backend

    if backend not in SUPPORTED_BACKENDS:
        raise TequilaException("Backend {backend} not supported ".format(backend=backend))

    if noise is False and samples is None and backend not in INSTALLED_SIMULATORS:
        raise TequilaException("Backend {backend} not installed ".format(backend=backend))
    elif noise is False and samples is not None and backend not in INSTALLED_SAMPLERS:
        raise TequilaException("Backend {backend} not installed ".format(backend=backend))
    elif noise is not False and samples is not None and backend not in INSTALLED_NOISE_SAMPLERS:
        raise TequilaException("Backend {backend} not installed or else Noise has not been implemented".format(backend=backend))

    return backend


def compile_objective(objective: 'Objective',
                      variables: typing.Dict['Variable', 'RealNumber'],
                      backend: str = None,
                      samples: int = None,
                      noise_model=None,
                      *args,
                      **kwargs):
    """
    Compiles an objective to a chosen backend
    The abstract circuits are replaced by the circuit objects of the backend
    Direct return if the objective was alrady compiled
    :param objective: abstract objective
    :param variables: The variables of the objective given as dictionary
    with keys as tequila Variables and values the corresponding real numbers
    :param backend: specify the backend or give None for automatic assignment
    :param noise_model: the NoiseModel to apply to the objective.
    :return: Compiled Objective
    """

    backend = pick_backend(backend=backend, samples=samples,noise=noise_model is not None)

    ExpValueType = INSTALLED_SIMULATORS[pick_backend(backend=backend)].ExpValueType
    if hasattr(objective, "simulate"):
        for arg in objective.args:
            if hasattr(arg, "U") and not isinstance(arg, ExpValueType):
                raise TequilaException(
                    "Looks like the objective was already compiled for another backend. You gave {} and tried to compile to {}".format(
                        type(objective), ExpValueType))
        return objective

    compiled_args = []
    for arg in objective.args:
        if hasattr(arg, "H") and hasattr(arg, "U") and not isinstance(arg, ExpValueType):
            compiled_args.append(ExpValueType(arg, variables,noise_model))
        else:
            compiled_args.append(arg)
    return type(objective)(args=compiled_args, transformation=objective._transformation)


def simulate_objective(objective: 'Objective',
                       variables: typing.Dict['Variable', 'RealNumber'],
                       backend: str = None,
                       *args,
                       **kwargs):
    """
    Simulate a tequila Objective
    The Objective will be compiled and then simulated
    :param objective: abstract or compiled objective
    :param variables: The variables of the objective given as dictionary
    :param backend: specify the backend or give None for automatic assignment
    :return: The evaluated objective
    """
    compiled = compile_objective(objective, variables, backend)

    E = []
    for Ei in compiled.args:
        if hasattr(Ei, "simulate"):
            E.append(Ei.simulate(variables=variables,*args, **kwargs))
        else:
            E.append(Ei(variables=variables))
    # return evaluated result
    return to_float(objective.transformation(*E))


def sample_objective(objective: 'Objective',
                     variables: typing.Dict['Variable', 'RealNumber'],
                     samples: int,
                     backend: str = None,
                     noise_model = None,
                     *args,
                     **kwargs):
    """
    Sample a tequila Objective
    The Objective will be compiled and then simulated
    :param objective: abstract or compiled objective
    :param variables: The variables of the objective given as dictionary
    :param samples: The number of samples/measurements to take for each expectationvalue
    :param backend: specify the backend or give None for automatic assignment
    :param noise_model: a noise model to use in simulation.
    :return: The sampled objective
    """

    backend = pick_backend(backend=backend, samples=samples,noise=noise_model is not None)
    compiled = compile_objective(objective, variables, backend,noise_model=noise_model)

    # break the objective apart into its individual pauli components in every expectationvalue
    # then sample all of those
    evaluated = []
    for arg in compiled.args:
        if hasattr(arg, "H"):
            E = 0.0
            for ps in arg.H.paulistrings:
                E += arg.sample_paulistring(variables=variables, samples=samples, paulistring=ps, *args, **kwargs)
            evaluated.append(E)
        else:
            evaluated.append(arg(variables))

    return to_float(compiled.transformation(*evaluated))


def compile_circuit(abstract_circuit: 'QCircuit',
                    variables: typing.Dict['Variable', 'RealNumber'],
                    backend: str = None,
                    noise_model= None,
                    *args,
                    **kwargs) -> BackendCircuit:
    """
    Compile an abstract tequila circuit into a circuit corresponding to a supported backend
    direct return if the abstract circuit was already compiled
    :param abstract_circuit: The abstract tequila circuit
    :param variables: The variables of the objective given as dictionary
    with keys as tequila Variables and values the corresponding real numbers
    :param backend: specify the backend or give None for automatic assignment
    :param noise_model: specify a NoiseModel object to convert to the backend's noise
    :return: The compiled circuit object
    """

    CircType = INSTALLED_SIMULATORS[pick_backend(backend=backend,noise=noise_model is not None)].CircType

    if hasattr(abstract_circuit, "simulate"):
        if not isinstance(abstract_circuit, CircType):
            raise TequilaException(
                "Looks like the circuit was already compiled for another backend. You gave {} and tried to compile to {}".format(
                    type(abstract_circuit), CircType))
        else:
            return abstract_circuit

    return CircType(abstract_circuit=abstract_circuit, variables=variables,noise_model=noise_model)


def simulate_wavefunction(abstract_circuit: 'QCircuit',
                          variables: typing.Dict['Variable', 'RealNumber'],
                          backend: str = None,
                          *args,
                          **kwargs):
    """
    Simulate an abstract tequila circuit
    :param abstract_circuit: abstract or compiled circuit
    :param variables: The variables of the objective given as dictionary
    with keys as tequila Variables and values the corresponding real numbers
    :param backend: specify the backend or give None for automatic assignment
    :return:
    """

    compiled = compile_circuit(abstract_circuit=abstract_circuit, variables=variables, backend=backend, *args, **kwargs)
    return compiled.simulate(variables=variables, *args, **kwargs)


def sample_wavefunction(abstract_circuit: 'QCircuit',
                        variables: typing.Dict['Variable', 'RealNumber'],
                        samples: int,
                        backend: str = None,
                        noise_model = None,
                        *args, **kwargs):
    """
    Sample an abstract tequila circuit
    :param abstract_circuit: abstract or compiled circuit
    :param variables: The variables of the objective given as dictionary
    with keys as tequila Variables and values the corresponding real numbers
    :param samples: Number of samples/measurements
    :param backend: specify the backend or give None for automatic assignment
    :param noise_model: NoiseModel to apply to simulation.
    :return:
    """

    backend = pick_backend(backend, samples=samples,noise=noise_model is not None)
    compiled = compile_circuit(abstract_circuit=abstract_circuit, variables=variables, backend=backend,
                               noise_model=noise_model, *args, **kwargs)
    return compiled.sample(variables=variables, samples=samples, *args, **kwargs)


def simulate(objective: typing.Union['Objective', 'QCircuit'],
             variables: Dict[Union[Variable, Hashable], RealNumber] = None,
             samples: int = None,
             backend: str = None,
             noise_model=None,
             *args,
             **kwargs) -> Union[RealNumber, 'QubitWaveFunction']:
    """Simulate a tequila objective or circuit

    Parameters
    ----------
    objective :
        tequila objective or circuit
    variables :
        The variables of the objective given as dictionary
        with keys as tequila Variables/hashable types and values the corresponding real numbers
    samples : int : (Default value = None)
        if None a full wavefunction simulation is performed, otherwise a fixed number of samples is simulated
    backend : str : (Default value = None)
        specify the backend or give None for automatic assignment
    noise_model: NoiseModel :
        specify a noise model to apply to simulation/sampling

    *args :

    **kwargs :


    Returns
    -------
    type
        simulated/sampled objective or simulated/sampled wavefunction

    """
    if variables is None and not (len(objective.extract_variables()) == 0):
        raise TequilaException(
            "You called simulate for a parametrized type but forgot to pass down the variables: {}".format(
                objective.extract_variables()))
    elif variables is not None:
        # allow hashable types as keys without casting it to variables
        variables = {assign_variable(k): v for k, v in variables.items()}

    if isinstance(objective, Objective) or hasattr(objective, "args"):
        if samples is None:
            return simulate_objective(objective=objective, variables=variables, backend=backend)
        else:
            return sample_objective(objective=objective, variables=variables, samples=samples, backend=backend,
                                    noise_model=noise_model)
    elif hasattr(objective, "gates"):
        if samples is None:
            return simulate_wavefunction(abstract_circuit=objective, variables=variables, backend=backend, *args,
                                         **kwargs)
        else:
            return sample_wavefunction(abstract_circuit=objective, variables=variables, samples=samples,
                                       backend=backend,noise_model=noise_model, *args, **kwargs)
    else:
        raise TequilaException(
            "Don't know how to simulate object of type: {type}, \n{object}".format(type=type(objective),
                                                                                   object=objective))


def draw(objective, variables=None, backend: str = None):
    """

    Pretty output (depends on installed backends)

    Parameters
    ----------
    objective :
        the tequila objective to print out
    variables :
         (Default value = None)
         Give variables if the objective is parametrized
    backend:str :
         (Default value = None)
         chose backend (of None it will be automatically picked)
    """
    if backend is None:
        if "cirq" in INSTALLED_SIMULATORS:
            backend = "cirq"
        elif "qiskit" in INSTALLED_SIMULATORS:
            backend = "qiskit"

    if isinstance(objective, Objective):
        # pretty printer not here yet
        print(objective)
    else:
        if backend is None:
            print(objective)
        else:
            if variables is None:
                variables = {}
            for k in objective.extract_variables():
                if k not in variables:
                    variables[k] = 0.0
            variables = format_variable_dictionary(variables)
            compiled = compile_circuit(abstract_circuit=objective, backend=backend,
                                                                        variables=variables)
            print(compiled.circuit)


def compile(objective: typing.Union['Objective', 'QCircuit'],
            variables: Dict[Union['Variable', Hashable], RealNumber] = None,
            samples: int = None,
            backend: str = None,
            noise_model= None,
            *args,
            **kwargs) -> typing.Union['BackendCircuit', 'BackendExpectationValue']:
    """Compile a tequila objective or circuit to a backend

    Parameters
    ----------
    objective : Objective:
        tequila objective or circuit
    variables : Dict[Union[Variable :Hashable]:RealNumber]:
        The variables of the objective given as dictionary
        with keys as tequila Variables and values the corresponding real numbers
    samples : str : (Default value = None) :
        if None a full wavefunction simulation is performed, otherwise a fixed number of samples is simulated
    backend : str : (Default value = None) :
        specify the backend or give None for automatic assignment
    noise_model: NoiseModel : (Default value =None) :
        the noise model to apply to the objective or QCircuit.

    Returns
    -------
    simulators.BackendCircuit
        simulated/sampled objective or simulated/sampled wavefunction

    """
    if variables is None and not (len(objective.extract_variables()) == 0):
        raise TequilaException(
            "You called simulate for a parametrized type but forgot to pass down the variables: {}".format(
                objective.extract_variables()))
    elif variables is not None:
        # allow hashable types as keys without casting it to variables
        variables = {assign_variable(k): v for k, v in variables.items()}

    backend = pick_backend(backend=backend, samples=samples)

    if isinstance(objective, Objective) or hasattr(objective, "args"):
        return compile_objective(objective=objective, variables=variables, backend=backend,noise_model=noise_model)
    elif hasattr(objective, "gates"):
        return compile_circuit(abstract_circuit=objective, variables=variables, backend=backend,noise_model=noise_model, *args, **kwargs)
    else:
        raise TequilaException(
            "Don't know how to compile object of type: {type}, \n{object}".format(type=type(objective),
                                                                                  object=objective))

INSTALLED_BACKENDS = {**INSTALLED_SIMULATORS, **INSTALLED_SAMPLERS}
