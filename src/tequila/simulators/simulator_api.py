from collections import namedtuple
import typing, warnings
from numbers import Real as RealNumber
from typing import Dict, Union, Hashable

from tequila.objective import Objective, Variable, assign_variable, format_variable_dictionary
from tequila.utils.exceptions import TequilaException, TequilaWarning
from tequila.simulators.simulator_base import BackendCircuit, BackendExpectationValue
from tequila.circuit.noise import NoiseModel

SUPPORTED_BACKENDS = ["qulacs", "qiskit", "cirq", "pyquil", "symbolic"]
SUPPORTS_DEVICES = ['qiskit','pyquil','cirq']
SUPPORTED_NOISE_BACKENDS = ["qiskit", 'cirq', 'pyquil', 'qulacs']
BackendTypes = namedtuple('BackendTypes', 'CircType ExpValueType')
INSTALLED_SIMULATORS = {}
INSTALLED_SAMPLERS = {}
HAS_QULACS = True
INSTALLED_NOISE_SAMPLERS = {}
INSTALLED_DEVICE_SAMPLERS = {}
if typing.TYPE_CHECKING:
    from tequila.objective import Objective, Variable
    from tequila.circuit.gates import QCircuit
    import numbers.Real as RealNumber
    from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction

"""
Check which simulators are installed
We are distinguishing two classes of simulators: Samplers and full wavefunction simuators
"""


HAS_QISKIT = True
try:
    from tequila.simulators.simulator_qiskit import BackendCircuitQiskit, BackendExpectationValueQiskit, check_qiskit_device,retrieve_qiskit_device
    HAS_QISKIT = True
    INSTALLED_SIMULATORS["qiskit"] = BackendTypes(BackendCircuitQiskit, BackendExpectationValueQiskit)
    INSTALLED_SAMPLERS["qiskit"] = BackendTypes(BackendCircuitQiskit, BackendExpectationValueQiskit)
    INSTALLED_NOISE_SAMPLERS["qiskit"] = BackendTypes(BackendCircuitQiskit, BackendExpectationValueQiskit)
    INSTALLED_DEVICE_SAMPLERS['qiskit'] = BackendTypes(BackendCircuitQiskit, BackendExpectationValueQiskit)
except ImportError:
    HAS_QISKIT = False

HAS_CIRQ = True
try:
    from tequila.simulators.simulator_cirq import BackendCircuitCirq, BackendExpectationValueCirq, check_cirq_device,retrieve_cirq_device

    HAS_CIRQ = True
    INSTALLED_SIMULATORS["cirq"] = BackendTypes(CircType=BackendCircuitCirq, ExpValueType=BackendExpectationValueCirq)
    INSTALLED_SAMPLERS["cirq"] = BackendTypes(CircType=BackendCircuitCirq, ExpValueType=BackendExpectationValueCirq)
    INSTALLED_NOISE_SAMPLERS["cirq"] = BackendTypes(CircType=BackendCircuitCirq,
                                                    ExpValueType=BackendExpectationValueCirq)

    INSTALLED_DEVICE_SAMPLERS['cirq'] = BackendTypes(BackendCircuitCirq, BackendExpectationValueCirq)
except ImportError:
    HAS_CIRQ = False

try:
    import qulacs
    from tequila.simulators.simulator_qulacs import BackendCircuitQulacs, BackendExpectationValueQulacs

    HAS_QULACS = True
    INSTALLED_SIMULATORS["qulacs"] = BackendTypes(CircType=BackendCircuitQulacs,
                                                  ExpValueType=BackendExpectationValueQulacs)
    INSTALLED_SAMPLERS["qulacs"] = BackendTypes(CircType=BackendCircuitQulacs,
                                                ExpValueType=BackendExpectationValueQulacs)
    INSTALLED_NOISE_SAMPLERS["qulacs"] = BackendTypes(CircType=BackendCircuitQulacs,
                                                      ExpValueType=BackendExpectationValueQulacs)
except ImportError:
    HAS_QULACS = False

HAS_PYQUIL = True

try:
    from tequila.simulators.simulator_pyquil import BackendCircuitPyquil, BackendExpectationValuePyquil, check_pyquil_device,retrieve_pyquil_device

    HAS_PYQUIL = True
    INSTALLED_SIMULATORS["pyquil"] = BackendTypes(BackendCircuitPyquil, BackendExpectationValuePyquil)
    INSTALLED_SAMPLERS["pyquil"] = BackendTypes(BackendCircuitPyquil, BackendExpectationValuePyquil)
    INSTALLED_NOISE_SAMPLERS["pyquil"] = BackendTypes(BackendCircuitPyquil, BackendExpectationValuePyquil)
    INSTALLED_DEVICE_SAMPLERS['pyquil'] = BackendTypes(BackendCircuitPyquil, BackendExpectationValuePyquil)
except ImportError:
    HAS_PYQUIL = False

from tequila.simulators.simulator_symbolic import BackendCircuitSymbolic, BackendExpectationValueSymbolic

INSTALLED_SIMULATORS["symbolic"] = BackendTypes(CircType=BackendCircuitSymbolic,
                                                ExpValueType=BackendExpectationValueSymbolic)
HAS_SYMBOLIC = True


def show_available_simulators():
    """ """
    print("{:15} | {:10} | {:10} | {:10} | {:10} | {:10}".format("backend", "wfn", "sampling", "noise", 'devices', "installed"))
    print("--------------------------------------------------------------------")
    for k in SUPPORTED_BACKENDS:
        print("{:15} | {:10} | {:10} | {:10} | {:10} | {:10}".format(k,
                                                             str(k in INSTALLED_SIMULATORS),
                                                             str(k in INSTALLED_SAMPLERS),
                                                             str(k in INSTALLED_NOISE_SAMPLERS),
                                                             str(k in INSTALLED_DEVICE_SAMPLERS),
                                                             str(k in INSTALLED_BACKENDS)))


def pick_backend(backend: str = None, samples: int = None, noise: NoiseModel = None, device: str = None,
                 exclude_symbolic: bool = True) -> str:
    """
    TODO: break this up into smaller functions!
    verifies if the backend is installed and picks one automatically if set to None
    :param backend: the demanded backend
    :param samples: if not None the simulator needs to be able to sample wavefunctions
    :param noise: if true,
    :param exclude_symbolic: only for random choice
    :return: An installed backend as string
    """
    if len(INSTALLED_SIMULATORS) == 0:
        raise TequilaException("No simulators installed on your system")

    if backend is None and device is not None:
        raise TequilaException('device use requires backend specification!')

    if backend is None:
        if noise is None:
            if samples is None:
                for f in SUPPORTED_BACKENDS:
                    if f in INSTALLED_SIMULATORS:
                        return f
            else:
                for f in INSTALLED_SAMPLERS.keys():
                    return f
        else:
            if samples is None:
                raise TequilaException(
                    "Noise requires sampling; please provide a positive, integer value for samples")
            for f in SUPPORTED_NOISE_BACKENDS:
                if noise == 'device':
                    raise TequilaException('device noise requires a device, which requires a named backend!')
                else:
                    return f
            raise TequilaException(
                            'Could not find any installed sampler!')


    if hasattr(backend, "lower"):
        backend = backend.lower()

    if backend == "random":
        if device is not None:
            raise TequilaException('cannot ask for a random backend and a specific device!')
        from numpy import random as random
        import time
        state = random.RandomState(int(str(time.process_time()).split('.')[-1]) % 2 ** 32)
        if samples is None:
            backend = state.choice(list(INSTALLED_SIMULATORS.keys()), 1)[0]
        else:
            backend = state.choice(list(INSTALLED_SAMPLERS.keys()), 1)[0]

        if exclude_symbolic:
            while (backend == "symbolic"):
                backend = state.choice(list(INSTALLED_SIMULATORS.keys()), 1)[0]
        return backend

    if backend is not None:
        if device is not None:
            if samples is not None:
                check_device_validity(device,backend)
            else:
                raise TequilaException('choosing a device requires non-zero samples!')

    if backend not in SUPPORTED_BACKENDS:
        raise TequilaException("Backend {backend} not supported ".format(backend=backend))

    elif noise is False and samples is None and backend not in INSTALLED_SIMULATORS.keys():
        raise TequilaException("Backend {backend} not installed ".format(backend=backend))
    elif noise is False and samples is not None and backend not in INSTALLED_SAMPLERS.keys():
        raise TequilaException("Backend {backend} not installed or sampling not supported".format(backend=backend))
    elif noise is not False and samples is not None and backend not in INSTALLED_NOISE_SAMPLERS.keys():
        raise TequilaException(
            "Backend {backend} not installed or else Noise has not been implemented".format(backend=backend))

    return backend

def check_device_validity(device,backend):
    if device is None:
        return
    if backend not in INSTALLED_DEVICE_SAMPLERS.keys():
        raise TequilaException('Chosen backend {backend} does not support devices!'.format(backend=backend))
    if backend == 'pyquil':
        check_pyquil_device(device)
    elif backend == 'qiskit':
        check_qiskit_device(device)
    elif backend == 'cirq':
        check_cirq_device(device)

def retrieve_device(device,backend,use_device_noise=False):
    print('retrieve device called on ', device)
    if backend not in INSTALLED_DEVICE_SAMPLERS.keys():
        raise TequilaException('Chosen backend {backend} does not support devices!'.format(backend=backend))
    if backend == 'pyquil':
        return retrieve_pyquil_device(device,use_device_noise)
    elif backend == 'qiskit':
        return retrieve_qiskit_device(device)
    elif backend == 'cirq':
        return retrieve_cirq_device(device)
    else:
        raise TequilaException('could not retrieve requested device!')


def compile_objective(objective: 'Objective',
                      variables: typing.Dict['Variable', 'RealNumber'] = None,
                      backend: str = None,
                      samples: int = None,
                      device: str = None,
                      noise: NoiseModel = None,
                      *args,
                      **kwargs) -> Objective:
    """
    Compiles an objective to a chosen backend
    The abstract circuits are replaced by the circuit objects of the backend
    Direct return if the objective was alrady compiled
    :param objective: abstract objective
    :param variables: The variables of the objective given as dictionary
    with keys as tequila Variables and values the corresponding real numbers
    :param backend: specify the backend or give None for automatic assignment
    :param noise: the NoiseModel to apply to the objective.
    :return: Compiled Objective
    """

    backend = pick_backend(backend=backend, samples=samples, noise=noise, device=device)

    # dummy variables
    if variables is None:
        variables = {k: 0.0 for k in objective.extract_variables()}

    ExpValueType = INSTALLED_SIMULATORS[pick_backend(backend=backend)].ExpValueType
    all_compiled = True
    # check if compiling is necessary
    for arg in objective.args:
        if hasattr(arg, "U") and isinstance(arg, BackendExpectationValue):
            if not isinstance(arg, ExpValueType):
                warnings.warn(
                    "Looks like part the objective was already compiled for another backend.\nFound ExpectationValue of type {} and {}\n... proceeding with hybrid\n".format(
                        type(arg), ExpValueType), TequilaWarning)
        elif hasattr(arg, "U") and not isinstance(arg, BackendExpectationValue):
            all_compiled = False

    if all_compiled:
        return objective

    compiled_args = []
    # avoid double compilations
    expectationvalues = {}
    for arg in objective.args:
        if hasattr(arg, "H") and hasattr(arg, "U") and not isinstance(arg, BackendExpectationValue):
            if arg not in expectationvalues:
                compiled_expval = ExpValueType(arg, variables=variables, noise=noise, device=device)
                expectationvalues[arg] = compiled_expval
            else:
                compiled_expval = expectationvalues[arg]
            compiled_args.append(compiled_expval)
        else:
            compiled_args.append(arg)
    return type(objective)(args=compiled_args, transformation=objective._transformation)


def compile_circuit(abstract_circuit: 'QCircuit',
                    variables: typing.Dict['Variable', 'RealNumber'] = None,
                    backend: str = None,
                    samples: int = None,
                    noise: NoiseModel = None,
                    device: str = None,
                    *args,
                    **kwargs) -> BackendCircuit:
    """
    Compile an abstract tequila circuit into a circuit corresponding to a supported backend
    direct return if the abstract circuit was already compiled
    :param abstract_circuit: The abstract tequila circuit
    :param variables: The variables of the objective given as dictionary
    with keys as tequila Variables and values the corresponding real numbers
    :param backend: specify the backend or give None for automatic assignment
    :param noise: specify a NoiseModel object to convert to the backend's noise
    :return: The compiled circuit object
    """

    CircType = INSTALLED_SIMULATORS[
        pick_backend(backend=backend, samples=samples, noise=noise, device=device)].CircType

    # dummy variables
    if variables is None:
        variables = {k: 0.0 for k in abstract_circuit.extract_variables()}

    if hasattr(abstract_circuit, "simulate"):
        if not isinstance(abstract_circuit, CircType):
            abstract_circuit = abstract_circuit.abstract_circuit
            warnings.warn(
                "Looks like the circuit was already compiled for another backend.\nChanging from {} to {}\n".format(
                    type(abstract_circuit), CircType), TequilaWarning)
        else:
            return abstract_circuit

    return CircType(abstract_circuit=abstract_circuit, variables=variables, noise=noise, device=device)


def simulate(objective: typing.Union['Objective', 'QCircuit'],
             variables: Dict[Union[Variable, Hashable], RealNumber] = None,
             samples: int = None,
             backend: str = None,
             noise: NoiseModel = None,
             device: str = None,
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
    noise: NoiseModel :
        specify a noise model to apply to simulation/sampling
    device: str:
        a string (or iff using cirq, a cirq.Device object) upon which (or in emulation of which) to sample
    *args :

    **kwargs :


    Returns
    -------
    type
        simulated/sampled objective or simulated/sampled wavefunction

    """

    variables = format_variable_dictionary(variables)

    if variables is None and not (len(objective.extract_variables()) == 0):
        raise TequilaException(
            "You called simulate for a parametrized type but forgot to pass down the variables: {}".format(
                objective.extract_variables()))

    compiled_objective = compile(objective=objective, samples=samples, variables=variables, backend=backend,
                                 noise=noise,device=device, *args, **kwargs)

    return compiled_objective(variables=variables, samples=samples, *args, **kwargs)


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
        print(objective)
        drawn = {}
        for i, E in enumerate(objective.get_expectationvalues()):
            if E in drawn:
                print("\nExpectation Value {} is the same as {}".format(i, drawn[E]))
            else:
                print("\nExpectation Value {}".format(i))
                print("Hamiltonian : ", E.H)
                print("variables : ", E.U.extract_variables())
                print("circuit:\n")
                draw(E.U)
            drawn[E] = i

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
            noise: NoiseModel = None,
            device: str = None,
            *args,
            **kwargs) -> typing.Union['BackendCircuit', 'Objective']:
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
    noise: NoiseModel : (Default value = None) :
        the noise model to apply to the objective or QCircuit.
    device: str: (Default value = None) :
        a device on which (or in emulation of which) to sample the circuit.
    Returns
    -------
    simulators.BackendCircuit
        simulated/sampled objective or simulated/sampled wavefunction

    """

    backend = pick_backend(backend=backend, noise=noise, samples=samples, device=device)

    if variables is None and not (len(objective.extract_variables()) == 0):
        variables = {key: 0.0 for key in objective.extract_variables()}
    elif variables is not None:
        # allow hashable types as keys without casting it to variables
        variables = {assign_variable(k): v for k, v in variables.items()}

    if isinstance(objective, Objective) or hasattr(objective, "args"):
        return compile_objective(objective=objective, samples=samples, variables=variables, backend=backend, noise=noise, device=device)
    elif hasattr(objective, "gates") or hasattr(objective, "abstract_circuit"):
        return compile_circuit(abstract_circuit=objective, variables=variables, backend=backend,samples=samples,
                               noise=noise, device=device, *args, **kwargs)
    else:
        raise TequilaException(
            "Don't know how to compile object of type: {type}, \n{object}".format(type=type(objective),
                                                                                  object=objective))


def compile_to_function(objective: typing.Union['Objective', 'QCircuit'], *args,
                        **kwargs) -> typing.Union['BackendCircuit', 'Objective']:
    """
    Notes
    ----------
    Same as compile but gives back callable wrapper
    where parameters are passed down as arguments instead of dictionaries
    the order of those arguments is the order of the parameter dictionary
    given here. If not given it is the order returned by objective.extract_variables()

    See compile for more information on the parameters of this function

    Returns
    -------
    wrapper over a compiled objective/circuit
    can be called like: function(0.0,1.0,...,samples=None)
    """

    compiled_objective = compile(objective, *args, **kwargs)
    if 'variables' in kwargs:
        varnames = list(kwargs['variables'].keys())
    else:
        varnames = objective.extract_variables()

    def objective_function(*fargs, **fkwargs):
        if len(fargs) != len(varnames):
            raise Exception("Compiled function takes {} variables. You passed down {} arguments."
                            "Use keywords for samples and other instructions\n"
                            "like function(a,b,c, samples=10)".format(len(varnames), len(fargs)))
        vars = {varnames[i]: fargs[i] for i, v in enumerate(fargs)}
        return compiled_objective(variables=vars, **fkwargs)

    return objective_function


INSTALLED_BACKENDS = {**INSTALLED_SIMULATORS, **INSTALLED_SAMPLERS}
