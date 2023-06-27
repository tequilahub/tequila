from collections import namedtuple
import typing, warnings, numpy
from numbers import Real as RealNumber
from typing import Dict, Union, Hashable
import pkg_resources
from pkg_resources import DistributionNotFound

from tequila.objective import Objective, Variable, assign_variable, format_variable_dictionary, QTensor
from tequila.utils.exceptions import TequilaException, TequilaWarning
from tequila.simulators.simulator_base import BackendCircuit, BackendExpectationValue
from tequila.circuit.noise import NoiseModel

SUPPORTED_BACKENDS = ["qulacs_gpu", "qulacs",'qibo', "qiskit", "cirq", "pyquil", "symbolic", "qlm"]
SUPPORTED_NOISE_BACKENDS = ["qiskit", 'cirq', 'pyquil'] # qulacs removed in v.1.9
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
We are distinguishing two classes of simulators: Samplers and full wavefunction simulators
"""


HAS_QISKIT = True
try:
    from tequila.simulators.simulator_qiskit import BackendCircuitQiskit, BackendExpectationValueQiskit
    HAS_QISKIT = True
    INSTALLED_SIMULATORS["qiskit"] = BackendTypes(BackendCircuitQiskit, BackendExpectationValueQiskit)
    INSTALLED_SAMPLERS["qiskit"] = BackendTypes(BackendCircuitQiskit, BackendExpectationValueQiskit)
    INSTALLED_NOISE_SAMPLERS["qiskit"] = BackendTypes(BackendCircuitQiskit, BackendExpectationValueQiskit)
except ImportError:
    HAS_QISKIT = False

HAS_QIBO = True
try:
    from tequila.simulators.simulator_qibo import BackendCircuitQibo, BackendExpectationValueQibo
    HAS_QIBO = True
    INSTALLED_SIMULATORS["qibo"] = BackendTypes(BackendCircuitQibo, BackendExpectationValueQibo)
    INSTALLED_SAMPLERS["qibo"] = BackendTypes(BackendCircuitQibo, BackendExpectationValueQibo)
    INSTALLED_NOISE_SAMPLERS["qibo"] = BackendTypes(BackendCircuitQibo, BackendExpectationValueQibo)
except ImportError:
    HAS_QIBO = False

HAS_CIRQ = True
try:
    from tequila.simulators.simulator_cirq import BackendCircuitCirq, BackendExpectationValueCirq

    HAS_CIRQ = True
    INSTALLED_SIMULATORS["cirq"] = BackendTypes(CircType=BackendCircuitCirq, ExpValueType=BackendExpectationValueCirq)
    INSTALLED_SAMPLERS["cirq"] = BackendTypes(CircType=BackendCircuitCirq, ExpValueType=BackendExpectationValueCirq)
    INSTALLED_NOISE_SAMPLERS["cirq"] = BackendTypes(CircType=BackendCircuitCirq,
                                                    ExpValueType=BackendExpectationValueCirq)

except ImportError:
    HAS_CIRQ = False

try:
    pkg_resources.require("qulacs")
    import qulacs
    from tequila.simulators.simulator_qulacs import BackendCircuitQulacs, BackendExpectationValueQulacs

    HAS_QULACS = True
    INSTALLED_SIMULATORS["qulacs"] = BackendTypes(CircType=BackendCircuitQulacs,
                                                  ExpValueType=BackendExpectationValueQulacs)
    INSTALLED_SAMPLERS["qulacs"] = BackendTypes(CircType=BackendCircuitQulacs,
                                                ExpValueType=BackendExpectationValueQulacs)
    INSTALLED_NOISE_SAMPLERS["qulacs"] = BackendTypes(CircType=BackendCircuitQulacs,
                                                      ExpValueType=BackendExpectationValueQulacs)
except (ImportError, DistributionNotFound):
    HAS_QULACS = False

try:
    pkg_resources.require("qulacs-gpu")
    import qulacs
    from tequila.simulators.simulator_qulacs_gpu import BackendCircuitQulacsGpu, BackendExpectationValueQulacsGpu

    HAS_QULACS_GPU = True
    INSTALLED_SIMULATORS["qulacs_gpu"] = BackendTypes(CircType=BackendCircuitQulacsGpu,
                                                  ExpValueType=BackendExpectationValueQulacsGpu)
    INSTALLED_SAMPLERS["qulacs_gpu"] = BackendTypes(CircType=BackendCircuitQulacsGpu,
                                                ExpValueType=BackendExpectationValueQulacsGpu)
    INSTALLED_NOISE_SAMPLERS["qulacs_gpu"] = BackendTypes(CircType=BackendCircuitQulacsGpu,
                                                      ExpValueType=BackendExpectationValueQulacsGpu)
except (ImportError, DistributionNotFound):
    HAS_QULACS_GPU = False


HAS_PYQUIL = True

try:
    from tequila.simulators.simulator_pyquil import BackendCircuitPyquil, BackendExpectationValuePyquil

    HAS_PYQUIL = True
    INSTALLED_SIMULATORS["pyquil"] = BackendTypes(BackendCircuitPyquil, BackendExpectationValuePyquil)
    INSTALLED_SAMPLERS["pyquil"] = BackendTypes(BackendCircuitPyquil, BackendExpectationValuePyquil)
    INSTALLED_NOISE_SAMPLERS["pyquil"] = BackendTypes(BackendCircuitPyquil, BackendExpectationValuePyquil)
except ImportError:
    HAS_PYQUIL = False


HAS_QLM = True
try:
    from tequila.simulators.simulator_qlm import BackendCircuitQLM, BackendExpectationValueQLM

    INSTALLED_SIMULATORS["qlm"] = BackendTypes(BackendCircuitQLM, BackendExpectationValueQLM)
    INSTALLED_SAMPLERS["qlm"] = BackendTypes(BackendCircuitQLM, BackendExpectationValueQLM)
except ImportError:
    HAS_QLM = False

from tequila.simulators.simulator_symbolic import BackendCircuitSymbolic, BackendExpectationValueSymbolic

INSTALLED_SIMULATORS["symbolic"] = BackendTypes(CircType=BackendCircuitSymbolic,
                                                ExpValueType=BackendExpectationValueSymbolic)
HAS_SYMBOLIC = True


def show_available_simulators():
    """ """
    print("{:15} | {:10} | {:10} | {:10} | {:10}".format("backend", "wfn", "sampling", "noise", "installed"))
    print("--------------------------------------------------------------------")
    for k in SUPPORTED_BACKENDS:
        print("{:15} | {:10} | {:10} | {:10} | {:10}".format(k,
                                                             str(k in INSTALLED_SIMULATORS),
                                                             str(k in INSTALLED_SAMPLERS),
                                                             str(k in INSTALLED_NOISE_SAMPLERS),
                                                             str(k in INSTALLED_BACKENDS)))


def pick_backend(backend: str = None, samples: int = None, noise: NoiseModel = None, device=None,
                 exclude_symbolic: bool = True) -> str:

    """
    choose, or verify, a backend for the user.
    Parameters
    ----------
    backend: str, optional:
        what backend to choose or verify. if None: choose for the user.
    samples: int, optional:
        if int and not None, choose (verify) a simulator which supports sampling.
    noise: str or NoiseModel, optional:
        if not None, choose (verify) a simulator supports the specified noise.
    device: optional:
        verify that a given backend supports the specified device. MUST specify backend, if not None.
        if None: do not emulate or use real device.
    exclude_symbolic: bool, optional:
        whether or not to exclude the tequila debugging simulator from the available simulators, when choosing.

    Returns
    -------
    str:
        the name of the chosen (or verified) backend.
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

    if backend not in SUPPORTED_BACKENDS:
        raise TequilaException("Backend {backend} not supported ".format(backend=backend))

    elif noise is None and samples is None and backend not in INSTALLED_SIMULATORS.keys():
        raise TequilaException("Backend {backend} not installed ".format(backend=backend))
    elif noise is None and samples is not None and backend not in INSTALLED_SAMPLERS.keys():
        raise TequilaException("Backend {backend} not installed or sampling not supported".format(backend=backend))
    elif noise is not None and samples is not None and backend not in INSTALLED_NOISE_SAMPLERS.keys():
        raise TequilaException(
            "Backend {backend} not installed or else Noise has not been implemented".format(backend=backend))

    return backend


def compile_objective(objective: typing.Union['Objective'],
                      variables: typing.Dict['Variable', 'RealNumber'] = None,
                      backend: str = None,
                      samples: int = None,
                      device: str = None,
                      noise: NoiseModel = None,
                      *args,
                      **kwargs) -> Objective:
    """
    compile an objective to render it callable and return it.
    Parameters
    ----------
    objective: Objective:
        the objective to compile
    variables: dict, optional:
        the variables to compile the objective with. Will autogenerate zeros for all variables if not supplied.
    backend: str, optional:
        the backend to compile the objective to.
    samples: int, optional:
        only matters if not None; compile the objective for sampling/verify backend can do so
    device: optional:
        the device on which the objective should (perhaps emulatedly) sample.
    noise: str or NoiseModel, optional:
        the noise to apply to all circuits in the objective.
    args
    kwargs

    Returns
    -------
    Objective:
        the compiled objective.
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

    argsets = objective.argsets
    compiled_sets = []
    for argset in argsets:
        compiled_args = []
        # avoid double compilations
        expectationvalues = {}
        for arg in argset:
            if hasattr(arg, "H") and hasattr(arg, "U") and not isinstance(arg, BackendExpectationValue):
                if arg not in expectationvalues:
                    compiled_expval = ExpValueType(arg, variables=variables, noise=noise, device=device, *args, **kwargs)
                    expectationvalues[arg] = compiled_expval
                else:
                    compiled_expval = expectationvalues[arg]
                compiled_args.append(compiled_expval)
            else:
                compiled_args.append(arg)
        compiled_sets.append(compiled_args)
    if isinstance(objective, Objective):
        return type(objective)(args=compiled_sets[0], transformation=objective.transformation)


def compile_circuit(abstract_circuit: 'QCircuit',
                    variables: typing.Dict['Variable', 'RealNumber'] = None,
                    backend: str = None,
                    samples: int = None,
                    noise: NoiseModel = None,
                    device: str = None,
                    *args,
                    **kwargs) -> BackendCircuit:
    """
    compile a circuit to render it callable and return it.
    Parameters
    ----------
    abstract_circuit: QCircuit:
        the circuit to compile
    variables: dict, optional:
        the variables to compile the circuit with.
    backend: str, optional:
        the backend to compile the circuit to.
    samples: int, optional:
        only matters if not None; compile the circuit for sampling/verify backend can do so
    device: optional:
        the device on which the circuit should (perhaps emulatedly) sample.
    noise: str or NoiseModel, optional:
        the noise to apply to the circuit
    args
    kwargs

    Returns
    -------
    BackendCircuit:
        the compiled circuit.
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

    return CircType(abstract_circuit=abstract_circuit, variables=variables, noise=noise, device=device, *args, **kwargs)


def simulate(objective: typing.Union['Objective', 'QCircuit','QTensor'],
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
    objective: Objective:
        tequila objective or circuit
    variables: Dict:
        The variables of the objective given as dictionary
        with keys as tequila Variables/hashable types and values the corresponding real numbers
    samples : int, optional:
        if None a full wavefunction simulation is performed, otherwise a fixed number of samples is simulated
    backend : str, optional:
        specify the backend or give None for automatic assignment
    noise: NoiseModel, optional:
        specify a noise model to apply to simulation/sampling
    device:
        a device upon which (or in emulation of which) to sample
    *args :

    **kwargs :
        read_out_qubits = list[int] (define the qubits which shall be measured, has only effect on pure QCircuit simulation with samples)

    Returns
    -------
    float or QubitWaveFunction
        the result of simulation.
    """

    variables = format_variable_dictionary(variables)

    if variables is None and not (len(objective.extract_variables()) == 0):
        raise TequilaException(
            "You called simulate for a parametrized type but forgot to pass down the variables: {}".format(
                objective.extract_variables()))

    compiled_objective = compile(objective=objective, samples=samples, variables=variables, backend=backend,
                                 noise=noise,device=device, *args, **kwargs)

    return compiled_objective(variables=variables, samples=samples, *args, **kwargs)


def draw(objective, variables=None, backend: str = None, name=None, *args, **kwargs):
    """
    Pretty output (depends on installed backends) for jupyter notebooks
    or similar HTML environments

    Parameters
    ----------
    objective :
        the tequila objective to print out
    variables : optional:
         Give variables if the objective is parametrized (not necesarry for displaying)
    name: optional:
         Name the objective (changes circuit filenames for qpic backend)
    backend: str, optional:
         chose preferred backend (of None or not found it will be automatically picked)
    """
    if backend not in INSTALLED_SIMULATORS:
        backend = None
    if name is None:
        name = abs(hash("tmp"))

    if backend is None:
        from tequila.circuit.qpic import system_has_qpic
        if system_has_qpic:
            backend = "qpic"
        elif "cirq" in INSTALLED_SIMULATORS:
            backend = "cirq"
        elif "qiskit" in INSTALLED_SIMULATORS:
            backend = "qiskit"

    if isinstance(objective, QTensor):
        print("won't draw out all objectives in a tensor")
        print(objective)

    if isinstance(objective, Objective):
        print(objective)
        drawn = {}
        for i, E in enumerate(objective.get_expectationvalues()):
            if E in drawn:
                print("\nExpectation Value {} is the same as {}".format(i, drawn[E]))
            else:
                print("\nExpectation Value {}:".format(i))
                measurements = E.count_measurements()
                print("total measurements = {}".format(measurements))
                variables = E.U.extract_variables()
                print("variables          = {}".format(len(variables)))
                filename = "{}_{}.png".format(name,i)
                print("circuit            = {}".format(filename))
                draw(E.U, backend=backend, filename=filename)
            drawn[E] = i
    else:
        if backend is None:
            print(objective)
        elif backend.lower() in ["qpic", "html"]:
            try:
                import IPython
                import qpic
                from tequila.circuit.qpic import export_to
                if "filename" not in kwargs:
                    kwargs["filename"] = "tmp_{}.png".format(hash(backend))

                circuit = objective
                if hasattr(circuit, "U"):
                    circuit = circuit.U
                if hasattr(circuit, "abstract_circuit"):
                    circuit = objective.abstract_circuit

                export_to(circuit=circuit, *args, **kwargs)
                width=None # full size
                height=200
                if "width" in kwargs:
                    width=kwargs["width"]
                if "height" in kwargs:
                    height=kwargs["height"] # this is buggy in jupyter and will be ignored
                image=IPython.display.Image(filename=kwargs["filename"], height=height, width=width)
                IPython.display.display(image)

            except ImportError as E:
                raise Exception("Original Error Message:{}\nYou are missing dependencies for drawing: You need IPython, qpic and pdfatex.\n".format(E))
        else:
            compiled = compile_circuit(abstract_circuit=objective, backend=backend)
            if backend == "qiskit":
                return compiled.circuit.draw(*args, **kwargs)
            else:
                print(compiled.circuit)
                return ""

def compile(objective: typing.Union['Objective', 'QCircuit', 'QTensor'],
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
    objective: Objective:
        tequila objective or circuit
    variables: dict, optional:
        The variables of the objective given as dictionary
        with keys as tequila Variables and values the corresponding real numbers
    samples: int, optional:
        if None a full wavefunction simulation is performed, otherwise a fixed number of samples is simulated
    backend : str, optional:
        specify the backend or give None for automatic assignment
    noise: NoiseModel, optional:
        the noise model to apply to the objective or QCircuit.
    device: optional:
        a device on which (or in emulation of which) to sample the circuit.
    Returns
    -------
    simulators.BackendCircuit or Objective
        the compiled object.

    """

    backend = pick_backend(backend=backend, noise=noise, samples=samples, device=device)

    if variables is not None:
        # allow hashable types as keys without casting it to variables
        variables = {assign_variable(k): v for k, v in variables.items()}

    if isinstance(objective, QTensor):
        ff = numpy.vectorize(compile_objective)
        return ff(objective=objective, samples=samples, variables=variables, backend=backend, noise=noise, device=device, *args, **kwargs)
    
    if isinstance(objective, Objective) or hasattr(objective, "args"):
        return compile_objective(objective=objective, samples=samples, variables=variables, backend=backend, noise=noise, device=device, *args, **kwargs)
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
    BackendCircuit or Objective:
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
