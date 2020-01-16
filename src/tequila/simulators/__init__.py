SUPPORTED_SIMULATORS = ["qulacs", "pyquil", "qiskit", "cirq"]


def supported_simulators():
    """
    :return: List of all supported simulators
    """
    return SUPPORTED_SIMULATORS


"""
Check which simulators are installed
"""

INSTALLED_FULL_WFN_SIMULATORS = {}
INSTALLED_SAMPLERS = {}

HAS_QULACS = True
try:
    import qulacs
    from tequila.simulators.simulator_qulacs import SimulatorQulacs

    HAS_QULACS = True
    INSTALLED_FULL_WFN_SIMULATORS["qulacs"] = SimulatorQulacs
except ImportError:
    HAS_QULACS = False

HAS_PYQUIL = True
from shutil import which

HAS_QVM = which("qvm") is not None
try:
    from tequila.simulators.simulator_pyquil import SimulatorPyquil

    HAS_PYQUIL = True
    INSTALLED_FULL_WFN_SIMULATORS["pyquil"] = SimulatorPyquil
    # INSTALLED_SAMPLERS["pyquil"] = SimulatorPyquil # not yet implemented
except ImportError:
    HAS_PYQUIL = False

if not HAS_QVM:
    HAS_PYQUIL = False

HAS_QISKIT = True
try:
    from tequila.simulators.simulator_qiskit import SimulatorQiskit

    HAS_QISKIT = True
    INSTALLED_FULL_WFN_SIMULATORS["qiskit"] = SimulatorQiskit
    INSTALLED_SAMPLERS["qiskit"] = SimulatorQiskit
except ImportError:
    HAS_QISKIT = False

HAS_CIRQ = True
try:
    from tequila.simulators.simulator_cirq import SimulatorCirq

    HAS_CIRQ = True
    INSTALLED_FULL_WFN_SIMULATORS["cirq"] = SimulatorCirq
    INSTALLED_SAMPLERS["cirq"] = SimulatorCirq
except ImportError:
    HAS_CIRQ = False

from tequila.simulators.simulator_symbolic import SimulatorSymbolic


def show_available_simulators() -> str:
    print("Full Wavefunction Simulators:\n")
    for k in INSTALLED_FULL_WFN_SIMULATORS.keys():
        print(k)
    print("\nWavefunction Samplers:\n")
    for k in INSTALLED_SAMPLERS.keys():
        print(k)


def pick_simulator(samples=None, demand_full_wfn=None):
    if samples is None:
        # need full wavefunction simulators
        if HAS_QULACS:
            return SimulatorQulacs
        elif HAS_QISKIT:
            return SimulatorQiskit
        elif HAS_CIRQ:
            return SimulatorCirq
        elif HAS_PYQUIL:
            return SimulatorPyquil
        else:
            return SimulatorSymbolic

    elif samples is not None and demand_full_wfn:
        if HAS_QISKIT:
            return SimulatorQiskit
        if HAS_CIRQ:
            return SimulatorCirq
        else:
            raise Exception(
                "You have no simulators installed which can simulate finite measurements as well as full wavefunctions\n"
                "Use different simulators or install Cirq\n"
                "Or contribute to this package and implement a measurement sampler from full wavefunctions :-) ")
    else:
        # Measurement based simulations
        if HAS_QISKIT:
            return SimulatorQiskit
        elif HAS_CIRQ:
            return SimulatorCirq
        else:
            raise Exception(
                "You have no simulators installed which can simulate finite measurements\nInstall Qiskit or Cirq")


from tequila.simulators.simulatorbase import SimulatorBase, SimulatorReturnType
from tequila.simulators.simulator_symbolic import SimulatorSymbolic


def get_all_wfn_simulators():
    """
    :return: List of all currently availabe wfn simulators as noninitialized types
    """
    result = []
    if HAS_CIRQ:
        result.append(SimulatorCirq)
    if HAS_PYQUIL:
        result.append(SimulatorPyquil)
    if HAS_QULACS:
        result.append(SimulatorQulacs)
    return result


def get_all_samplers():
    """
    :return: List of all currently availabe sampling based simulators as noninitialized types
    """
    result = []
    if HAS_CIRQ:
        result.append(SimulatorCirq)
    if HAS_QISKIT:
        result.append(SimulatorQiskit)
    return result


def initialize_simulator(backend: str = None, samples=None, *args, **kwargs):
    """
    Initializes simulator based on simulation type demaded
    checks if necesarry simulators are installed
    :param backend: string specifying the backend, if none it will be automatically picked
    :param samples: if None a full wavefunction simulation is demaded otherwise shot based
    :return: initializes simulator object
    """

    if backend is None:
        return pick_simulator(samples=samples)

    assert (isinstance(backend, str))

    if backend not in SUPPORTED_SIMULATORS:
        raise Exception("Simulator " + backend + " is not known to by tequila")

    if backend not in INSTALLED_SAMPLERS and backend not in INSTALLED_FULL_WFN_SIMULATORS:
        raise Exception("Simulator " + backend + " is not installed on your system")

    if samples is None:
        if backend.lower() not in INSTALLED_FULL_WFN_SIMULATORS:
            raise Exception(
                "You demaded a full wavefunction simulation with the simulator " + backend + " but this is not possible")
        return INSTALLED_FULL_WFN_SIMULATORS[backend](*args, **kwargs)
    else:
        if backend.lower() not in INSTALLED_SAMPLERS:
            raise Exception(
                "You demaded shot based simulation with the simulator " + backend + " but this is not possible")
        return INSTALLED_SAMPLERS[backend](*args, **kwargs)


def simulate(objective, variables=None, samples=None, backend: str = None, *args, **kwargs):
    """
    Convenience function which automatically picks the best simulator available and runs it
    :param objective: A tequila Objective/ExpectationValue or Circuit
    :param variables: If the objective is parametrized pass down the values of the parameters as dictionary
    of the with tq.Variable as keys and float-types as Values
    :param samples: If None a full wavefunction simulation is carried out, otherwise a shot based simulation is performed
    affects the type of simulator chosen
    :param backend: specify which simulator you want to chose, can be passed down as string or as tq.SimulatorObject/type
    :return: The evaluated objective, returns an energy or a wavefunction depending on the input type
    """

    if backend is not None:
        simulator = initialize_simulator(backend=backend, samples=samples, *args, **kwargs)
    else:
        simulator = pick_simulator(samples=samples)

    return simulator(objective=objective, variables=variables, samples=samples, *args, **kwargs)

def draw(objective, backend:str=None, *args, **kwargs):
    """
    Draw a circuit or objective with the prettiest backend you have
    Circuit is translated into the backend, so avoid using this in loops
    :param objective: objective or circuit object
    :param backend: choose the backend by keyword, if None it is autopicked
    :return: pretty printout
    """

    if backend is not None:
        simulator = initialize_simulator(backend)
    else:
        if HAS_QISKIT:
            simulator = SimulatorQiskit(*args, **kwargs)
        elif HAS_CIRQ:
            simulator = SimulatorCirq(*args, **kwargs)
        else:
            simulator = SimulatorBase(*args, **kwargs)

    simulator.draw(objective=objective, *args, **kwargs)









