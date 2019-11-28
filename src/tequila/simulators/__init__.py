def supported_simulators():
    """
    :return: List of all supported simulators
    """
    return [
        "qulacs",
        "pyquil",
        "qiskit",
        "cirq"
    ]
"""
Check which simulators are installed
"""
HAS_QULACS = True
try:
    import qulacs
    from tequila.simulators.simulator_qulacs import SimulatorQulacs
    HAS_QULACS = True
except ImportError:
    HAS_QULACS = False

HAS_PYQUIL = True
from shutil import which

HAS_QVM = which("qvm") is not None
try:
    from tequila.simulators.simulator_pyquil import SimulatorPyquil

    HAS_PYQUIL = True
except ImportError:
    HAS_PYQUIL = False

if not HAS_QVM:
    HAS_PYQUIL = False

HAS_QISKIT = True
try:
    from tequila.simulators.simulator_qiskit import SimulatorQiskit

    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

HAS_CIRQ = True
try:
    from tequila.simulators.simulator_cirq import SimulatorCirq

    HAS_CIRQ = True
except ImportError:
    HAS_CIRQ = False

from tequila.simulators.simulator_symbolic import SimulatorSymbolic


def show_available_simulators() -> str:
    return "Avaliable Simulators:\n" \
           + "qiskit = " + str(HAS_QISKIT) + "\n" \
           + "cirq   = " + str(HAS_CIRQ) + "\n" \
           + "qulacs = " + str(HAS_QULACS) + "\n" \
           + "pyquil = " + str(HAS_PYQUIL) + "\n"


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
