from openvqe.simulator.simulator import Simulator, SimulatorReturnType
from openvqe.simulator.simulator_symbolic import SimulatorSymbolic
from openvqe.qubit_wavefunction import QubitWaveFunction
from openvqe.keymap import KeyMapSubregisterToRegister

def pick_simulator(samples=None):

    if samples is None:
        try:
            from openvqe.simulator.simulator_cirq import SimulatorCirq
            return SimulatorCirq
        except:
            try:
                from openvqe.simulator.simulator_qulacs import SimulatorQulacs
                return SimulatorQulacs
            except:
                try:
                    from openvqe.simulator.simulator_pyquil import SimulatorPyquil
                    from shutil import which
                    assert(which("qvm") is not None)
                    return SimulatorPyquil
                except:
                    return SimulatorSymbolic

    else:
        try:
            from openvqe.simulator.simulator_cirq import SimulatorQiskit
            return SimulatorQiskit
        except:
            try:
                from openvqe.simulator.simulator_cirq import SimulatorCirq
                return SimulatorCirq
            except:
                raise Exception("You have no simulator installed which can simulate finite measurements")