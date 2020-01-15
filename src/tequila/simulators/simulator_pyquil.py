from tequila.simulators.simulatorbase import SimulatorBase, QCircuit, TequilaException, \
    SimulatorReturnType, BackendHandler
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila import BitString, BitStringLSB, BitNumbering
import subprocess

import pyquil


class TequilaPyquilException(TequilaException):
    def __str__(self):
        return "simulator_pyquil: " + self.message


class BackenHandlerPyquil(BackendHandler):
    recompile_swap = True
    recompile_multitarget = True
    recompile_controlled_rotation = False

    def fast_return(self, abstract_circuit):
        return isinstance(abstract_circuit, pyquil.Program)

    def initialize_circuit(self, qubit_map, *args, **kwargs):
        return pyquil.Program()

    def add_gate(self, gate, circuit, qubit_map, *args, **kwargs):
        circuit += getattr(pyquil.gates, gate.name.upper())(qubit_map[gate.target[0]])

    def add_controlled_gate(self, gate, qubit_map, circuit, *args, **kwargs):
        pyquil_gate = getattr(pyquil.gates, gate.name.upper())(qubit_map[gate.target[0]])
        for c in gate.control:
            pyquil_gate = pyquil_gate.controlled(qubit_map[c])
        circuit += pyquil_gate

    def add_rotation_gate(self, gate, variables, qubit_map, circuit, *args, **kwargs):
        circuit += getattr(pyquil.gates, gate.name.upper())(gate.angle(variables), qubit_map[gate.target[0]])

    def add_controlled_rotation_gate(self, gate, variables, qubit_map, circuit, *args, **kwargs):
        pyquil_gate = getattr(pyquil.gates, gate.name.upper())(gate.angle(variables), qubit_map[gate.target[0]])
        for c in gate.control:
            pyquil_gate = pyquil_gate.controlled(qubit_map[c])
        circuit += pyquil_gate

    def add_power_gate(self, gate, qubit_map, circuit, *args, **kwargs):
        raise TequilaPyquilException("PowerGates are not supported")

    def add_controlled_power_gate(self, gate, qubit_map, circuit, *args, **kwargs):
        raise TequilaPyquilException("controlled PowerGates are not supported")

    def add_measurement(self, gate, qubit_map, circuit, *args, **kwargs):
        bits = len(gate.target)
        ro = circuit.declare('ro', 'BIT', bits)
        for i, t in enumerate(gate.target):
            circuit += pyquil.gates.MEASURE(qubit_map[t], ro[i])

    def make_qubit_map(self, abstract_circuit: QCircuit):
        n_qubits = abstract_circuit.n_qubits
        qubit_map = [i for i in range(n_qubits)]
        return qubit_map


class SimulatorPyquil(SimulatorBase):

    @property
    def numbering(self):
        return BitNumbering.LSB

    backend_handler = BackenHandlerPyquil()

    def __init__(self, initialize_qvm: bool = True,*args, **kwargs):
        super().__init__(*args, **kwargs)
        if initialize_qvm:
            self.qvm = subprocess.Popen(["qvm", "-S"])
        else:
            self.qvm = None

    def __del__(self):
        if self.qvm is not None:
            self.qvm.terminate()

    def do_simulate_wavefunction(self, abstract_circuit: QCircuit, variables, initial_state=0):
        try:
            simulator = pyquil.api.WavefunctionSimulator()
            circuit = self.create_circuit(abstract_circuit=abstract_circuit, variables=variables)
            n_qubits = len(abstract_circuit.qubits)
            msb = BitString.from_int(initial_state, nbits=n_qubits)
            iprep = pyquil.Program()
            for i, val in enumerate(msb.array):
                if val > 0:
                    iprep += pyquil.gates.X(i)

            backend_result = simulator.wavefunction(iprep + circuit)
            return SimulatorReturnType(abstract_circuit=abstract_circuit,
                                       circuit=circuit,
                                       backend_result=backend_result,
                                       wavefunction=QubitWaveFunction.from_array(arr=backend_result.amplitudes,
                                                                                 numbering=self.numbering))

        except Exception as e:
            print("\n\n\n!!!!Make sure Rigettis Quantum-Virtual-Machine is running somewhere in the back!!!!\n\n\n")
            raise e
