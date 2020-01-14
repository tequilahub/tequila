from tequila.simulators.simulatorbase import SimulatorBase, QCircuit, SimulatorReturnType, BackendHandler
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila import TequilaException
from tequila import BitString, BitNumbering
import typing
import cirq


class BackenHandlerCirq(BackendHandler):

    recompile_swap = False
    recompile_multitarget = True
    recompile_controlled_rotation = False

    def fast_return(self, abstract_circuit):
        return isinstance(abstract_circuit, cirq.Circuit)

    def initialize_circuit(self, *args, **kwargs):
        return cirq.Circuit()

    def add_gate(self, gate, circuit, qubit_map, *args, **kwargs):
        cirq_gate = getattr(cirq, gate.name)
        cirq_gate = cirq_gate.on(*[qubit_map[t] for t in gate.target])
        circuit.append(cirq_gate)

    def add_controlled_gate(self, gate, qubit_map, circuit, *args, **kwargs):
        cirq_gate = getattr(cirq, gate.name)
        cirq_gate = cirq_gate.on(*[qubit_map[t] for t in gate.target])
        cirq_gate = cirq_gate.controlled_by(*[qubit_map[t] for t in gate.control])
        circuit.append(cirq_gate)

    def add_rotation_gate(self, gate, qubit_map, circuit, *args, **kwargs):
        cirq_gate = getattr(cirq, gate.name)(rads=gate.angle())
        cirq_gate = cirq_gate.on(*[qubit_map[t] for t in gate.target])
        circuit.append(cirq_gate)

    def add_controlled_rotation_gate(self, gate, qubit_map, circuit, *args, **kwargs):
        cirq_gate = getattr(cirq, gate.name)(rads=gate.angle())
        cirq_gate = cirq_gate.on(*[qubit_map[t] for t in gate.target])
        cirq_gate = cirq_gate.controlled_by(*[qubit_map[t] for t in gate.control])
        circuit.append(cirq_gate)

    def add_power_gate(self, gate, qubit_map, circuit, *args, **kwargs):
        cirq_gate = getattr(cirq, gate.name + "PowGate")(exponent=gate.power())
        cirq_gate = cirq_gate.on(*[qubit_map[t] for t in gate.target])
        circuit.append(cirq_gate)

    def add_controlled_power_gate(self, gate, qubit_map, circuit, *args, **kwargs):
        cirq_gate = getattr(cirq, gate.name + "PowGate")(exponent=gate.power())
        cirq_gate = cirq_gate.on(*[qubit_map[t] for t in gate.target])
        cirq_gate = cirq_gate.controlled_by(*[qubit_map[t] for t in gate.control])
        circuit.append(cirq_gate)

    def add_measurement(self, gate, qubit_map, circuit, *args, **kwargs):
        qubits = [qubit_map[i] for i in gate.target]
        m = cirq.measure(*qubits, key=gate.name)
        circuit.append(m)

    def make_qubit_map(self, abstract_circuit: QCircuit):
        n_qubits = abstract_circuit.n_qubits
        qubit_map = [cirq.LineQubit(i) for i in range(n_qubits)]
        return qubit_map


class TequilaCirqException(TequilaException):
    def __str__(self):
        return "Error in cirq backend:" + self.message

class SimulatorCirq(SimulatorBase):

    numbering: BitNumbering = BitNumbering.MSB

    backend_handler = BackenHandlerCirq()

    def convert_measurements(self, backend_result: cirq.TrialResult) -> typing.Dict[str, QubitWaveFunction]:
        result = dict()
        for key, value in backend_result.measurements.items():
            counter = QubitWaveFunction()
            for sample in value:
                binary = BitString.from_array(array=sample.astype(int))
                if binary in counter._state:
                    counter._state[binary] += 1
                else:
                    counter._state[binary] = 1
            result[key] = counter
        return result

    def do_run(self, circuit: cirq.Circuit, samples: int = 1) -> cirq.TrialResult:
        return cirq.Simulator().run(program=circuit, repetitions=samples)

    def do_simulate_wavefunction(self, abstract_circuit: QCircuit, initial_state=0) -> SimulatorReturnType:
        simulator = cirq.Simulator()
        circuit = self.create_circuit(abstract_circuit=abstract_circuit)
        backend_result = simulator.simulate(program=circuit, initial_state=initial_state)
        return SimulatorReturnType(abstract_circuit=abstract_circuit,
                                   circuit=circuit,
                                   wavefunction=QubitWaveFunction.from_array(arr=backend_result.final_state,
                                   numbering=self.numbering),
                                   backend_result=backend_result)

    def do_simulate_density_matrix(self, circuit: cirq.Circuit, initial_state=0):
        simulator = cirq.DensityMatrixSimulator()
        result = SimulatorReturnType(result=simulator.simulate(program=circuit, initial_state=initial_state),
                                     circuit=circuit)
        result.density_matrix = result.backend_result.final_density_matrix
        return result
