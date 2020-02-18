from tequila.simulators.simulatorbase import QCircuit, BackendCircuit, BackendExpectationValue
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila import TequilaException
from tequila import BitString, BitNumbering

import typing, numbers

import cirq


class TequilaCirqException(TequilaException):
    def __str__(self):
        return "Error in cirq backend:" + self.message


class BackendCircuitCirq(BackendCircuit):
    recompile_swap = False
    recompile_multitarget = True
    recompile_controlled_rotation = False
    recompile_hadamard_power= False
    recompile_controlled_power = False
    recompile_power = False
    recompile_phase_to_z=True
    recompile_toffoli=False
    recompile_trotter = True

    numbering: BitNumbering = BitNumbering.MSB

    def do_simulate(self, variables, initial_state=0, *args, **kwargs) -> QubitWaveFunction:
        simulator = cirq.Simulator()
        backend_result = simulator.simulate(program=self.circuit, initial_state=initial_state)
        return QubitWaveFunction.from_array(arr=backend_result.final_state, numbering=self.numbering)

    def convert_measurements(self, backend_result: cirq.TrialResult) -> QubitWaveFunction:
        assert (len(backend_result.measurements) == 1)
        for key, value in backend_result.measurements.items():
            counter = QubitWaveFunction()
            for sample in value:
                binary = BitString.from_array(array=sample.astype(int))
                if binary in counter._state:
                    counter._state[binary] += 1
                else:
                    counter._state[binary] = 1
            return counter

    def do_sample(self, samples, circuit, *args, **kwargs) -> QubitWaveFunction:
        return self.convert_measurements(cirq.Simulator().run(program=circuit, repetitions=samples))

    def fast_return(self, abstract_circuit):
        return isinstance(abstract_circuit, cirq.Circuit)

    def initialize_circuit(self, *args, **kwargs):
        return cirq.Circuit()

    def add_gate(self, gate, circuit, *args, **kwargs):
        cirq_gate = getattr(cirq, gate.name)
        cirq_gate = cirq_gate.on(*[self.qubit_map[t] for t in gate.target])
        circuit.append(cirq_gate)

    def add_controlled_gate(self, gate, circuit, *args, **kwargs):
        cirq_gate = getattr(cirq, gate.name)
        cirq_gate = cirq_gate.on(*[self.qubit_map[t] for t in gate.target])
        cirq_gate = cirq_gate.controlled_by(*[self.qubit_map[t] for t in gate.control])
        circuit.append(cirq_gate)

    def add_rotation_gate(self, gate, variables, circuit, *args, **kwargs):
        angle = gate.angle(variables)
        cirq_gate = getattr(cirq, gate.name)(rads=angle)
        cirq_gate = cirq_gate.on(*[self.qubit_map[t] for t in gate.target])
        circuit.append(cirq_gate)

    def add_controlled_rotation_gate(self, gate, variables, circuit, *args, **kwargs):
        angle = gate.angle(variables)
        cirq_gate = getattr(cirq, gate.name)(rads=angle)
        cirq_gate = cirq_gate.on(*[self.qubit_map[t] for t in gate.target])
        cirq_gate = cirq_gate.controlled_by(*[self.qubit_map[t] for t in gate.control])
        circuit.append(cirq_gate)

    def add_power_gate(self, gate, variables, circuit, *args, **kwargs):
        power = gate.power(variables)
        cirq_gate = getattr(cirq, gate.name + "PowGate")(exponent=power)
        cirq_gate = cirq_gate.on(*[self.qubit_map[t] for t in gate.target])
        circuit.append(cirq_gate)

    def add_controlled_power_gate(self, gate, variables, circuit, *args, **kwargs):
        power = gate.power(variables)
        cirq_gate = getattr(cirq, gate.name + "PowGate")(exponent=power)
        cirq_gate = cirq_gate.on(*[self.qubit_map[t] for t in gate.target])
        cirq_gate = cirq_gate.controlled_by(*[self.qubit_map[t] for t in gate.control])
        circuit.append(cirq_gate)

    def add_measurement(self, gate, circuit, *args, **kwargs):
        qubits = [self.qubit_map[i] for i in gate.target]
        m = cirq.measure(*qubits, key=gate.name)
        circuit.append(m)

    def make_qubit_map(self, qubits) -> typing.Dict[numbers.Integral, cirq.LineQubit]:
        return {q: cirq.LineQubit(i) for i,q in enumerate(qubits)}




class BackendExpectationValueCirq(BackendExpectationValue):
    BackendCircuitType = BackendCircuitCirq