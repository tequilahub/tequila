from tequila.simulators.simulator_base import QCircuit, BackendCircuit, BackendExpectationValue
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila import TequilaException
from tequila import BitString, BitNumbering
import sympy

import numpy as np
import typing, numbers

import cirq


noise_lookup={
    'bit flip': [lambda x: cirq.bit_flip(x)],
    'phase flip': [lambda x: cirq.phase_flip(x)],
    'phase damp': [cirq.phase_damp],
    'amplitude damp':[cirq.amplitude_damp],
    'phase-amplitude damp': [cirq.amplitude_damp,cirq.phase_damp],
    'depolarizing': [lambda x: cirq.depolarize(p=(3/4)*x)]
}


map_1 = lambda x: {'exponent':x}
map_2 = lambda x: {'exponent':x/np.pi,'global_shift':-0.5}


op_lookup={
    'I':(cirq.ops.IdentityGate,None),
    'X':(cirq.ops.common_gates.XPowGate,map_1),
    'Y':(cirq.ops.common_gates.YPowGate,map_1),
    'Z':(cirq.ops.common_gates.ZPowGate,map_1),
    'H':(cirq.ops.common_gates.HPowGate,map_1),
    'Rx': (cirq.ops.common_gates.XPowGate,map_2),
    'Ry': (cirq.ops.common_gates.YPowGate,map_2),
    'Rz': (cirq.ops.common_gates.ZPowGate,map_2),
    'SWAP': (cirq.ops.SwapPowGate,None),
}



'''
potentially useful in hash table merge, leaving in
type_lookup={
    'x':[cirq.ops.pauli_gates._PauliX,cirq.ops.common_gates.XPowGate],
    'rx':[cirq.ops.common_gates.XPowGate],
    'y': [cirq.ops.pauli_gates._PauliY,cirq.ops.common_gates.YPowGate],
    'ry': [cirq.ops.common_gates.YPowGate],
    'z': [cirq.ops.pauli_gates._PauliZ,cirq.ops.common_gates.ZPowGate],
    'r':[cirq.ops.common_gates.ZPowGate,cirq.ops.common_gates.XPowGate,
         cirq.ops.common_gates.YPowGate],
    'single':[cirq.ops.pauli_gates._PauliX,cirq.ops.pauli_gates._PauliY,
                cirq.ops.pauli_gates._PauliZ,
              cirq.ops.common_gates.ZPowGate,cirq.ops.common_gates.XPowGate,
         cirq.ops.common_gates.YPowGate,cirq.ops.common_gates.HPowGate],
    'control':[cirq.ops.pauli_gates._PauliX,
               cirq.ops.pauli_gates._PauliY,
               cirq.ops.pauli_gates._PauliZ,
               cirq.ops.common_gates.ZPowGate,
               cirq.ops.common_gates.XPowGate,
               cirq.ops.common_gates.YPowGate,
               cirq.ops.common_gates.HPowGate,
               cirq.ops.common_gates.CNotPowGate,
               cirq.ops.common_gates.CZPowGate,
               cirq.ops.SwapPowGate],
    'multicontrol':[cirq.ops.pauli_gates._PauliX,
               cirq.ops.pauli_gates._PauliY,
               cirq.ops.pauli_gates._PauliZ,
               cirq.ops.common_gates.ZPowGate,
               cirq.ops.common_gates.XPowGate,
               cirq.ops.common_gates.YPowGate,
               cirq.ops.common_gates.HPowGate,
               cirq.ops.common_gates.CNotPowGate,
               cirq.ops.common_gates.CZPowGate,
               cirq.ops.SwapPowGate,
               cirq.ops.three_qubit_gates.CCXPowGate,
               cirq.ops.three_qubit_gates.CCZPowGate,
               cirq.ops.three_qubit_gates.CSwapGate
                ],
    'rz': [cirq.ops.common_gates.ZPowGate],
    'h': [cirq.ops.common_gates.HPowGate],
    'crx':[cirq.ops.common_gates.XPowGate,cirq.ops.common_gates.CNotPowGate],
    'cry': [cirq.ops.common_gates.YPowGate],
    'crz': [cirq.ops.common_gates.ZPowGate,cirq.ops.common_gates.CZPowGate],
    'cx': [cirq.ops.pauli_gates._PauliX, cirq.ops.common_gates.XPowGate,cirq.ops.common_gates.CNotPowGate],
    'cy': [cirq.ops.pauli_gates._PauliY, cirq.ops.common_gates.YPowGate],
    'cz': [cirq.ops.pauli_gates._PauliZ, cirq.ops.common_gates.ZPowGate,cirq.ops.common_gates.CZPowGate],
    'ch': [cirq.ops.common_gates.HPowGate],
    'cnot':[cirq.ops.pauli_gates._PauliX, cirq.ops.common_gates.XPowGate,cirq.ops.common_gates.CNotPowGate],
    'ccrx': [cirq.ops.pauli_gates._PauliX, cirq.ops.common_gates.XPowGate,cirq.ops.common_gates.CNotPowGate,cirq.ops.three_qubit_gates.CCXPowGate],
    'ccry': [cirq.ops.pauli_gates._PauliY, cirq.ops.common_gates.YPowGate],
    'ccrz': [cirq.ops.pauli_gates._PauliZ, cirq.ops.common_gates.ZPowGate,cirq.ops.common_gates.CZPowGate,cirq.ops.three_qubit_gates.CCZPowGate],
    'ccx': [cirq.ops.pauli_gates._PauliX, cirq.ops.common_gates.XPowGate,cirq.ops.common_gates.CNotPowGate,cirq.ops.three_qubit_gates.CCXPowGate],
    'ccnot':[cirq.ops.pauli_gates._PauliX, cirq.ops.common_gates.XPowGate,cirq.ops.common_gates.CNotPowGate,cirq.ops.three_qubit_gates.CCXPowGate],
    'ccy': [cirq.ops.pauli_gates._PauliY, cirq.ops.common_gates.YPowGate],
    'ccz': [cirq.ops.pauli_gates._PauliZ, cirq.ops.common_gates.ZPowGate,cirq.ops.common_gates.CZPowGate,cirq.ops.three_qubit_gates.CCZPowGate],
    'cch': [cirq.ops.common_gates.HPowGate],
    'swap':[cirq.ops.SwapPowGate],
}
'''


'''
potentially useful in hash table merge, leaving in
qubit_lookup ={
    'x':1,
    'rx':1,
    'y': 1,
    'ry': 1,
    'z': 1,
    'rz': 1,
    'h': 1,
    'r':1,
    'single':1,
    'control':2,
    'multicontrol':3,
    'crx':2,
    'cry': 2,
    'crz': 2,
    'cx': 2,
    'cy': 2,
    'cz': 2,
    'ch': 2,
    'cnot':2,
    'ccrx':3,
    'ccry':3,
    'ccrz':3,
    'ccx':3,
    'ccnot':3,
    'ccy':3,
    'ccz':3,
    'cch':3,
    'swap':2
}
'''

def qubit_satisfier(op,level):
    oplen=len(op.qubits)
    if level <3:
        return oplen ==level
    else:
        return oplen >=level

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

    def __init__(self, abstract_circuit: QCircuit, variables, use_mapping=True,noise_model=None, *args, **kwargs):
        self.tq_to_sympy={}
        self.counter=0
        super().__init__(abstract_circuit=abstract_circuit, variables=variables,noise_model=noise_model, use_mapping=use_mapping, *args, **kwargs)
        if len(self.tq_to_sympy.keys()) is None:
            self.sympy_to_tq = None
            self.resolver=None
        else:
            self.sympy_to_tq = {v: k for k, v in self.tq_to_sympy.items()}
            self.resolver=cirq.ParamResolver({k:v(variables) for k,v in self.sympy_to_tq.items()})
        if self.noise_model is not None:
            self.circuit=self.build_noise_model(self.noise_model)

    def do_simulate(self, variables, initial_state=0, *args, **kwargs) -> QubitWaveFunction:
        simulator = cirq.Simulator()
        backend_result = simulator.simulate(program=self.circuit,param_resolver=self.resolver, initial_state=initial_state)
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

    def do_sample(self, samples,circuit, *args, **kwargs) -> QubitWaveFunction:
        return self.convert_measurements(cirq.sample(program=circuit,param_resolver=self.resolver, repetitions=samples))

    def fast_return(self, abstract_circuit):
        return isinstance(abstract_circuit, cirq.Circuit)

    def initialize_circuit(self, *args, **kwargs):
        return cirq.Circuit()

    def add_parametrized_gate(self, gate, circuit, *args, **kwargs):
        op, mapping = op_lookup[gate.name]
        if isinstance(gate.parameter, float):
            par = gate.parameter
        else:
            try:
                par = self.tq_to_sympy[gate.parameter]
            except:
                par = sympy.Symbol('{}_{}'.format(self._name_variable_objective(gate.parameter),str(self.counter)))
                self.tq_to_sympy[gate.parameter] = par
                self.counter += 1
        cirq_gate = op(**mapping(par)).on(*[self.qubit_map[t] for t in gate.target])
        if gate.is_controlled():
            cirq_gate = cirq_gate.controlled_by(*[self.qubit_map[c] for c in gate.control])
        circuit.append(cirq_gate)

    def add_basic_gate(self, gate, circuit, *args, **kwargs):
        op, mapping = op_lookup[gate.name]
        cirq_gate = op().on(*[self.qubit_map[t] for t in gate.target])
        if gate.is_controlled():
            cirq_gate = cirq_gate.controlled_by(*[self.qubit_map[c] for c in gate.control])
        circuit.append(cirq_gate)

    def add_measurement(self, gate, circuit, *args, **kwargs):
        cirq_gate = cirq.MeasurementGate(len(gate.target)).on(*[self.qubit_map[t] for t in gate.target])
        circuit.append(cirq_gate)

    def make_qubit_map(self, qubits) -> typing.Dict[numbers.Integral, cirq.LineQubit]:
        return {q: cirq.LineQubit(i) for i,q in enumerate(qubits)}

    def build_noise_model(self,noise_model):
        c=self.circuit
        n=noise_model
        new_ops=[]
        for op in c.all_operations():
            new_ops.append(op)
            for noise in n.noises:
                if qubit_satisfier(op,noise.level):
                    for i,channel in enumerate(noise_lookup[noise.name]):
                        new_ops.append(channel(noise.probs[i]).on_each([q for q in op.qubits]))
        return cirq.Circuit.from_ops(new_ops)

    def update_variables(self, variables):
        """
        overriding the underlying base to make sure this stuff remains noisy
        """
        if self.sympy_to_tq is not None:
            self.resolver=cirq.ParamResolver({k:v(variables) for k,v in self.sympy_to_tq.items()})
        else:
            self.resolver=None

class BackendExpectationValueCirq(BackendExpectationValue):
    BackendCircuitType = BackendCircuitCirq