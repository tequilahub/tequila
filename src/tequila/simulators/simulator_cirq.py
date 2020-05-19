from tequila.simulators.simulator_base import QCircuit, BackendCircuit, BackendExpectationValue
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila import TequilaException
from tequila import BitString, BitNumbering
import sympy

import numpy as np
import typing, numbers

import cirq

cirq_devices = {'foxtail':cirq.google.Foxtail,
                'sycamore':cirq.google.Sycamore,
                'sycamore23':cirq.google.Sycamore23,
                'bristlecone':cirq.google.Bristlecone
               }

map_1 = lambda x: {'exponent':x}
map_2 = lambda x: {'exponent':x/np.pi,'global_shift':-0.5}

def check_cirq_device(device):
    if device is None:
        return
    if isinstance(device,cirq.Device):
        return
    else:
        assert isinstance(device,str)
        if device.lower() in ['foxtail','sycamore','sycamore23','bristlecone']:
            pass
        else:
            raise TequilaException('requested device {} could not be found!'.format(device))


def qubit_satisfier(op,level):
    oplen=len(op.qubits)
    return oplen ==level

class TequilaCirqException(TequilaException):
    def __str__(self):
        return "Error in cirq backend:" + self.message


class BackendCircuitCirq(BackendCircuit):

    compiler_arguments = {
    "trotterized" : True,
    "swap" : False,
    "multitarget" : True,
    "controlled_rotation" : False,
    "gaussian" : True,
    "exponential_pauli" : True,
    "controlled_exponential_pauli" : True,
    "phase" : True,
    "power" : False,
    "hadamard_power" : False,
    "controlled_power" : False,
    "controlled_phase" : True,
    "toffoli" : False,
    "phase_to_z" : False,
    "cc_max" : False
    }

    numbering: BitNumbering = BitNumbering.MSB

    def __init__(self, abstract_circuit: QCircuit, variables, use_mapping=True, noise=None,device=None, *args, **kwargs):


        self.op_lookup = {
            'I': (cirq.ops.IdentityGate, None),
            'X': (cirq.ops.common_gates.XPowGate, map_1),
            'Y': (cirq.ops.common_gates.YPowGate, map_1),
            'Z': (cirq.ops.common_gates.ZPowGate, map_1),
            'H': (cirq.ops.common_gates.HPowGate, map_1),
            'Rx': (cirq.ops.common_gates.XPowGate, map_2),
            'Ry': (cirq.ops.common_gates.YPowGate, map_2),
            'Rz': (cirq.ops.common_gates.ZPowGate, map_2),
            'SWAP': (cirq.ops.SwapPowGate, None),
        }

        self.tq_to_sympy={}
        self.counter=0
        if device is not None:
            self.compiler_arguments['cc_max']=True
        super().__init__(abstract_circuit=abstract_circuit, variables=variables, noise=noise, use_mapping=use_mapping, *args, **kwargs)
        if len(self.tq_to_sympy.keys()) is None:
            self.sympy_to_tq = None
            self.resolver=None
        else:
            self.sympy_to_tq = {v: k for k, v in self.tq_to_sympy.items()}
            self.resolver = cirq.ParamResolver({k:v(variables) for k,v in self.sympy_to_tq.items()})
        if device is not None:
            check_cirq_device(device)
            self.device = device
            self.circuit = self.build_device_circuit(device)
        if self.noise is not None:
            if self.noise == 'device':
                raise TequilaException('cannot get device noise for cirq yet, sorry!')
            self.noise_lookup = {
                'bit flip': [lambda x: cirq.bit_flip(x)],
                'phase flip': [lambda x: cirq.phase_flip(x)],
                'phase damp': [cirq.phase_damp],
                'amplitude damp': [cirq.amplitude_damp],
                'phase-amplitude damp': [cirq.amplitude_damp, cirq.phase_damp],
                'depolarizing': [lambda x: cirq.depolarize(p=(3 / 4) * x)]
            }
            self.circuit=self.build_noisy_circuit(self.noise)



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
        op, mapping = self.op_lookup[gate.name]
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
        op, mapping = self.op_lookup[gate.name]
        cirq_gate = op().on(*[self.qubit_map[t] for t in gate.target])
        if gate.is_controlled():
            cirq_gate = cirq_gate.controlled_by(*[self.qubit_map[c] for c in gate.control])
        circuit.append(cirq_gate)

    def add_measurement(self, gate, circuit, *args, **kwargs):
        cirq_gate = cirq.MeasurementGate(len(gate.target)).on(*[self.qubit_map[t] for t in gate.target])
        circuit.append(cirq_gate)

    def make_qubit_map(self, qubits) -> typing.Dict[numbers.Integral, cirq.LineQubit]:
        return {q: cirq.LineQubit(i) for i,q in enumerate(qubits)}

    def build_device_circuit(self,device):
        c = self.circuit
        if isinstance(device,str):
            d = cirq_devices[device.lower()]
            return self.build_device_circuit(d)
        else:
            if isinstance(device,cirq.Device):
                d=device
                if d in cirq_devices.values():
                    line = cirq.google.line_on_device(d, length=len(self.abstract_circuit.qubits))
                    for k in self.qubit_map.keys():
                        self.qubit_map[k] = line[self.qubit_map[k].x]
                    if d is cirq.google.Sycamore or d is cirq.google.Sycamore23:
                        circuit = cirq.google.optimized_for_sycamore(circuit=c, new_device=d,
                                                                     qubit_map=lambda q: line[q.x])
                    else:
                        circuit = cirq.google.optimized_for_xmon(circuit=c, new_device=d,
                                                                 qubit_map=lambda q: line[q.x])

                else:

                    if isinstance(d,cirq.NeutralAtomDevice):
                        raise TequilaException('sorry, NeutralAtoms Not Working!')
                    elif isinstance(d,cirq.IonDevice):
                        raise TequilaException('sorry,Ions not working well')
                    else:
                        raise TequilaException('sorry, I have not idea what to do with this device.')
            else:
                raise TequilaException('build_device_circuit demands a cirq.Device object')
        self.device=d
        return circuit

    def build_noisy_circuit(self,noise):
        c=self.circuit
        n=noise
        new_ops=[]
        for op in c.all_operations():
            new_ops.append(op)
            for noise in n.noises:
                if qubit_satisfier(op,noise.level):
                    for i,channel in enumerate(self.noise_lookup[noise.name]):
                        new_ops.append(channel(noise.probs[i]).on_each([q for q in op.qubits]))
        return cirq.Circuit(*new_ops)

    def update_variables(self, variables):

        if self.sympy_to_tq is not None:
            self.resolver=cirq.ParamResolver({k:v(variables) for k,v in self.sympy_to_tq.items()})
        else:
            self.resolver=None

def retrieve_cirq_device(device):
    if isinstance(device,str):
        return cirq_devices[device.lower()]
    else:
        if isinstance(device, cirq.Device):
            return device
        else:
            raise TequilaException('no clue what to do with what you gave me!')


class BackendExpectationValueCirq(BackendExpectationValue):
    BackendCircuitType = BackendCircuitCirq