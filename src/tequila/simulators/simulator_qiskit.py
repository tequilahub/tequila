from tequila.simulators.simulator_base import BackendCircuit, QCircuit, BackendExpectationValue
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila import TequilaException
from tequila import BitString, BitNumbering, BitStringLSB
import qiskit, numpy
import qiskit.providers.aer.noise as qiskitnoise
from tequila.utils import to_float


def get_bit_flip(p):
    return qiskitnoise.pauli_error(noise_ops=[('X',p),('I',1-p)])

def get_phase_flip(p):
    return qiskitnoise.pauli_error(noise_ops=[('Z',p),('I',1-p)])

noise_lookup={
    'phase damp':qiskitnoise.phase_damping_error,
    'amplitude damp':qiskitnoise.amplitude_damping_error,
    'bit flip':get_bit_flip,
    'phase flip':get_phase_flip,
    'phase-amplitude damp':qiskitnoise.phase_amplitude_damping_error,
    'depolarizing':qiskitnoise.depolarizing_error
}

gate_qubit_lookup={
    'x':1,
    'y':1,
    'z':1,
    'h':1,
    'u1':1,
    'u2':1,
    'u3':1,
    'cx': 2,
    'cy': 2,
    'cz': 2,
    'ch':2,
    'cu3':2,
    'ccx':3,
    'r':1,
    'single':1,
    'control':2,
    'multicontrol':3
}

basis=['x','y','z','id','u1','u2','u3','h',
       'cx','cy','cz','cu3','ccx']
class TequilaQiskitException(TequilaException):
    def __str__(self):
        return "Error in qiskit backend:" + self.message


op_lookup={
    'I': (lambda c: c.iden),
    'X': (lambda c: c.x,lambda c: c.cx, lambda c: c.ccx),
    'Y': (lambda c: c.y,lambda c: c.cy, lambda c: c.ccy),
    'Z': (lambda c: c.z,lambda c: c.cz, lambda c: c.ccz),
    'H': (lambda c: c.h,lambda c: c.ch, lambda c: c.cch),
    'Rx': (lambda c: c.rx,lambda c: c.mcrx),
    'Ry': (lambda c: c.ry,lambda c: c.mcry),
    'Rz': (lambda c: c.rz,lambda c: c.mcrz),
    'Phase': (lambda c: c.u1,lambda c: c.cu1),
    'SWAP': (lambda c: c.swap,lambda c: c.cswap),
}


class BackendCircuitQiskit(BackendCircuit):
    recompile_swap = True
    recompile_multitarget = True
    recompile_controlled_rotation = True
    recompile_controlled_power = True
    recompile_power = True
    recompile_hadamard_power = True
    recompile_phase = False
    cc_max=True
    numbering = BitNumbering.LSB

    def __init__(self, abstract_circuit: QCircuit, variables, use_mapping=True,noise_model=None, *args, **kwargs):
        if use_mapping:
            qubits = abstract_circuit.qubits
        else:
            qubits = range(abstract_circuit.n_qubits)

        nm=self.noise_model_converter(noise_model)
        self.noise_model=nm
        n_qubits = len(qubits)
        self.q = qiskit.QuantumRegister(n_qubits, "q")
        self.c = qiskit.ClassicalRegister(n_qubits, "c")
        self.classical_map = {i: self.c[j] for j, i in enumerate(qubits)}
        self.qubit_map = {i: self.q[j] for j, i in enumerate(qubits)}
        self.resolver = {}
        self.tq_to_sympy={}
        self.counter = 0
        super().__init__(abstract_circuit=abstract_circuit, variables=variables,noise_model=self.noise_model, use_mapping=use_mapping, *args, **kwargs)
        if len(self.tq_to_sympy.keys()) is None:
            self.sympy_to_tq = None
            self.resolver = None
        else:
            self.sympy_to_tq = {v: k for k, v in self.tq_to_sympy.items()}
            self.resolver = {k: to_float(v(variables)) for k, v in self.sympy_to_tq.items()}
        if self.noise_model is None:
            self.ol = 1
        else:
            self.ol = 0

    def do_simulate(self, variables, initial_state=0, *args, **kwargs) -> QubitWaveFunction:
        if self.noise_model is None:
            simulator = qiskit.Aer.get_backend("statevector_simulator")
        else:
            raise TequilaQiskitException("wave function simulation with noise cannot be performed presently")

        opts = None
        if initial_state != 0:
            array = numpy.zeros(shape=[2**self.n_qubits])
            i = BitStringLSB.from_binary(BitString.from_int(integer=initial_state, nbits=self.n_qubits).binary)
            print(initial_state, " -> ", i)
            array[i.integer] = 1.0
            opts = {"initial_statevector": array}
            print(opts)

        backend_result = qiskit.execute(experiments=self.circuit, backend=simulator,parameter_binds=[self.resolver], backend_options=opts).result()
        return QubitWaveFunction.from_array(arr=backend_result.get_statevector(self.circuit), numbering=self.numbering)

    def do_sample(self, circuit: qiskit.QuantumCircuit, samples: int, *args, **kwargs) -> QubitWaveFunction:
        simulator = qiskit.providers.aer.QasmSimulator()
        return self.convert_measurements(qiskit.execute(circuit,backend=simulator, shots=samples,
                                                        optimization_level=0,
                                                        noise_model=self.noise_model,parameter_binds=[self.resolver]))

    def convert_measurements(self, backend_result) -> QubitWaveFunction:
        """0.
        :param qiskit_counts: qiskit counts as dictionary, states are binary in little endian (LSB)
        :return: Counts in OpenVQE format, states are big endian (MSB)
        """
        qiskit_counts = backend_result.result().get_counts()
        result = QubitWaveFunction()
        # todo there are faster ways
        for k, v in qiskit_counts.items():
            converted_key = BitString.from_bitstring(other=BitStringLSB.from_binary(binary=k))
            result._state[converted_key] = v
        return result

    def fast_return(self, abstract_circuit):
        return isinstance(abstract_circuit, qiskit.QuantumCircuit)

    def initialize_circuit(self, *args, **kwargs):
        return qiskit.QuantumCircuit(self.q, self.c)

    def add_parametrized_gate(self, gate, circuit, *args, **kwargs):
        ops = op_lookup[gate.name]
        if len(gate.extract_variables()) > 0:
            try:
                par = self.tq_to_sympy[gate.parameter]
            except:
                par = qiskit.circuit.parameter.Parameter('{}_{}'.format(self._name_variable_objective(gate.parameter),str(self.counter)))
                self.tq_to_sympy[gate.parameter] = par
                self.counter += 1
        else:
            par = float(gate.parameter)
        if gate.is_controlled():
            ops[1](circuit)(par, q_controls=[self.qubit_map[c] for c in gate.control],
                            q_target=self.qubit_map[gate.target[0]], q_ancillae=None, mode='noancilla')
        else:
            ops[0](circuit)(par, self.qubit_map[gate.target[0]])

    def add_measurement(self,gate, circuit, *args, **kwargs):
        tq = [self.qubit_map[t] for t in gate.target]
        tc = [self.classical_map[t] for t in gate.target]
        circuit.measure(tq, tc)

    def add_basic_gate(self, gate, circuit, *args, **kwargs):
        ops = op_lookup[gate.name]
        if gate.is_controlled():
            ops[len(gate.control)](circuit)(*[self.qubit_map[q] for q in gate.control+gate.target])
        else:
            ops[0](circuit)(*[self.qubit_map[q] for q in gate.target])

    def make_map(self, qubits):
        # for qiskit this is done in init
        assert(self.q is not None)
        assert(self.c is not None)
        assert(len(self.qubit_map) == len(qubits))
        assert(len(self.abstract_qubit_map) == len(qubits))
        return self.qubit_map

    def noise_model_converter(self,nm):
        if nm is None:
            return None
        basis_gates = basis
        qnoise=qiskitnoise.NoiseModel(basis_gates)
        for noise in nm.noises:
            op=noise_lookup[noise.name]
            if op is qiskitnoise.depolarizing_error:
                active=op(noise.probs[0],noise.level)
            else:
                if noise.level==1:
                    active=op(*noise.probs)
                else:
                    active=op(*noise.probs)
                    action=op(*noise.probs)
                    for i in range(noise.level-1):
                        active=active.tensor(action)

            if noise.level is 2:
                targets=['cx',
                         'cy',
                         'cz',
                         'crz',
                         'crx',
                         'cry',
                         'cu3',
                         'ch']

            elif noise.level is 1:
                targets=['x',
                         'y',
                         'z',
                         'u3',
                         'u1',
                         'u2',
                         'h']

            elif noise.level is 3:
                targets=['ccx']
            qnoise.add_all_qubit_quantum_error(active,targets)

        return qnoise

    def update_variables(self, variables):
        """
        overriding the underlying base to make sure this stuff remains noisy
        """
        if self.sympy_to_tq is not None:
            self.resolver={k : to_float(v(variables)) for k,v in self.sympy_to_tq.items()}
        else:
            self.resolver=None

class BackendExpectationValueQiskit(BackendExpectationValue):
    BackendCircuitType = BackendCircuitQiskit
