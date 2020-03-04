from tequila.simulators.simulator_base import BackendCircuit, QCircuit, BackendExpectationValue
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila import TequilaException
from tequila import BitString, BitNumbering, BitStringLSB
import qiskit
import qiskit.providers.aer.noise as qiskitnoise


def get_bit_flip(p):
    return qiskitnoise.pauli_error(noise_ops=[('X',p),('I',1-p)])

def get_phase_flip(p):
    return qiskitnoise.pauli_error(noise_ops=[('Y',p),('I',1-p)])

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
    'rx': 1,
    'ry': 1,
    'rz': 1,
    'cx': 2,
    'cy': 2,
    'cz': 2,
    'ch':2,
    'crx': 2,
    'cry': 2,
    'crz': 2,
    'Control':2,
    'Single':1,
    'doublecontrol':3,
    'ccx':3
}

class TequilaQiskitException(TequilaException):
    def __str__(self):
        return "Error in qiskit backend:" + self.message


class BackendCircuitQiskit(BackendCircuit):
    recompile_swap = True
    recompile_multitarget = True
    recompile_controlled_rotation = True
    recompile_controlled_power = True
    recompile_power = True
    recompile_hadamard_power = True
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
        super().__init__(abstract_circuit=abstract_circuit, variables=variables,noise_model=self.noise_model, use_mapping=use_mapping, *args, **kwargs)

    def do_simulate(self, variables, initial_state=0, *args, **kwargs) -> QubitWaveFunction:
        if self.noise_model is None:
            simulator = qiskit.Aer.get_backend("statevector_simulator")
        else:
            raise TequilaQiskitException("wave function simulation with noise cannot be performed presently")
        if initial_state != 0:
            # need something like this
            # there is a keyword for the backend for tolerance on norm
            # circuit.initialize(normed_array)
            raise TequilaQiskitException("initial state for Qiskit not yet supported here")
        backend_result = qiskit.execute(experiments=self.circuit, backend=simulator,noise_model=self.noise_model).result()
        return QubitWaveFunction.from_array(arr=backend_result.get_statevector(self.circuit), numbering=self.numbering)

    def do_sample(self, circuit: qiskit.QuantumCircuit, samples: int, *args, **kwargs) -> QubitWaveFunction:
        simulator = qiskit.Aer.get_backend("qasm_simulator")
        return self.convert_measurements(qiskit.execute(experiments=circuit, backend=simulator, shots=samples,
                                                        noise_model=self.noise_model))

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

    def add_gate(self, gate, circuit, *args, **kwargs):
        if len(gate.target) > 1:
            raise TequilaQiskitException("multi targets need to be explicitly recompiled for Qiskit")
        gfunc = getattr(circuit, gate.name.lower())
        gfunc(self.qubit_map[gate.target[0]])

    def add_controlled_gate(self, gate, circuit, *args, **kwargs):
        if len(gate.target) > 1:
            raise TequilaQiskitException("multi targets need to be explicitly recompiled for Qiskit")
        if len(gate.control) == 1:
            gfunc = getattr(circuit, "c" + gate.name.lower())
            gfunc(self.qubit_map[gate.control[0]], self.qubit_map[gate.target[0]])
        elif len(gate.control) == 2:
            try:
                gfunc = getattr(circuit, "cc" + gate.name.lower())
            except AttributeError:
                raise TequilaQiskitException("Double controls are currenty only supported for CCX in quiskit")
            gfunc(self.qubit_map[gate.control[0]], self.qubit_map[gate.control[1]], self.qubit_map[gate.target[0]])
        else:
            raise TequilaQiskitException("More than two control gates currently not supported")

    def add_rotation_gate(self, gate, variables, circuit, *args, **kwargs):
        if len(gate.target) > 1:
            raise TequilaQiskitException("multi targets need to be explicitly recompiled for Qiskit")
        gfunc = getattr(circuit, gate.name.lower())
        gfunc(gate.angle(variables), self.qubit_map[gate.target[0]])

    def add_controlled_rotation_gate(self, gate, variables, circuit, *args, **kwargs):
        if len(gate.target) > 1:
            raise TequilaQiskitException("multi targets need to be explicitly recompiled for Qiskit")
        if len(gate.control) == 1:
            gfunc = getattr(circuit, "c" + gate.name.lower())
            gfunc(gate.angle(variables), self.qubit_map[gate.control[0]], self.qubit_map[gate.target[0]])
        elif len(gate.control) == 2:
            gfunc = getattr(circuit, "cc" + gate.name.lower())
            gfunc(gate.angle(variables), self.qubit_map[gate.control[0]], self.qubit_map[gate.control[1]],
                  self.qubit_map[gate.target[0]])
        else:
            raise TequilaQiskitException("More than two control gates currently not supported")

    def add_measurement(self, gate, circuit, *args, **kwargs):
        tq = [self.qubit_map[t] for t in gate.target]
        tc = [self.classical_map[t] for t in gate.target]
        circuit.measure(tq, tc)

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
        qnoise=qiskitnoise.NoiseModel()
        for noise in nm.noises:
            op=noise_lookup[noise.name]
            if op is qiskitnoise.depolarizing_error:
                active=op(noise.probs[0],gate_qubit_lookup[noise.gate.lower()])
            else:
                active=op(*noise.probs)
            if noise.gate is 'Control':
                targets=['cx',
                         'cy',
                         'cz',
                         'crx',
                         'cry',
                         'crz',
                         'ch']
            elif noise.gate is 'Single':
                targets=['x',
                         'y',
                         'z',
                         'rx',
                         'ry',
                         'rz',
                         'h']
            elif noise.gate is 'multicontrol':
                targets=['ccx']
            else:
                targets=[noise.gate.lower()]
            qnoise.add_all_qubit_quantum_error(active,targets)
        return qnoise


class BackendExpectationValueQiskit(BackendExpectationValue):
    BackendCircuitType = BackendCircuitQiskit
