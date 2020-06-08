from tequila.simulators.simulator_base import BackendCircuit, QCircuit, BackendExpectationValue
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila import TequilaException
from tequila import BitString, BitNumbering, BitStringLSB
import qiskit, numpy
import qiskit.providers.aer.noise as qiskitnoise
from tequila.utils import to_float
import qiskit.test.mock.backends

def get_bit_flip(p):
    return qiskitnoise.pauli_error(noise_ops=[('X', p), ('I', 1 - p)])


def get_phase_flip(p):
    return qiskitnoise.pauli_error(noise_ops=[('Z', p), ('I', 1 - p)])


gate_qubit_lookup = {
    'x': 1,
    'y': 1,
    'z': 1,
    'h': 1,
    'u1': 1,
    'u2': 1,
    'u3': 1,
    'cx': 2,
    'cy': 2,
    'cz': 2,
    'ch': 2,
    'cu3': 2,
    'ccx': 3,
    'r': 1,
    'single': 1,
    'control': 2,
    'multicontrol': 3
}

full_basis = ['x', 'y', 'z', 'id', 'u1', 'u2', 'u3', 'h',
         'cx', 'cy', 'cz', 'cu3', 'ccx']


class TequilaQiskitException(TequilaException):
    def __str__(self):
        return "Error in qiskit backend:" + self.message


class BackendCircuitQiskit(BackendCircuit):
    compiler_arguments = {
        "trotterized": True,
        "swap": False,
        "multitarget": True,
        "controlled_rotation": True,
        "gaussian": True,
        "exponential_pauli": True,
        "controlled_exponential_pauli": True,
        "phase": True,
        "power": True,
        "hadamard_power": True,
        "controlled_power": True,
        "controlled_phase": False,
        "toffoli": False,
        "phase_to_z": False,
        "cc_max": False
    }

    numbering = BitNumbering.LSB

    def __init__(self, abstract_circuit: QCircuit, variables, use_mapping=True, noise=None,device=None, *args, **kwargs):

        self.op_lookup = {
            'I': (lambda c: c.iden),
            'X': (lambda c: c.x, lambda c: c.cx, lambda c: c.ccx),
            'Y': (lambda c: c.y, lambda c: c.cy, lambda c: c.ccy),
            'Z': (lambda c: c.z, lambda c: c.cz, lambda c: c.ccz),
            'H': (lambda c: c.h, lambda c: c.ch, lambda c: c.cch),
            'Rx': (lambda c: c.rx, lambda c: c.mcrx),
            'Ry': (lambda c: c.ry, lambda c: c.mcry),
            'Rz': (lambda c: c.rz, lambda c: c.mcrz),
            'Phase': (lambda c: c.u1, lambda c: c.cu1),
            'SWAP': (lambda c: c.swap, lambda c: c.cswap),
        }

        if use_mapping:
            qubits = abstract_circuit.qubits
        else:
            qubits = range(abstract_circuit.n_qubits)


        n_qubits = len(qubits)
        self.q = qiskit.QuantumRegister(n_qubits, "q")
        self.c = qiskit.ClassicalRegister(n_qubits, "c")
        self.classical_map = {i: self.c[j] for j, i in enumerate(qubits)}
        self.qubit_map = {i: self.q[j] for j, i in enumerate(qubits)}
        self.resolver = {}
        self.tq_to_sympy = {}
        self.counter = 0

        super().__init__(abstract_circuit=abstract_circuit, variables=variables, noise=noise,device=device,
                         use_mapping=use_mapping, *args, **kwargs)

        if noise != None:
            self.noise_lookup = {
                'phase damp': qiskitnoise.phase_damping_error,
                'amplitude damp': qiskitnoise.amplitude_damping_error,
                'bit flip': get_bit_flip,
                'phase flip': get_phase_flip,
                'phase-amplitude damp': qiskitnoise.phase_amplitude_damping_error,
                'depolarizing': qiskitnoise.depolarizing_error
            }

            if isinstance(noise, str):
                if noise == 'device':
                    if device is not None:
                        self.noise_model = qiskitnoise.NoiseModel.from_backend(self.device)
                    else:
                        raise TequilaException('cannot get device noise without specifying a device!')
                else:
                    raise TequilaException('The only allowed string for noise is \'device\'; recieved {}. Please try again!'.format(str(noise)))
            else:
                nm = self.noise_model_converter(noise)
                self.noise_model = nm
        else:
            self.noise_model = None

        if len(self.tq_to_sympy.keys()) is None:
            self.sympy_to_tq = None
            self.resolver = None
        else:
            self.sympy_to_tq = {v: k for k, v in self.tq_to_sympy.items()}
            self.resolver = {k: to_float(v(variables)) for k, v in self.sympy_to_tq.items()}

    def do_simulate(self, variables, initial_state=0, *args, **kwargs) -> QubitWaveFunction:
        if self.noise_model is None:
            qiskit_backend = self.retrieve_device('statevector_simulator')
        else:
            raise TequilaQiskitException("wave function simulation with noise cannot be performed presently")

        optimization_level = None
        if "optimization_level" in kwargs:
            optimization_level = kwargs['optimization_level']

        opts = None
        if initial_state != 0:
            array = numpy.zeros(shape=[2 ** self.n_qubits])
            i = BitStringLSB.from_binary(BitString.from_int(integer=initial_state, nbits=self.n_qubits).binary)
            print(initial_state, " -> ", i)
            array[i.integer] = 1.0
            opts = {"initial_statevector": array}
            print(opts)

        backend_result = qiskit.execute(experiments=self.circuit, optimization_level=optimization_level,
                                        backend=qiskit_backend, parameter_binds=[self.resolver],
                                        backend_options=opts).result()
        return QubitWaveFunction.from_array(arr=backend_result.get_statevector(self.circuit), numbering=self.numbering)

    def do_sample(self, circuit: qiskit.QuantumCircuit, samples: int, *args, **kwargs) -> QubitWaveFunction:
        optimization_level = 1
        if 'optimization_level' in kwargs:
            optimization_level = kwargs['optimization_level']
        if self.device is None:
            qiskit_backend=self.retrieve_device('qasm_simulator')
        else:
            qiskit_backend=self.device
        if qiskit_backend in qiskit.Aer.backends() or str(qiskit_backend).lower() == "ibmq_qasm_simulator":
            return self.convert_measurements(qiskit.execute(circuit, backend=qiskit_backend, shots=samples,
                                                            basis_gates=full_basis,
                                                            optimization_level=optimization_level,
                                                            noise_model=self.noise_model,
                                                            parameter_binds=[self.resolver]))
        else:
            if isinstance(qiskit_backend,qiskit.test.mock.FakeBackend):
                coupling_map=qiskit_backend.configuration().coupling_map
                basis=qiskitnoise.NoiseModel.from_backend(qiskit_backend).basis_gates
                return self.convert_measurements(qiskit.execute(circuit,self.retrieve_device('qasm_simulator'),
                                                                shots=samples,
                                                                basis_gates=basis,
                                                                coupling_map=coupling_map,
                                                                noise_model=self.noise_model,
                                                                optimization_level=optimization_level,
                                                                parameter_binds=[self.resolver]
                                                                ))
            else:
                if self.noise_model is not None:
                    print("WARNING: There are no noise models when running on real machines!")
                return self.convert_measurements(qiskit.execute(circuit, backend=qiskit_backend, shots=samples,
                                                                optimization_level=optimization_level,
                                                                parameter_binds=[self.resolver]))


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
        ops = self.op_lookup[gate.name]
        if len(gate.extract_variables()) > 0:
            try:
                par = self.tq_to_sympy[gate.parameter]
            except:
                par = qiskit.circuit.parameter.Parameter(
                    '{}_{}'.format(self._name_variable_objective(gate.parameter), str(self.counter)))
                self.tq_to_sympy[gate.parameter] = par
                self.counter += 1
        else:
            par = float(gate.parameter)
        if gate.is_controlled():
            if len(gate.control) > 2:
                pass
                # raise TequilaQiskitException("multi-controls beyond 2 not yet supported for the qiskit backend. Gate was:\n{}".format(gate) )
            ops[1](circuit)(par, q_controls=[self.qubit_map[c] for c in gate.control],
                            q_target=self.qubit_map[gate.target[0]], q_ancillae=None, mode='noancilla')
        else:
            ops[0](circuit)(par, self.qubit_map[gate.target[0]])

    def add_measurement(self, gate, circuit, *args, **kwargs):
        tq = [self.qubit_map[t] for t in gate.target]
        tc = [self.classical_map[t] for t in gate.target]
        circuit.measure(tq, tc)

    def add_basic_gate(self, gate, circuit, *args, **kwargs):
        ops = self.op_lookup[gate.name]
        if gate.is_controlled():
            if len(gate.control) > 2:
                raise TequilaQiskitException(
                    "multi-controls beyond 2 not yet supported for the qiskit backend. Gate was:\n{}".format(gate))
            ops[len(gate.control)](circuit)(*[self.qubit_map[q] for q in gate.control + gate.target])
        else:
            ops[0](circuit)(*[self.qubit_map[q] for q in gate.target])

    def make_map(self, qubits):
        # for qiskit this is done in init
        assert (self.q is not None)
        assert (self.c is not None)
        assert (len(self.qubit_map) == len(qubits))
        assert (len(self.abstract_qubit_map) == len(qubits))
        return self.qubit_map

    def noise_model_converter(self, nm):
        if nm is None:
            return None
        basis_gates = full_basis
        qnoise = qiskitnoise.NoiseModel(basis_gates)
        for noise in nm.noises:
            op = self.noise_lookup[noise.name]
            if op is qiskitnoise.depolarizing_error:
                active = op(noise.probs[0], noise.level)
            else:
                if noise.level == 1:
                    active = op(*noise.probs)
                else:
                    active = op(*noise.probs)
                    action = op(*noise.probs)
                    for i in range(noise.level - 1):
                        active = active.tensor(action)

            if noise.level == 2:
                targets = ['cx',
                           'cy',
                           'cz',
                           'crz',
                           'crx',
                           'cry',
                           'cu3',
                           'ch']

            elif noise.level == 1:
                targets = ['x',
                           'y',
                           'z',
                           'u3',
                           'u1',
                           'u2',
                           'h']

            elif noise.level == 3:
                targets = ['ccx']

            else:
                raise TequilaQiskitException('Sorry, no support yet for qiskit for noise on more than 3 qubits.')
            qnoise.add_all_qubit_quantum_error(active, targets)

        return qnoise

    def update_variables(self, variables):
        """
        overriding the underlying base to make sure this stuff remains noisy
        """
        if self.sympy_to_tq is not None:
            self.resolver = {k: to_float(v(variables)) for k, v in self.sympy_to_tq.items()}
        else:
            self.resolver = None

    def check_device(self,device):
        if device is None:
            return

        if isinstance(device, qiskit.providers.basebackend.BaseBackend):
            return

        elif isinstance(device, dict):
            try:
                qiskit_provider = device['provider']
                d = device['name'].lower()
                qiskit_provider.get_backend(name=d)
                return
            except:
                raise TequilaQiskitException('dictionary initialization with device = {} failed.'.format(str(device)))

        elif isinstance(device, str):
            l = device.lower()
            if 'fake_' in l:
                qiskit_provider=qiskit.test.mock.FakeProvider()
            else:
                if device in [str(x).lower() for x in qiskit.Aer.backends()] + [qiskit.Aer.backends()]:
                    qiskit_provider = qiskit.Aer
                else:
                    if qiskit.IBMQ.active_account() is None:
                        qiskit.IBMQ.load_account()
                    qiskit_provider = qiskit.IBMQ.get_provider()
            try:
                qiskit_provider.get_backend(name=device)
                return
            except:
                try:
                    if 'ibmq_' in device.lower():
                        l = 'fake_' + device.lower()[5:]
                    else:
                        l = 'fake_' + device.lower()
                    self.check_device(l)
                    return
                except:
                    raise TequilaQiskitException(
                            'could not obtain the desired device OR a fake version thereof from string init, with device = {}. Sorry!'.format(str(device)))
        else:
            raise TequilaQiskitException('received device {} of unrecognized type {}; only None, strings, dicts, and qiskit backends allowed'.format(str(device),type(device)))

    def retrieve_device(self, device):
        if device is None:
            return device
        if isinstance(device, qiskit.providers.basebackend.BaseBackend):
            return device

        elif isinstance(device, dict):
            qiskit_provider = device['provider']
            d = device['name'].lower()
            return qiskit_provider.get_backend(name=d)

        elif isinstance(device, str):
            device = device.lower()
            if device in [str(x).lower() for x in qiskit.Aer.backends()] + [qiskit.Aer.backends()]:
                return qiskit.Aer.get_backend(device)
            else:
                if 'fake_' in device:
                     try:
                        return qiskit.test.mock.FakeProvider().get_backend(device)
                     except:
                         raise TequilaQiskitException('Unable to retrieve fake device {}'.format(device))
                else:
                    try:
                        if qiskit.IBMQ.active_account() is None:
                            qiskit.IBMQ.load_account()
                        qiskit_provider = qiskit.IBMQ.get_provider()
                        return qiskit_provider.get_backend(name=device)
                    except:
                        print(
                            'Warning: could not get the desired backend {} from any IBMQ account. Attempting to retrieve a fake.'.format(device))
                        if 'ibmq_' in device.lower():
                            l = 'fake_' + device.lower()[5:]
                        else:
                            l = 'fake_' + device.lower()
                        return self.retrieve_device(device=l)
        else:
            raise TequilaQiskitException('received device {} of unrecognized type {}; only None, strings, dicts, and qiskit backends allowed'.format(str(device),type(device)))

class BackendExpectationValueQiskit(BackendExpectationValue):
    BackendCircuitType = BackendCircuitQiskit
