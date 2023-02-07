from tequila.simulators.simulator_base import BackendCircuit, QCircuit, BackendExpectationValue
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila import TequilaException, TequilaWarning
from tequila import BitString, BitNumbering, BitStringLSB
from tequila.utils.keymap import KeyMapRegisterToSubregister
import qiskit, numpy, warnings
import qiskit.providers.aer.noise as qiskitnoise
from tequila.utils import to_float
import qiskit.test.mock.backends
from qiskit.providers.ibmq import IBMQBackend


def get_bit_flip(p):
    """
    Return a bit flip error.
    Parameters
    ----------
    p: float:
        a probability.

    Returns
    -------
    type:
        qiskit pauli error
    """
    return qiskitnoise.pauli_error(noise_ops=[('X', p), ('I', 1 - p)])


def get_phase_flip(p):
    """
    Return a phase flip error in qiskit.
    Parameters
    ----------
    p: float:
        a probability.

    Returns
    -------
    type:
        qiskit pauli error
    """
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

full_basis = ['x', 'y', 'z', 'id', 'u1', 'u2', 'u3', 'h','unitary','sx',
              'cx', 'cy', 'cz', 'cu3', 'ccx']

def qiskit_device_dict():
    devices = {}
    devices.update({str(x).lower():x for x in qiskit.Aer.backends()})
    devices.update({str(x).lower():x for x in qiskit.test.mock.FakeProvider().backends()})

    return devices

class TequilaQiskitException(TequilaException):
    def __str__(self):
        return "Error in qiskit backend:" + self.message


class BackendCircuitQiskit(BackendCircuit):
    """
    Type representing circuits compiled for execution in qiskit.

    See BackendCircuit for documentation on inherited attributes and methods.


    Attributes
    ----------
    c: the number of classical channels in the circuit.
    classical_map:
        dictionary mapping qubits in tequila to classical registers representing measurement therefrom
    counter:
        counts how many distinct sympy.Symbol objects are employed in the circuit.
    noise_lookup: dict:
        dict mapping strings to qiskitnoise objects.
    numbering:
        tequila object for qubit order resolution.
    noise_model:
        a qiskit noise model built from a tequila NoiseModel
    op_lookup: dict:
        dictionary mapping strings (tequila gate names) to qiskit gate addition functions.
    pars_to_tq_: dict:
        dictionary mapping qiskit.Parameter objects back to tequila Variables and Objectives.
    q:
        the number of qubits in the circuit.
    qubit_map:
        mapping for qubit positions of gates to their location in a qiskit circuit
    resolver:
        dictionary for resolving parameters at runtime for circuits.

    tq_to_pars: dict:
        dictionary mapping tequila Variables and Objectives to qiskit.Parameters, for parameter resolution.

    Methods
    -------
    noise_model_converter:
        transform a tequila NoiseModel into a qiskit noise model.

    """
    compiler_arguments = {
        "trotterized": True,
        "swap": False,
        "multitarget": True,
        "multicontrol": True,
        "controlled_rotation": True,
        "generalized_rotation": True,
        "exponential_pauli": True,
        "controlled_exponential_pauli": True,
        "phase": True,
        "power": True,
        "hadamard_power": True,
        "controlled_power": True,
        "controlled_phase": False,
        "toffoli": False,
        "phase_to_z": False,
        "cc_max": True
    }

    numbering = BitNumbering.LSB

    def __init__(self, abstract_circuit: QCircuit, variables, qubit_map=None, noise=None,
                 device=None, *args, **kwargs):
        """

        Parameters
        ----------
        abstract_circuit: QCircuit:
            the circuit to be compiled to qiskit.
        variables: dict:
            variables to compile the circuit with
        qubit_map: dictionary:
            a qubit map which maps the abstract qubits in the abstract_circuit to the qubits on the backend
            there is no need to initialize the corresponding backend types
            the dictionary should simply be {int:int} (preferred) or {int:name}
            if None the default will map to qubits 0 ... n_qubits -1 in the backend
        noise:
            noise to apply to the circuit.
        device:
            device on which to (perhaps, via emulation) execute the circuit.
        args
        kwargs
        """
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

        self.resolver = {}
        self.tq_to_pars = {}
        self.counter = 0

        if qubit_map is None:
            qubit_map = {q: i for i, q in enumerate(abstract_circuit.qubits)}
        else:
            warnings.warn("reveived custom qubit_map = {}\n"
                          "This is not fully integrated with qiskit and might result in unexpected behaviour".format(qubit_map), TequilaWarning)

        n_qubits = max(qubit_map.values()) + 1

        self.q = qiskit.QuantumRegister(n_qubits, "q")
        self.c = qiskit.ClassicalRegister(n_qubits, "c")

        super().__init__(abstract_circuit=abstract_circuit, variables=variables, noise=noise, device=device,
                         qubit_map=qubit_map, *args, **kwargs)

        self.classical_map = self.make_classical_map(qubit_map=self.qubit_map)

        if noise != None:
            self.noise_lookup = {
                'phase damp': qiskitnoise.phase_damping_error,
                'amplitude damp': qiskitnoise.amplitude_damping_error,
                'bit flip': get_bit_flip,
                'phase flip': get_phase_flip,
                'phase-amplitude damp': qiskitnoise.phase_amplitude_damping_error,
                'depolarizing': qiskitnoise.depolarizing_error
            }

            if isinstance(noise, str): #string noise means "use the same noise as the device I tell you to get."
                try:
                    self.check_device(noise)
                    self.noise_model = qiskitnoise.NoiseModel.from_backend(noise)
                except TequilaQiskitException:
                    raise TequilaException("noise init from string requires that noise names a device. Got {}".format(noise))

            else:
                self.noise_model = self.noise_model_converter(noise)

        else:
            self.noise_model = None

        if len(self.tq_to_pars.keys()) is None:
            self.pars_to_tq = None
            self.resolver = None
        else:
            self.pars_to_tq = {v: k for k, v in self.tq_to_pars.items()}
            self.resolver = {k: to_float(v(variables)) for k, v in self.pars_to_tq.items()}

    def make_qubit_map(self, qubits: dict = None):
        qubit_map = super().make_qubit_map(qubits=qubits)
        mapped_qubits = [q.number for q in qubit_map.values()]
        for k, v in qubit_map.items():
            qubit_map[k].instance = self.q [v.number]

        return qubit_map

    def make_classical_map(self, qubit_map: dict):
        mapped_qubits = [q.number for q in qubit_map.values()]
        classical_map = {}
        for k, v in qubit_map.items():
            classical_map[k] = self.c[v.number]

        return classical_map

    def do_simulate(self, variables, initial_state=0, *args, **kwargs) -> QubitWaveFunction:
        """
        Helper function for performing simulation.
        Parameters
        ----------
        variables:
            variables to pass to the circuit for simulation.
        initial_state:
            indicate initial state on which the unitary self.circuit should act.
        args
        kwargs

        Returns
        -------
        QubitWaveFunction:
            the result of simulation.
        """
        if self.noise_model is None:
            if self.device is None:
                qiskit_backend = self.retrieve_device('statevector_simulator')
            else:
                if 'statevector' not in str(self.device):
                    raise TequilaException('For simulation, only state vector simulators are supported; recieved device={}, you might have forgoten to set the samples keyword - e.g. (device={}, samples=1000). If not set, tequila assumes that full wavefunction simualtion is demanded which is not compatible with qiskit devices or fake devices except for device=statevector'.format(self.device, self.device))
                else:
                    qiskit_backend = self.retrieve_device(self.device)
        else:
            raise TequilaQiskitException("wave function simulation with noise cannot be performed presently.")

        optimization_level = None
        if "optimization_level" in kwargs:
            optimization_level = kwargs['optimization_level']

        opts = {}
        if initial_state != 0:
            array = numpy.zeros(shape=[2 ** self.n_qubits])
            i = BitStringLSB.from_binary(BitString.from_int(integer=initial_state, nbits=self.n_qubits).binary)
            print(initial_state, " -> ", i)
            array[i.integer] = 1.0
            opts = {"initial_statevector": array}

        circuit = self.circuit.bind_parameters(self.resolver)

        qiskit_job = qiskit_backend.run(circuit,optimization_level=optimization_level,**opts)

        backend_result = qiskit_job.result()
        return QubitWaveFunction.from_array(arr=backend_result.get_statevector(circuit), numbering=self.numbering)

    def do_sample(self, circuit: qiskit.QuantumCircuit, samples: int, read_out_qubits, *args, **kwargs) -> QubitWaveFunction:
        """
        Helper function for performing sampling.
        Parameters
        ----------
        circuit: qiskit.QuantumCircuit:
            the circuit from which to sample.
        samples:
            the number of samples to take.
        args
        kwargs

        Returns
        -------
        QubitWaveFunction:
            the result of sampling.
        """
        optimization_level = 1
        if 'optimization_level' in kwargs:
            optimization_level = kwargs['optimization_level']
        if self.device is None:
            qiskit_backend = self.retrieve_device('aer_simulator')
        else:
            qiskit_backend = self.retrieve_device(self.device)

        if isinstance(qiskit_backend,IBMQBackend):
            if self.noise_model is not None:
                raise TequilaException('Cannot combine backend {} with custom noise models.'.format(str(qiskit_backend)))
            circuit = circuit.bind_parameters(self.resolver)  # this is necessary in spite of qiskit "fixing" it
            circuit = qiskit.transpile(circuit, qiskit_backend)
            return self.convert_measurements(qiskit_backend.run(circuit,shots=samples,
                                                            optimization_level=optimization_level),
                                             target_qubits=read_out_qubits)
        else:
            if isinstance(qiskit_backend, qiskit.test.mock.FakeBackend):
                circuit = circuit.bind_parameters(self.resolver)  # this is necessary in spite of qiskit "fixing" it
                coupling_map = qiskit_backend.configuration().coupling_map
                from_back = qiskitnoise.NoiseModel.from_backend(qiskit_backend)
                if self.noise_model is not None:
                    from_back = self.noise_model
                basis = from_back.basis_gates
                use_backend = self.retrieve_device('aer_simulator')
                use_backend.set_options(noise_model=from_back)
                circuit = qiskit.transpile(circuit, backend=use_backend,
                                           basis_gates=basis,
                                           coupling_map=coupling_map,
                                           optimization_level=optimization_level
                                           )

                job=qiskit_backend.run(circuit, shots=samples)
                return self.convert_measurements(job,target_qubits=read_out_qubits)
            else:
                if self.noise_model is not None:
                    qiskit_backend.set_options(noise_model=self.noise_model)  # fits better with our methodology.
                    use_basis = full_basis
                else:
                    use_basis = qiskit_backend.configuration().basis_gates
                circuit = circuit.bind_parameters(self.resolver)  # this is necessary -- see qiskit-aer issue 1346
                circuit = qiskit.transpile(circuit, backend=qiskit_backend,
                                           basis_gates=use_basis,
                                           optimization_level=optimization_level
                                           )

                job = qiskit_backend.run(circuit, shots=samples)
                return self.convert_measurements(job,
                                                 target_qubits=read_out_qubits)

    def convert_measurements(self, backend_result, target_qubits=None) -> QubitWaveFunction:
        """
        map backend results to QubitWaveFunction
        Parameters
        ----------
        backend_result:
            the result returned directly qiskit simulation.
        Returns
        -------
        QubitWaveFunction:
            measurements converted into wave function form.
        """
        qiskit_counts = backend_result.result().get_counts()
        result = QubitWaveFunction()
        # todo there are faster ways
        for k, v in qiskit_counts.items():
            converted_key = BitString.from_bitstring(other=BitStringLSB.from_binary(binary=k))
            result._state[converted_key] = v
        if target_qubits is not None:
            mapped_target = [self.qubit_map[q].number for q in target_qubits]
            mapped_full = [self.qubit_map[q].number for q in self.abstract_qubits]
            keymap = KeyMapRegisterToSubregister(subregister=mapped_target, register=mapped_full)
            result = result.apply_keymap(keymap=keymap)

        return result

    def no_translation(self, abstract_circuit):
        return isinstance(abstract_circuit, qiskit.QuantumCircuit)

    def initialize_circuit(self, *args, **kwargs):
        """
        return an empty qiskit circuit.
        Parameters
        ----------
        args
        kwargs

        Returns
        -------
        qiskit.QuantumCircuit:
            an empty qiskit circuit.
        """
        return qiskit.QuantumCircuit(self.q, self.c)

    def add_parametrized_gate(self, gate, circuit, *args, **kwargs):
        """
        add a parametrized gate to a circuit.
        Parameters
        ----------
        gate: QGateImpl:
            the  gate to apply to the circuit.
        circuit: qiskit.QuantumCircuit:
            the circuit, to apply the gate to.
        args
        kwargs

        Returns
        -------
        None

        """

        ops = self.op_lookup[gate.name]
        if len(gate.extract_variables()) > 0:
            try:
                par = self.tq_to_pars[gate.parameter]
            except:
                par = qiskit.circuit.parameter.Parameter(
                    '{}_{}'.format(self._name_variable_objective(gate.parameter), str(self.counter)))
                self.tq_to_pars[gate.parameter] = par
                self.counter += 1
        else:
            par = float(gate.parameter)
        if gate.is_controlled():
            if len(gate.control) > 2:
                raise TequilaQiskitException("multi-controls beyond 2 not yet supported for the qiskit backend. Gate was:\n{}".format(gate) )
            ops[1](circuit)(par, self.qubit(gate.control[0]), self.qubit(gate.target[0]))
        else:
            ops[0](circuit)(par, self.qubit(gate.target[0]))

    def add_measurement(self, circuit, target_qubits, *args, **kwargs):
        """
        add a measurement to a circuit.
        Parameters
        ----------
        circuit: qiskit.QuantumCircuit:
            the circuit, to apply measurement to.

        args
        kwargs

        Returns
        -------
        None

        """
        target_qubits = sorted(target_qubits)
        tq = [self.qubit(t) for t in target_qubits]
        tc = [self.classical_map[t] for t in target_qubits]
        measurement = self.initialize_circuit()
        measurement.barrier(range(self.n_qubits))
        measurement.measure(tq, tc)
        result = self.initialize_circuit()
        result = result.compose(circuit)
        result = result.compose(measurement)
        return result

    def add_basic_gate(self, gate, circuit, *args, **kwargs):
        """
        add an unparametrized gate to a circuit.
        Parameters
        ----------
        gate: QGateImpl:
            the  gate to apply to the circuit.
        circuit: qiskit.QuantumCircuit:
            the circuit, to apply the gate to.
        args
        kwargs

        Returns
        -------
        None

        """
        ops = self.op_lookup[gate.name]
        if gate.is_controlled():
            if len(gate.control) > 2:
                raise TequilaQiskitException(
                    "multi-controls beyond 2 not yet supported for the qiskit backend. Gate was:\n{}".format(gate))
            ops[len(gate.control)](circuit)(*[self.qubit(q) for q in gate.control + gate.target])
        else:
            ops[0](circuit)(*[self.qubit(q) for q in gate.target])

    def noise_model_converter(self, nm):
        """
        Convert a tequila NoiseModel to the native qiskit type.
        Parameters
        ----------
        nm: NoiseModel:
            a tequila noisemodel.

        Returns
        -------
        qiskit.NoiseModel:
            a qiskit noise model.

        """
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
                           'h',
                           'sx',
                           'unitary'
                           ]

            elif noise.level == 3:
                targets = ['ccx']

            else:
                raise TequilaQiskitException('Sorry, no support yet for qiskit for noise on more than 3 qubits.')
            qnoise.add_all_qubit_quantum_error(active, targets)

        return qnoise

    def update_variables(self, variables):
        """
        Update circuit variables for use in simulation or sampling
        Parameters
        ----------
        variables:
             a new set of variables for use in the circuit.

        Returns
        -------
        None
        """

        if self.pars_to_tq is not None:
            self.resolver = {k: to_float(v(variables)) for k, v in self.pars_to_tq.items()}
        else:
            self.resolver = None

    def check_device(self, device):
        """
        check if a device can be initialized
        Parameters
        ----------
        device:
            qiskit device or string valid for get_backend.

        Returns
        -------
        None

        """
        if device is None:
            return

        elif isinstance(device,qiskit.providers.Backend):
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
            if device.lower() in qiskit_device_dict().keys():
                return
            else:
                if qiskit.IBMQ.active_account() is None:
                    qiskit.IBMQ.load_account()
                qiskit_provider = qiskit.IBMQ.providers()[-1]
                qiskit_provider.get_backend(device)
                return
        else:
            raise TequilaQiskitException(
                'received device {} of unrecognized type {}; only None, strings, dicts, and qiskit backends allowed'.format(
                    str(device), type(device)))

    def retrieve_device(self, device):
        """
        Attempt to retrieve an instantiated qiskit device object for use in sampling.
        Parameters
        ----------
        device:
            qiskit device, or information that can be used to instantiate one.

        Returns
        -------
        type
            type is variable. Returns qiskit backend object.
        """

        if device is None:
            return device

        elif isinstance(device, qiskit.providers.Backend):
            return device

        elif isinstance(device, dict):
            qiskit_provider = device['provider']
            d = device['name'].lower()
            return qiskit_provider.get_backend(name=d)

        elif isinstance(device, str):
            check = qiskit_device_dict()
            if device.lower() in check.keys():
                return check[device.lower()]
            else:
                if qiskit.IBMQ.active_account() is None:
                    qiskit.IBMQ.load_account()
                qiskit_provider = qiskit.IBMQ.providers()[-1]
                return qiskit_provider.get_backend(device)
        else:
            raise TequilaQiskitException(
                'received device {} of unrecognized type {}; only None, strings, dicts, and qiskit backends allowed'.format(
                    str(device), type(device)))


class BackendExpectationValueQiskit(BackendExpectationValue):
    BackendCircuitType = BackendCircuitQiskit
