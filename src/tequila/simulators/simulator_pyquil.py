from tequila.simulators.simulator_base import QCircuit, TequilaException, BackendCircuit, BackendExpectationValue
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila import BitString, BitNumbering
import numpy as np
import pyquil
from pyquil.api import get_qc
from pyquil.noise import combine_kraus_maps
from tequila.utils import to_float

name_dict = {
    'I': 'I',
    'ry': 'parametrized',
    'rx': 'parametrized',
    'rz': 'parametrized',
    'Rz': 'parametrized',
    'Ry': 'parametrized',
    'Rx': 'parametrized',
    'RZ': 'parametrized',
    'RY': 'parametrized',
    'RX': 'parametrized',
    'r': 'parametrized',
    'X': 'X',
    'x': 'X',
    'Y': 'Y',
    'y': 'Y',
    'Z': 'Z',
    'z': 'Z',
    'Cz': 'control',
    'CZ': 'control',
    'cz': 'control',
    'SWAP': 'control',
    'CX': 'control',
    'Cx': 'control',
    'cx': 'control',
    'CNOT': 'control',
    'ccx': 'multicontrol',
    'CCx': 'multicontrol',
    'CSWAP': 'multicontrol',
    'H': 'H',
    'h': 'H',
    'Phase': 'parametrized',
    'PHASE': 'parametrized'
}

gate_qubit_lookup = {
    'X': 1,
    'Y': 1,
    'Z': 1,
    'H': 1,
    'RX': 1,
    'RY': 1,
    'RZ': 1,
    'CX': 2,
    'CY': 2,
    'CZ': 2,
    'CH': 2,
    'CRX': 2,
    'CRY': 2,
    'CRZ': 2,
    'CNOT': 2,
    'CCNOT': 3
}

name_unitary_dict = {
    'I': np.eye(2),
    'X': np.array([[0., 1.], [1., 0.]]),
    'Y': np.array([[0., -1.j], [1.j, 0.]]),
    'Z': np.array([[1., 0.], [0., -1.]]),
    'H': np.array([[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]]),
    'CNOT': np.array([[1., 0., 0., 0.],
                      [0., 1., 0., 0., ],
                      [0., 0., 0., 1.],
                      [0., 0., 1.0, 0.]
                      ]),
    'SWAP': np.array([[1., 0., 0., 0.],
                      [0., 0., 1., 0.],
                      [0., 1., 0., 0.],
                      [0., 0., 0., 1.]
                      ]),
    'CCNOT': np.array([[1., 0., 0., 0., 0., 0., 0., 0.],
                       [0., 1., 0., 0., 0., 0., 0., 0.],
                       [0., 0., 1., 0., 0., 0., 0., 0.],
                       [0., 0., 0., 1., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 1., 0., 0., 0.],
                       [0., 0., 0., 0., 0., 1., 0., 0.],
                       [0., 0., 0., 0., 0., 0., 0., 1.],
                       [0., 0., 0., 0., 0., 0., 1., 0.]
                       ]),
}


def amp_damp_map(p):
    """
    Generate the Kraus operators corresponding to an amplitude damping
    noise channel.

    :params float p: The one-step damping probability.
    :return: A list [k1, k2] of the Kraus operators that parametrize the map.
    :rtype: list
    """
    damping_op = np.sqrt(p) * np.array([[0, 1],
                                        [0, 0]])

    residual_kraus = np.diag([1, np.sqrt(1 - p)])
    return [residual_kraus, damping_op]


def phase_damp_map(p):
    """
    returns the kraus operators for phase damping with probability p
    Parameters
    ----------
    p: float:
        a probability.

    Returns
    -------
    list of numpy.ndarray:
        the krauss maps for phase damping
    """
    mat1 = np.array([[1, 0], [0, np.sqrt(1 - p)]])
    mat2 = np.array([[0, 0], [0, np.sqrt(p)]])
    return [mat1, mat2]


def bit_flip_map(p):
    """
    returns the kraus operators for bit flip with probability p
    Parameters
    ----------
    p: float:
        a probability.

    Returns
    -------
    list of numpy.ndarray:
        the kraus maps for bit flip
    """

    mat1 = np.array([[np.sqrt(1 - p), 0], [0, np.sqrt(1 - p)]])
    mat2 = np.array([[0, np.sqrt(p)], [np.sqrt(p), 0]])
    return [mat1, mat2]


def phase_flip_map(p):
    """
    returns the kraus operators for phase flip with probability p
    Parameters
    ----------
    p: float:
        a probability.

    Returns
    -------
    list of numpy.ndarray:
        the kraus maps for phase flipping
    """
    mat1 = np.array([[np.sqrt(1 - p), 0], [0, np.sqrt(1 - p)]])
    mat2 = np.array([[np.sqrt(p), 0], [0, -np.sqrt(p)]])
    return [mat1, mat2]


def phase_amp_damp_map(a, b):
    """
    the kraus maps for combined phase amplitude damping
    Parameters
    ----------
    a: float:
        a probability.
    b: float:
        a probability.

    Returns
    -------
    list of numpy.ndarray:
        the kraus maps for combined phase amplitude damping.
    """

    A0 = [[1, 0], [0, np.sqrt(1 - a - b)]]
    A1 = [[0, np.sqrt(a)], [0, 0]]
    A2 = [[0, 0], [0, np.sqrt(b)]]
    return [np.array(k) for k in [A0, A1, A2]]


def depolarizing_map(p):
    """
    the kraus maps for symmetric depolarizing
    Parameters
    ----------
    p: float:
        a probability.

    Returns
    -------
    list of numpy.ndarray:
        the depolarizing error kraus maps.

    """
    mat1 = np.array([[np.sqrt(1 - 3 * p / 4), 0], [0, np.sqrt(1 - 3 * p / 4)]])
    mat2 = np.array([[np.sqrt(p / 4), 0], [0, -np.sqrt(p / 4)]])
    mat3 = np.array([[0, np.sqrt(p / 4)], [np.sqrt(p / 4), 0]])
    mat4 = np.array([[0., -1.j * np.sqrt(p / 4)], [1.j * np.sqrt(p / 4), .0]])
    return [mat1, mat2, mat3, mat4]


def kraus_tensor(klist, n):
    """
    Recursive function that produces every (n-fold) tensor product of a list of kraus operators.
    Parameters
    ----------
    klist: list:
        a list of numpy.ndarrays, to tensor together
    n: int:
        the number of terms that must ultimately be tensored together into a single term (the number of qubits acted on)
    Returns
    -------
    list:
        a list of kraus operators

    """
    if n == 1:
        return klist
    if n == 2:
        return [np.kron(k1, k2) for k1 in klist for k2 in klist]
    elif n >= 3:
        return [np.kron(k1, k2) for k1 in kraus_tensor(klist, n - 1) for k2 in klist]
    else:
        raise TequilaPyquilException('wtf, you gave me n={}'.format(str(n)))


def append_kraus_to_gate(kraus_ops, g, level):
    """
    Combines the unitary of some gate with the n-fold tensor product of a list of kraus operators
    Parameters
    ----------
    kraus_ops: list:
        a list of kraus operators to apply to a gate.
    g:
        a gate, more specifically a unitary.
    level:
        n, the number of terms that should occur in a given tensor-product of kraus operators.

    Returns
    -------
    list:
        a list of matrices corresponding to the action of a gate followed by kraus operations.
    """

    return [kj.dot(g) for kj in kraus_tensor(kraus_ops, level)]


def add_controls(matrix, count):
    """
    Take a unitary matrix and return a controlled version thereof.
    Parameters
    ----------
    matrix:
        a unitary matrix.
    count:
        the number of control qubits to add.

    Returns
    -------
    numpy.ndarray:
        a matrix corresponding to the count-fold controlled version of the matrix 'matrix'.
    """
    gc = np.log2(matrix.shape[0])
    controls = count - gc
    if int(controls) == 0:
        return matrix
    new = np.eye(2 ** count)
    new[-matrix.shape[0]:, -matrix.shape[0]:] = matrix
    return new


def unitary_maker(gate):
    """
    Take a gate and return a unitary, potentially controlled.

    Parameters
    ----------
    gate:
        the gate whose matrix is sought

    Returns
    -------
    numpy.ndarray:
        the matrix corresponding to a given gate.
    """
    return add_controls(name_unitary_dict[gate.name], len(gate.qubits))


class TequilaPyquilException(TequilaException):
    def __str__(self):
        return "simulator_pyquil: " + self.message


class BackendCircuitPyquil(BackendCircuit):
    """
    Class representing circuits compiled for execution in Pyquil.

    See BackendCircuit for methods and attributes inherited therefrom.

    Attributes
    ----------
    counter: int
        an integer counting the number of distinct parameters (as seen by pyquil) that appear in the circuit.
    match_dummy_to_value: dict:
        a dictionary mapping pyquil variable instantiators to floats.
    match_par_to_dummy: dict:
        a dictionary mapping tequila Variables and Objectives to parameter instantiators for pyquil
    noise_lookup:
        dictionary matching strings to functions which return lists of kraus operators.
    numbering:
        a bitnumbering object, that informs tequila about the endianness of measurement.
    op_lookup:
        dictionary matching string to pyquil gates.
    resolver:
        dictionary resolving parameters for simulation.

    Methods
    -------
    build_noisy_circuit:
        takes in a noise model and a circuit and applies noise to it.

    """
    compiler_arguments = {
        "trotterized": True,
        "swap": False,
        "multitarget": True,
        "controlled_rotation": False,
        "generalized_rotation": True,
        "exponential_pauli": True,
        "controlled_exponential_pauli": True,
        "phase": False,
        "power": True,
        "hadamard_power": True,
        "controlled_power": True,
        "controlled_phase": False,
        "toffoli": False,
        "phase_to_z": False,
        "cc_max": False
    }

    numbering = BitNumbering.LSB

    def __init__(self, abstract_circuit: QCircuit, variables, qubit_map=None, noise=None, device=None, *args,
                 **kwargs):
        """
        Parameters
        ----------
        abstract_circuit: QCircuit:
            Tequila unitary to compile to Pyquil.
        variables: dict:
            values of all variables in the circuit, to compile with.
        qubit_map: dictionary:
            a qubit map which maps the abstract qubits in the abstract_circuit to the qubits on the backend
            there is no need to initialize the corresponding backend types
            the dictionary should simply be {int:int} (preferred) or {int:name}
            if None the default will map to qubits 0 ... n_qubits -1 in the backend
        noise:
            Noise to apply to the circuit.
        device:
            device on which to emulatedly execute all sampling.
        args
        kwargs
        """

        self.op_lookup = {
            'I': (pyquil.gates.I),
            'X': (pyquil.gates.X, pyquil.gates.CNOT, pyquil.gates.CCNOT),
            'Y': (pyquil.gates.Y,),
            'Z': (pyquil.gates.Z, pyquil.gates.CZ),
            'H': (pyquil.gates.H,),
            'Rx': pyquil.gates.RX,
            'Ry': pyquil.gates.RY,
            'Rz': pyquil.gates.RZ,
            'Phase': pyquil.gates.PHASE,
            'SWAP': (pyquil.gates.SWAP, pyquil.gates.CSWAP),
        }
        self.match_par_to_dummy = {}
        self.counter = 0
        if device is not None:
            self.compiler_arguments['cc_max'] = True
        super().__init__(abstract_circuit=abstract_circuit, variables=variables, noise=noise, device=device,
                         qubit_map=qubit_map, *args, **kwargs)
        if self.noise is not None:
            self.noise_lookup = {
                'amplitude damp': amp_damp_map,
                'phase damp': phase_damp_map,
                'bit flip': bit_flip_map,
                'phase flip': phase_flip_map,
                'phase-amplitude damp': phase_amp_damp_map,
                'depolarizing': depolarizing_map
            }

            if isinstance(self.noise, str):
                if self.noise == 'device':
                    pass
                else:
                    raise TequilaException(
                        'noise was a string: {}, which is not \'device\'. This is not allowed!'.format(self.noise))

            else:
                self.circuit = self.build_noisy_circuit(self.circuit, self.noise)

        if len(self.match_par_to_dummy.keys()) is None:
            self.match_dummy_to_value = None
            self.resolver = None
        else:
            self.match_dummy_to_value = {'theta_{}'.format(str(i)): k for i, k in
                                         enumerate(self.match_par_to_dummy.keys())}
            self.resolver = {k: [to_float(v(variables))] for k, v in self.match_dummy_to_value.items()}

    def do_simulate(self, variables, initial_state, *args, **kwargs):
        """
        Internal helper function for performing wavefunction simulation.

        Parameters
        ----------
        variables:
            the variables of parameters in the circuit to use during simulation
        initial_state:
            indicates, in some fashion, the initial state to which the self.circuit is applied.
        args
        kwargs

        Returns
        -------
        QubitWaveFunction:
            The wave function resulting from simulation.

        """
        simulator = pyquil.api.WavefunctionSimulator()
        n_qubits = self.n_qubits
        msb = BitString.from_int(initial_state, nbits=n_qubits)
        iprep = pyquil.Program()
        for i, val in enumerate(msb.array):
            if val > 0:
                iprep += pyquil.gates.X(i)
        backend_result = simulator.wavefunction(iprep + self.circuit, memory_map=self.resolver)
        return QubitWaveFunction.from_array(arr=backend_result.amplitudes, numbering=self.numbering)

    def do_sample(self, samples, circuit, *args, **kwargs) -> QubitWaveFunction:
        """
        Helper function, sampling an individual circuit.

        Parameters
        ----------
        samples: int:
            the number of samples of measurement to make.
        circuit:
            the circuit to sample.
        args
        kwargs

        Returns
        -------
        QubitWaveFunction:
            the result of sampled measurement, as a tequila wavefunction.
        """

        n_qubits = self.n_qubits
        p = circuit

        if self.device is None:
            qc = get_qc('{}q-qvm'.format(str(n_qubits)))
            p.wrap_in_numshots_loop(samples)
        else:
            qc = self.device
            p = qc.compile(p)
            p.attributes['num_shots'] = samples
        stacked = qc.run(p, memory_map=self.resolver)
        return self.convert_measurements(stacked)

    def convert_measurements(self, backend_result) -> QubitWaveFunction:
        """
        convert measurements from backend.

        Parameters
        ----------
        backend_result: list of ints:
            the result of measurement in pyquil.

        Returns
        -------
        QubitWaveFunction:
            measurement results translated into a QubitWaveFunction.
        """

        def string_to_array(s):
            listing = []
            for letter in s:
                if letter not in [',', ' ', '[', ']', '.']:
                    listing.append(int(letter))
            return listing

        result = QubitWaveFunction()
        bit_dict = {}
        for b in backend_result:
            try:
                bit_dict[str(b)] += 1
            except:
                bit_dict[str(b)] = 1

        for k, v in bit_dict.items():
            arr = string_to_array(k)
            result._state[BitString.from_array(arr)] = v
        return result

    def no_translation(self, abstract_circuit):
        return isinstance(abstract_circuit, pyquil.Program)

    def initialize_circuit(self, *args, **kwargs):
        """
        return an empty pyquil program.
        Parameters
        ----------
        args
        kwargs

        Returns
        -------
        pyquil.Program:
            an empty pyquil program.
        """
        return pyquil.Program()

    def add_parametrized_gate(self, gate, circuit, *args, **kwargs):
        """
       Add a parametrized gate to the circuit. Used in inherited method create_circuit.

       Additionally, builds and updates mappings so that pyquil can resolve the parametrization of the gate at runtime.
       Parameters
       ----------
       gate: QGateImpl:
           the gate to translate to pyquil.
       circuit:
           the pyquil circuit, to which a new gate is to be added
       args
       kwargs

       Returns
       -------
       None
       """
        op = self.op_lookup[gate.name]
        if isinstance(gate.parameter, float):
            par = gate.parameter
        else:
            try:
                par = self.match_par_to_dummy[gate.parameter]
            except:
                par = circuit.declare('theta_{}'.format(str(self.counter)), 'REAL')
                self.match_par_to_dummy[gate.parameter] = par
                self.counter += 1
        pyquil_gate = op(angle=par, qubit=self.qubit(gate.target[0]))
        if gate.is_controlled():
            for c in gate.control:
                pyquil_gate = pyquil_gate.controlled(self.qubit(c))
        circuit += pyquil_gate

    def add_measurement(self, circuit, target_qubits, *args, **kwargs):
        """
       Add a measurement to the circuit. Used in inherited method create_circuit.

       ----------
       gate: MeasurementGateImpl:
           the measurement, to be translated to pyquil
       circuit:
           the pyquil circuit, to which measurement is to be added
       args
       kwargs

       Returns
       -------
       None
       """
        bits = len(target_qubits)
        measurements = self.initialize_circuit()
        ro = measurements.declare('ro', 'BIT', bits)
        for i, t in enumerate(sorted(target_qubits)):
            measurements += pyquil.gates.MEASURE(self.qubit(t), ro[i])
        return circuit + measurements # avoid inplace operations

    def add_basic_gate(self, gate, circuit, *args, **kwargs):
        """
        Add an unparametrized gate to a circuit. Used in inherited method create_circuit.
        Parameters
        ----------
        gate: QGateImpl:
            the gate, to be translated to pyquil.
        circuit: pyquil.Program:
            the pyquil circuit, to which the gate is to be added
        args
        kwargs

        Returns
        -------
        None
        """
        op = self.op_lookup[gate.name]
        try:
            g = op[len(gate.control)]
            if gate.is_controlled():
                pyquil_gate = g(*[self.qubit(q) for q in gate.control + gate.target])
            else:
                pyquil_gate = g(*[self.qubit(t) for t in gate.target])
        except:
            g = op[0]
            for c in gate.control:
                pyquil_gate = g(*[self.qubit(t) for t in gate.target]).controlled(self.qubit(c))

        circuit += pyquil_gate

    def build_noisy_circuit(self, py_prog, noise_model):
        """
        Take a pyquil program, and add noise from a tequila NoiseModel.
        Parameters
        ----------
        py_prog: pyquil.Program:
            the program, to which noise should be added.
        noise_model: NoiseModel:
            the noise model, from whence noise should be added to the circuit.

        Returns
        -------
        pyquil.Program:
            A program, with noise added to it.
        """
        prog = py_prog
        new = pyquil.Program()
        collected = {}
        for noise in noise_model.noises:
            try:
                collected[str(noise.level)] = combine_kraus_maps(self.noise_lookup[noise.name](*noise.probs),
                                                                 collected[str(noise.level)])
            except:
                collected[str(noise.level)] = self.noise_lookup[noise.name](*noise.probs)
        done = []
        for gate in prog:
            new.inst(gate)
            if hasattr(gate, 'qubits'):
                level = str(len(gate.qubits))
                if level in collected.keys():
                    if name_dict[gate.name] == 'parametrized':
                        new.inst([pyquil.gates.I(q) for q in gate.qubits])
                        if ['parametrized', gate.qubits] not in done:
                            new.define_noisy_gate('I',
                                                  gate.qubits,
                                                  append_kraus_to_gate(collected[level], np.eye(2), int(level)))
                            done.append(['parametrized', 1, gate.qubits])

                    else:
                        if [gate.name, len(gate.qubits), gate.qubits] not in done:
                            k = unitary_maker(gate)
                            new.define_noisy_gate(gate.name,
                                                  gate.qubits,
                                                  append_kraus_to_gate(collected[level], k, int(level)))
                            done.append([gate.name, len(gate.qubits), gate.qubits])
                else:
                    pass
            else:
                pass
        return new

    def update_variables(self, variables):
        """
        Update the variables for resolution in simulation or sampling.
        Parameters
        ----------
        variables: dict:
            dictionary of tequila variables and values to resolve for simulation.

        Returns
        -------
        None
        """

        if self.match_dummy_to_value is not None:
            self.resolver = {k: [to_float(v(variables))] for k, v in self.match_dummy_to_value.items()}
        else:
            self.resolver = None

    def check_device(self, device):
        """
        Verify if a device is valid.
        Parameters
        ----------
        device:
            a pyquil.api.QuantumComputer, a string which picks one out, or a dictionary that can pass to get_qc.
        Returns
        -------
        None

        """
        if device is None:
            return
        if isinstance(device, str):
            d = device
            if '-qvm' in d.lower():
                d = d[:-4]
            if '-noisy' in d.lower():
                d = d[:-6]
            if d in pyquil.list_quantum_computers():
                return
            else:
                try:
                    get_qc(d)
                    return
                except:
                    try:
                        get_qc(d, as_qvm=True)
                        return
                    except:
                        raise TequilaException('could not obtain device from string; received {}'.format(device))

        elif isinstance(device, dict):
            try:
                get_qc(**device)
                return
            except:
                raise TequilaException('could not initialize device from dict; received {}'.format(device))
        elif isinstance(device, pyquil.api.QuantumComputer):
            return

        else:
            raise TequilaException(
                'Uninterpretable object {} of type {} passed to check_device!'.format(device, type(device)))

    def retrieve_device(self, device):
        """
        return an initialized pyquil quantum computer (or None)
        Parameters
        ----------
        device:
            pyquil.api.QuantumComputer, or arguments that can pass to pyquil.get_qc

        Returns
        -------
        pyquil.api.QuantumComputer
            an instantiated device object for pyquil simulation or execution.
        """
        use_device_noise = (self.noise == 'device')
        if device is None:
            return None
        if isinstance(device, str):
            try:
                back = get_qc(device, noisy=use_device_noise)
                return back
            except:
                try:
                    back = get_qc(device, as_qvm=True, noisy=use_device_noise)
                    return back
                except:
                    raise TequilaException('could not obtain device from string; received {}'.format(device))
        elif isinstance(device, pyquil.api.QuantumComputer):
            return device
        elif isinstance(device, dict):
            try:
                return get_qc(**device)
            except:
                raise TequilaException('could not initialize device from dict; received {}'.format(device))
        else:
            raise TequilaException(
                'Uninterpretable object {} of type {} passed to check_device!'.format(device, type(device)))


class BackendExpectationValuePyquil(BackendExpectationValue):
    """
    See BackendExpectationValue for information.
    """
    BackendCircuitType = BackendCircuitPyquil
