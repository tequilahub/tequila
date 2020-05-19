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
    mat1 = np.array([[1, 0], [0, np.sqrt(1 - p)]])
    mat2 = np.array([[0, 0], [0, np.sqrt(p)]])
    return [mat1, mat2]


def bit_flip_map(p):
    mat1 = np.array([[np.sqrt(1 - p), 0], [0, np.sqrt(1 - p)]])
    mat2 = np.array([[0, np.sqrt(p)], [np.sqrt(p), 0]])
    return [mat1, mat2]


def phase_flip_map(p):
    mat1 = np.array([[np.sqrt(1 - p), 0], [0, np.sqrt(1 - p)]])
    mat2 = np.array([[np.sqrt(p), 0], [0, -np.sqrt(p)]])
    return [mat1, mat2]


def phase_amp_damp_map(a, b):
    A0 = [[1, 0], [0, np.sqrt(1 - a - b)]]
    A1 = [[0, np.sqrt(a)], [0, 0]]
    A2 = [[0, 0], [0, np.sqrt(b)]]
    return [np.array(k) for k in [A0, A1, A2]]


def depolarizing_map(p):
    mat1 = np.array([[np.sqrt(1 - 3 * p / 4), 0], [0, np.sqrt(1 - 3 * p / 4)]])
    mat2 = np.array([[np.sqrt(p / 4), 0], [0, -np.sqrt(p / 4)]])
    mat3 = np.array([[0, np.sqrt(p / 4)], [np.sqrt(p / 4), 0]])
    mat4 = np.array([[0., -1.j * np.sqrt(p / 4)], [1.j * np.sqrt(p / 4), .0]])
    return [mat1, mat2, mat3, mat4]




def kraus_tensor(klist, n):
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
    Follow a gate `g` by a Kraus map described by `kraus_ops`.

    :param list kraus_ops: The Kraus operators.
    :param numpy.ndarray g: The unitary gate.
    :return: A list of transformed Kraus operators.
    """

    return [kj.dot(g) for kj in kraus_tensor(kraus_ops, level)]


def add_controls(matrix, count):
    gc = np.log2(matrix.shape[0])
    controls = count - gc
    if int(controls) == 0:
        return matrix
    new = np.eye(2 ** count)
    new[-matrix.shape[0]:, -matrix.shape[0]:] = matrix
    return new


def unitary_maker(gate):
    return add_controls(name_unitary_dict[gate.name], len(gate.qubits))


class TequilaPyquilException(TequilaException):
    def __str__(self):
        return "simulator_pyquil: " + self.message

def check_pyquil_device(device):
    if isinstance(device,str):
        d=device
        if '-qvm' in device.lower():
            d=device[:-4]
        if d in pyquil.list_quantum_computers():
            return
        else:
            try:
                get_qc(d)
                return
            except:
                try:
                    get_qc(d,as_qvm=True)
                    return
                except:
                    raise TequilaException('could not obtain device from string!')

    elif isinstance(device,dict):
        try:
            get_qc(**device)
            return
        except:
            raise TequilaException('could not initialize device from dict!')
    elif isinstance(device, pyquil.api.QuantumComputer):
            return

    elif device is None:
        return
    else:
        raise TequilaException('Uninterpretable object type passed to check_pyquil_device!')

def retrieve_pyquil_device(device,use_device_noise=False):
    if isinstance(device,str):
        try:
            back=get_qc(device,noisy=use_device_noise)
            print(back)
            return back
        except:
            try:
                back= get_qc(device,as_qvm=True,noisy=use_device_noise)
                return back
            except:
                raise TequilaException('could not initialize device from string!')
    elif isinstance(device,pyquil.api.QuantumComputer):
        return device
    elif isinstance(device,dict):
        try:
            return get_qc(**device)
        except:
            raise TequilaException('could not initialize device from dict!')
    elif device is None:
        return None
    else:
        raise TequilaException('Uninterpretable object type passed to get_pyquil_device!')

class BackendCircuitPyquil(BackendCircuit):

    compiler_arguments = {
        "trotterized": True,
        "swap": False,
        "multitarget": True,
        "controlled_rotation": False,
        "gaussian": True,
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


    def __init__(self, abstract_circuit: QCircuit, variables, use_mapping=True, noise=None,device=None, *args, **kwargs):
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
        super().__init__(abstract_circuit=abstract_circuit, variables=variables, noise=noise,
                         use_mapping=use_mapping, *args, **kwargs)
        if self.noise is not None:
            self.noise_lookup = {
                'amplitude damp': amp_damp_map,
                'phase damp': phase_damp_map,
                'bit flip': bit_flip_map,
                'phase flip': phase_flip_map,
                'phase-amplitude damp': phase_amp_damp_map,
                'depolarizing': depolarizing_map
            }

            if isinstance(self.noise,str):
                if device is not None:
                    if self.noise == 'device':
                        self.device=retrieve_pyquil_device(device,True)
                    else:
                        raise TequilaException('noise was a string, but was not the string device. Not allowed!')
                else:
                    raise TequilaException('noise was a string, but no device provided'
                                        )
            else:
                self.device = retrieve_pyquil_device(device, False)
                self.circuit = self.build_noisy_circuit(self.circuit, self.noise)
        else:
            if device is not None:
                self.device=retrieve_pyquil_device(device,False)
            else:
                self.device = None

        if len(self.match_par_to_dummy.keys()) is None:
            self.match_dummy_to_value = None
            self.resolver = None
        else:
            self.match_dummy_to_value = {'theta_{}'.format(str(i)): k for i, k in
                                         enumerate(self.match_par_to_dummy.keys())}
            self.resolver = {k: [to_float(v(variables))] for k, v in self.match_dummy_to_value.items()}

    def do_simulate(self, variables, initial_state, *args, **kwargs):

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
        n_qubits = self.n_qubits

        p = circuit

        if self.device is None:
            qc = get_qc('{}q-qvm'.format(str(n_qubits)))
            p.wrap_in_numshots_loop(samples)
        else:
            qc=self.device
            p=qc.compile(p)
            p.attributes['num_shots']=samples
        stacked = qc.run(p, memory_map=self.resolver)
        return self.convert_measurements(stacked)

    def convert_measurements(self, backend_result) -> QubitWaveFunction:
        """0.
        :param backend_result: array from pyquil as list of lists of integers.
        :return: backend_result in Tequila format.
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

    def fast_return(self, abstract_circuit):
        return isinstance(abstract_circuit, pyquil.Program)

    def initialize_circuit(self, *args, **kwargs):
        return pyquil.Program()

    def add_parametrized_gate(self, gate, circuit, *args, **kwargs):
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
        pyquil_gate = op(angle=par, qubit=self.qubit_map[gate.target[0]])
        if gate.is_controlled():
            for c in gate.control:
                pyquil_gate = pyquil_gate.controlled(self.qubit_map[c])
        circuit += pyquil_gate

    def add_measurement(self, gate, circuit, *args, **kwargs):
        bits = len(gate.target)
        ro = circuit.declare('ro', 'BIT', bits)
        for i, t in enumerate(gate.target):
            circuit += pyquil.gates.MEASURE(self.qubit_map[t], ro[i])

    def add_basic_gate(self, gate, circuit, *args, **kwargs):
        op = self.op_lookup[gate.name]
        try:
            g = op[len(gate.control)]
            if gate.is_controlled():
                pyquil_gate = g(*[self.qubit_map[q] for q in gate.control + gate.target])
            else:
                pyquil_gate = g(*[self.qubit_map[t] for t in gate.target])
        except:
            g = op[0]
            for c in gate.control:
                pyquil_gate = g(*[self.qubit_map[t] for t in gate.target]).controlled(self.qubit_map[c])

        circuit += pyquil_gate

    def build_noisy_circuit(self, py_prog, noise_model):
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
        overwriting the underlying code so that noise gets added when present
        """

        if self.match_dummy_to_value is not None:
            self.resolver = {k: [to_float(v(variables))] for k, v in self.match_dummy_to_value.items()}
        else:
            self.resolver = None


class BackendExpectationValuePyquil(BackendExpectationValue):
    BackendCircuitType = BackendCircuitPyquil
