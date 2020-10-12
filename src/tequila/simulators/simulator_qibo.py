from tequila.simulators.simulator_base import QCircuit, BackendCircuit, BackendExpectationValue
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila import TequilaException
from tequila import BitString, BitNumbering, BitStringLSB
from tequila.utils.keymap import KeyMapRegisterToSubregister


import numpy as np
import qibo
from qibo.models import Circuit
from qibo import gates
import tensorflow as tf

class TequilaQiboException(TequilaException):
    def __str__(self):
        return "Error in qibo backend:" + self.message


def bit_flip_map(qs,p):
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
    back= []
    for i in qs:
        for mat in [mat1,mat2]:
            back.append(((i,), mat))

    return back


def phase_flip_map(qs,p):
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
    back= []
    for i in qs:
        for mat in [mat1,mat2]:
            back.append(((i,), mat))
    return back


def amp_damp_map(qs,p):
    """
    Generate the Kraus operators corresponding to an amplitude damping
    noise channel.

    :params float p: The one-step damping probability.
    :return: A list [k1, k2] of the Kraus operators that parametrize the map.
    :rtype: list
    """
    mat1 = np.sqrt(p) * np.array([[0, 1],
                                        [0, 0]])
    mat2 = np.diag([1, np.sqrt(1 - p)])
    back= []
    for i in qs:
        for mat in [mat1,mat2]:
            back.append(((i,), mat))
    return back


def phase_damp_map(qs,p):
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

    back = []
    for i in qs:
        for mat in [mat1, mat2]:
            back.append(((i,), mat))
    return back

def phase_amp_damp_map(qs, a, b):
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

    mats= [np.array(k) for k in [A0, A1, A2]]
    back = []
    for i in qs:
        for mat in mats:
            back.append(((i,), mat))
    return back

def depolarizing_map(qs,p):
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
    mats= [k for k in [mat1,mat2,mat3,mat4]]
    back = []
    for i in qs:
        for mat in mats:
            back.append(((i,), mat))
    return back

class BackendCircuitQibo(BackendCircuit):
    """
    Class representing circuits compiled to qibo.
    See BackendCircuit for documentation of features and methods inherited therefrom

    Attributes
    ----------
    counter:
        counts how many distinct sympy.Symbol objects are employed in the circuit.
    has_noise:
        whether or not the circuit is noisy. needed by the expectationvalue to do sampling properly.
    noise_lookup: dict:
        dict mapping strings to lists of constructors for cirq noise channel objects.
    op_lookup: dict:
        dictionary mapping strings (tequila gate names) to cirq.ops objects.
    variables: list:
        a list of the qulacs variables of the circuit.

    Methods
    -------
    add_noise_to_circuit:
        apply a tequila NoiseModel to a qulacs circuit, by translating the NoiseModel's instructions into noise gates.
    """

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
        "controlled_power": False,
        "controlled_phase": False,
        "toffoli": False,
        "phase_to_z": False,
        "cc_max": True
    }


    def __init__(self, abstract_circuit, noise=None, device=None, *args, **kwargs):
        """

        Parameters
        ----------
        abstract_circuit: QCircuit:
            the circuit to compile to qulacs
        noise: optional:
            noise to apply to the circuit.
        device: optional:
            a specifications for hardware for simulation.
        args
        kwargs
        """
        self.op_lookup = {
            'I': gates.I,
            'X': gates.X,
            'Y': gates.Y,
            'Z': gates.Z,
            'H': gates.H,
            'Rx': gates.RX,
            'Ry': gates.RY,
            'Rz': gates.RZ,
            'SWAP': gates.SWAP,
            'Measure': gates.M,
            'Phase': gates.ZPow
        }
        self.counter = 0  # keeps track of how many parameters are in the circuit
        self.variables = []  # will map position to parameter better
        self.inst_list = []  # gates cannot be retrieved from an initialized circuit; needed for noise.
        self.flag = False
        if noise is not None:
            qibo.set_backend("defaulteinsum")  # necessary for Qibo to do density matrices!
        else:
            qibo.set_backend('custom')

        super().__init__(abstract_circuit=abstract_circuit, noise=noise,device=device, *args, **kwargs)

        if noise is not None:
            # a lookup table from tequila QuantumNoise to NoiseChannel qibo objects. See each function
            # for reference
            self.noise_lookup = {
                'bit flip': lambda qs,px: gates.GeneralChannel(bit_flip_map(qs,px)),
                'phase flip': lambda qs,px: gates.GeneralChannel(phase_flip_map(qs,px)),
                'phase damp': lambda qs, p: gates.GeneralChannel(phase_damp_map(qs,p)),
                'amplitude damp': lambda qs, p: gates.GeneralChannel(amp_damp_map(qs,p)),
                'phase-amplitude damp': lambda qs,a,b: gates.GeneralChannel(phase_amp_damp_map(qs,a,b)),
                'depolarizing': lambda qs,p: gates.GeneralChannel(depolarizing_map(qs,p))
            }
            self.circuit = self.add_noise_to_circuit(noise) # see this function for details

    def check_device(self, device):
        assert type(device) in [str,dict] or device is None
        if isinstance(device,str):
            d=device.upper()
            if d[:5] not in ["/GPU:","/CPU:"]:
                raise TequilaQiboException("Device names must begin with either /GPU: or /CPU:; received {}".format(d))
        elif isinstance(device,dict):
            if 'memory_device' in device.keys(): #full spec paralellism
                md = device['memory_device'].upper()
                self.check_device(md)
                if 'accelerators' in list(device.keys()):
                    for k in device['accelerators'].keys():
                        self.check_device(k)
                else:
                    raise TequilaQiboException('device dictionary formatted improperly!')
            else:
                for k in device['accelerators'].keys():
                    self.check_device(k)
        return

    def retrieve_device(self, device):
        if device is None:
            return device
        elif isinstance(device,str):
            return device
        elif isinstance(device,dict):
            if 'memory_device' in device.keys():  # full spec paralellism
                return device
            else:
                return {'accelerators':device,'memory_device':"/CPU:0"}
        else:
            raise TequilaQiboException('Invalid device of type {}'.format(type(device)))

    def update_variables(self, variables):
        """
        set new variable values for the circuit.
        Parameters
        ----------
        variables: dict:
            the variables to supply to the circuit.

        Returns
        -------
        None
        """
        if variables is not None:
            loaded = []
            for v in self.variables:
                loaded.append(v(variables))
            self.circuit.set_parameters(loaded)

    def do_simulate(self, variables, initial_state=None, *args, **kwargs):
        """
        Helper function to perform simulation.

        Parameters
        ----------
        variables: dict:
            variables to supply to the circuit.
        initial_state:
            information indicating the initial state on which the circuit should act.
        args
        kwargs

        Returns
        -------
        QubitWaveFunction:
            QubitWaveFunction representing result of the simulation.
        """
        if initial_state is not None:
            state = np.zeros(2**self.n_qubits)
            if initial_state == 0:
                state[0] = 1
            elif initial_state == 1:
                state[-1] = 1
                result = self.circuit(state)
        else:
            result = self.circuit()
        back= QubitWaveFunction.from_array(arr=result.numpy())
        return back

    def convert_measurements(self, backend_result, target_qubits=None) -> QubitWaveFunction:
        """
        Transform backend evaluation results into QubitWaveFunction
        Parameters
        ----------
        backend_result:
            the return value of backend simulation.

        Returns
        -------
        QubitWaveFunction
            results transformed to tequila native QubitWaveFunction
        """

        result = QubitWaveFunction()
        # todo there are faster ways

        for k, v in backend_result.frequencies(binary=True).items():
            converted_key = BitString.from_bitstring(other=BitStringLSB.from_binary(binary=k))
            result._state[converted_key] = v


        if target_qubits is not None:
            mapped_target = [self.qubit_map[q].number for q in target_qubits]
            mapped_full = [self.qubit_map[q].number for q in self.abstract_qubits]
            keymap = KeyMapRegisterToSubregister(subregister=mapped_target, register=mapped_full)
            result = result.apply_keymap(keymap=keymap)

        return result

    def do_sample(self, samples, circuit, noise_model=None, initial_state=None, *args, **kwargs) -> QubitWaveFunction:
        """
        Helper function for performing sampling.

        Parameters
        ----------
        samples: int:
            the number of samples to be taken.
        circuit:
            the circuit to sample from.
        noise_model: optional:
            noise model to be applied to the circuit.
        initial_state:
            sampling supports initial states for qulacs. Indicates the initial state to which circuit is applied.
        args
        kwargs

        Returns
        -------
        QubitWaveFunction:
            the results of sampling, as a Qubit Wave Function.
        """

        if initial_state is not None:
            state = np.zeros(2**self.n_qubits)
            if initial_state == 0:
                state[0] = 1
            elif initial_state == 1:
                state[-1] = 1
            else:
                if isinstance(initial_state,np.ndarray):
                    state=initial_state
            result = self.circuit(state,nshots=samples)
        else:
            result = self.circuit(nshots=samples)

        return self.convert_measurements(backend_result=result)

    def no_translation(self, abstract_circuit):
        """
        Todo: what is this for?
        Parameters
        ----------
        abstract_circuit

        Returns
        -------

        """
        return False

    def initialize_circuit(self, *args, **kwargs):
        """
        return an empty circuit.
        Parameters
        ----------
        args
        kwargs

        Returns
        -------
        qibo.tensorflow.circuit.TensorflowCircuit
            an empty, though initialized, circuit that can be executed or manipulated.
        """
        if self.device is None:
            return Circuit(self.n_qubits)
        else:
            if isinstance(self.device, str):
                if not self.flag:
                    qibo.set_device(self.device)
                    self.flag = True  # don't reset the device every time; such as during measurement.
                return Circuit(self.n_qubits)
            elif isinstance(self.device, dict):
                acc = self.device['accelerators']
                mem = self.device['memory device']
                return Circuit(self.n_qubits, accelerators=acc, memory_device=mem)

    def add_parametrized_gate(self, gate, circuit, variables, *args, **kwargs):
        """
        add a parametrized gate.
        Parameters
        ----------
        gate: QGateImpl:
            the gate to add to the circuit.
        circuit:
            the circuit to which the gate is to be added
        variables:
            dict that tells values of variables
        args
        kwargs

        Returns
        -------
        None
        """
        op = self.op_lookup[gate.name]
        t = gate.target
        c = gate.control
        self.variables.append(gate.parameter)
        value = gate.parameter(variables)
        if gate.is_controlled():
            inst= op(t[0], theta=value).controlled_by(c[0])
        else:
            inst = op(t[0],theta=value)
        self.inst_list.append(inst)
        circuit.add(inst)

    def add_basic_gate(self, gate, circuit, *args, **kwargs):
        """
        add an unparametrized gate to the circuit.
        Parameters
        ----------
        gate: QGateImpl:
            the gate to be added to the circuit.
        circuit:
            the circuit, to which a gate is to be added.
        args
        kwargs

        Returns
        -------
        None
        """
        op = self.op_lookup[gate.name]
        t = gate.target
        c = gate.control
        if gate.is_controlled():
            inst= op(t[0]).controlled_by(c[0])
        else:
            if gate.name is ['Swap']:
                inst = op(t[0], t[1])
            else:
                inst = op(t[0])
        self.inst_list.append(inst)
        circuit.add(inst)

    def add_measurement(self, circuit, target_qubits, *args, **kwargs):
        """
        Add a measurement operation to a circuit.
        Parameters
        ----------
        circuit:
            a circuit, to which the measurement is to be added.
        target_qubits: List[int]
            abstract target qubits
        args
        kwargs

        Returns
        -------
        None
        """
        for t in target_qubits:
            self.inst_list.append(gates.M(t))
            circuit.add(gates.M(t))

    def add_noise_to_circuit(self,noise_model):
        """
        Apply noise from a NoiseModel to a circuit.
        Parameters
        ----------
        noise_model: NoiseModel:
            the noisemodel to apply to the circuit.

        Returns
        -------
        qibo.tensorflow.circuit.TensorflowCircuit
            self.circuit, with noise added on.
        """

        n = noise_model
        new=self.initialize_circuit()
        for g in self.inst_list:
            new.add(g)
            qubits=g.qubits
            for noise in n.noises:
                if len(qubits) == noise.level:
                    ch = (self.noise_lookup[noise.name])
                    chargs = [qubits]
                    for p in noise.probs:
                        chargs.append(p)
                    chan = ch(*chargs)
                    new.add(chan)
        return new


class BackendExpectationValueQibo(BackendExpectationValue):
    BackendCircuitType = BackendCircuitQibo
