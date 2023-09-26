from tequila.simulators.simulator_base import QCircuit, BackendCircuit, BackendExpectationValue
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila import TequilaException
from tequila import BitString, BitNumbering
import sympy
from tequila.utils import to_float

import importlib
import numpy as np
import typing, numbers

import cirq
import cirq_google

map_1 = lambda x: {'exponent': x}
map_2 = lambda x: {'exponent': x / np.pi, 'global_shift': -0.5}


def qubit_satisfier(op, level):
    """
    check if a given operation acts on a certain number of qubits
    Parameters
    ----------
    op:
        the cirq operation
    level:
        the number of qubits in question

    Returns
    -------
    bool:
        whether or not the number of qubits in op is equal to level
    """
    oplen = len(op.qubits)
    return oplen == level


class TequilaCirqException(TequilaException):
    def __str__(self):
        return "Error in cirq backend:" + self.message


class BackendCircuitCirq(BackendCircuit):
    """
    Class for circuits compiled to be executed by Cirq.
    See documentation for BackendCircuit for methods and attributes not listed here.

    Attributes
    ----------
    counter:
        counts how many distinct sympy.Symbol objects are employed in the circuit.
    noise_lookup: dict:
        dict mapping strings to lists of constructors for cirq noise channel objects.
    op_lookup: dict:
        dictionary mapping strings (tequila gate names) to cirq.ops objects.
    resolver:
        cirq ParamResolver object; assigns values to parameters at runtime.
    sympy_to_tq_: dict:
        dictionary mapping sympy.Symbols back to tequila Variables and Objectives.
    tq_to_sympy: dict:
        dictionary mapping tequila Variables and Objectives to sympy.Symbols, for parameter resolution.

    Methods
    -------
    build_device_circuit:
        fit the cirq circuit to a specific device, by compiling its gates and matching its qubits to those the device
        supports.
    build_noisy_circuit:
        apply a tequila NoiseModel to a cirq circuit, by translating the NoiseModel's instructions into noise channels.

    """

    compiler_arguments = {
        "trotterized": True,
        "swap": False,
        "multitarget": True,
        "controlled_rotation": False,
        "generalized_rotation": True,
        "exponential_pauli": True,
        "controlled_exponential_pauli": True,
        "phase": True,
        "power": False,
        "hadamard_power": False,
        "controlled_power": False,
        "controlled_phase": True,
        "toffoli": False,
        "phase_to_z": False,
        "cc_max": False
    }

    numbering: BitNumbering = BitNumbering.MSB

    def __init__(self, abstract_circuit: QCircuit, variables, qubit_map=None, noise=None, device=None, *args, **kwargs):
        """

        Parameters
        ----------
        abstract_circuit: QCircuit:
            Tequila unitary to compile to cirq
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

        self.tq_to_sympy = {}
        self.counter = 0
        if device is not None:
            self.compiler_arguments['cc_max'] = True
        super().__init__(abstract_circuit=abstract_circuit, variables=variables,
                         noise=noise, qubit_map=qubit_map, device=device, *args, **kwargs)
        if len(self.tq_to_sympy.keys()) is None:
            self.sympy_to_tq = None
            self.resolver = None
        else:
            self.sympy_to_tq = {v: k for k, v in self.tq_to_sympy.items()}
            self.resolver = cirq.ParamResolver({k: v(variables) for k, v in self.sympy_to_tq.items()})
        if self.device is not None:
            self.circuit = self.build_device_circuit()
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
            self.circuit = self.build_noisy_circuit(self.noise)

    def do_simulate(self, variables, initial_state=0, *args, **kwargs) -> QubitWaveFunction:
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
        simulator = cirq.Simulator()
        backend_result = simulator.simulate(program=self.circuit, param_resolver=self.resolver,
                                            initial_state=initial_state)
        return QubitWaveFunction.from_array(arr=backend_result.final_state_vector, numbering=self.numbering)

    def convert_measurements(self, backend_result: cirq.Result) -> QubitWaveFunction:
        """
        Take the results of a cirq measurement and translate them to teuqila QubitWaveFunction.
        Parameters
        ----------
        backend_result: cirq.Result:
            the result of sampled measurements.

        Returns
        -------
        QubitWaveFunction:
            the result of sampling, as a tequila QubitWavefunction.

        """
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
        return self.convert_measurements(cirq.sample(program=circuit, param_resolver=self.resolver, repetitions=samples))

    def no_translation(self, abstract_circuit):
        return isinstance(abstract_circuit, cirq.Circuit)

    def initialize_circuit(self, *args, **kwargs):
        """
        Return an empty cirq Circuit.
        Parameters
        ----------
        args
        kwargs

        Returns
        -------
        cirq.Circuit
        """
        return cirq.Circuit()

    def add_parametrized_gate(self, gate, circuit, *args, **kwargs):
        """
        Add a parametrized gate to the circuit. Used in inherited method create_circuit.

        Additionally, builds and updates mappings so that cirq can resolve the parametrization of the gate at runtime.
        Parameters
        ----------
        gate: QGateImpl:
            the gate to translate to cirq.
        circuit:
            the cirq circuit, to which a new gate is to be added
        args
        kwargs

        Returns
        -------
        None
        """
        op, mapping = self.op_lookup[gate.name]
        parameter = gate.parameter
        if hasattr(gate, 'power'):
            parameter = gate.power
        if isinstance(parameter, float):
            par = parameter
        else:
            try:
                par = self.tq_to_sympy[parameter]
            except:
                par = sympy.Symbol('{}_{}'.format(self._name_variable_objective(parameter), str(self.counter)))
                self.tq_to_sympy[parameter] = par
                self.counter += 1
        cirq_gate = op(**mapping(par)).on(*[self.qubit(t) for t in gate.target])
        if gate.is_controlled():
            cirq_gate = cirq_gate.controlled_by(*[self.qubit(c) for c in gate.control])
        circuit.append(cirq_gate)

    def add_basic_gate(self, gate, circuit, *args, **kwargs):
        """
        Adds an unparametrized gate to a circuit.

        Parameters
        ----------
        gate: QGateImpl:
            the gate, to be added to the circuit.
        circuit: cirq.Circuit:
            the circuit, to which a gate is to be added.
        args
        kwargs

        Returns
        -------
        None
        """
        op, mapping = self.op_lookup[gate.name]
        cirq_gate = op().on(*[self.qubit(t) for t in gate.target])
        if gate.is_controlled():
            cirq_gate = cirq_gate.controlled_by(*[self.qubit(c) for c in gate.control])
        circuit.append(cirq_gate)

    def add_measurement(self, circuit, target_qubits, *args, **kwargs):
        """
        Adds a measurement operation to a cirq circuit.
        Parameters
        ----------
        circuit: cirq.Circuit:
            a cirq circuit, to add measurement to.
        target_qubits: list[int]:
            abstract target qubits
        args
        kwargs

        Returns
        -------
        None
        """
        target_qubits = sorted(target_qubits)
        cirq_gate = cirq.MeasurementGate(len(target_qubits)).on(*[self.qubit(t) for t in target_qubits])
        return circuit + cirq_gate # avoid inplace operations for measurements

    def make_qubit_map(self, qubits) -> typing.Dict[numbers.Integral, cirq.LineQubit]:
        """
        Map integers to cirq.Linequbits
        Parameters
        ----------
        qubits:
            the list of qubits.
        Returns
        -------
        dict:
            a qubit mapping lookup table.
        """
        qubit_map = super().make_qubit_map(qubits=qubits)

        # check if cirq qubits were already given from above
        # we're checking only if the instance is not abstract (same as number)
        if all(v.instance != v.number for v in qubit_map.values()):
            return qubit_map

        # initialize cirq_qubits
        for k, v in qubit_map.items():
            qubit_map[k].instance = cirq.LineQubit(qubit_map[k].number)
        return qubit_map

    def build_device_circuit(self, ignore_failures=False):
        """
        Attempts to configure a cirq circuit to run on a device
        Parameters
        ----------
        ignore_failures: bool:
            whether or not to include gates in the circuit that fail to compile. Ignore; currently under construction.
        Returns
        -------
        cirq.Circuit
            the circuit, reconfigured for the device.

        """
        c = self.circuit
        device = self.device
        line = None
        circuit = None
        if isinstance(device, cirq.Device):
            HAS_GOOGLE = importlib.util.find_spec('cirq_google')
            assert HAS_GOOGLE, TequilaCirqException(' cirq_google package is not installed.')
            
            if device in [cirq_google.Sycamore, cirq_google.Sycamore23]:
                try:
                    circuit = cirq.optimize_for_target_gateset(circuit=c, gateset=cirq_google.SycamoreTargetGateset())
                except ValueError as E:
                    original_message = str(E)
                    raise TequilaCirqException('original message:\n{}\n\ncould not optimize for device={}'.format(original_message,device))
            else:
                ### under construction (potentially on other branches)
                raise TequilaException('Only Sycamore and Sycamore23 devices currently functional. Sorry!')

        else:
            raise TequilaException(
                'build_device_circuit demands a cirq.Device object; received {}, of type {}'.format(str(device),
                                                                                                    type(device)))
        return circuit

    def build_noisy_circuit(self, noise):
        """

        Parameters
        ----------
        noise: NoiseModel:
            the NoiseModel, which supplies instructions for noising a circuit.
        Returns
        -------
        cirq.Circuit
            self.circuit, with noise applied thereto.

        """
        c = self.circuit
        n = noise
        new_ops = []
        for op in c.all_operations():
            new_ops.append(op)
            for noise in n.noises:
                if qubit_satisfier(op, noise.level):
                    for i, channel in enumerate(self.noise_lookup[noise.name]):
                        new_ops.append(channel(noise.probs[i]).on_each([q for q in op.qubits]))
        return cirq.Circuit(*new_ops)

    def update_variables(self, variables):
        """
        Update the variables of the circuit by modifying the cirq.ParameterResolver sent to simulator at runtime.
        Parameters
        ----------
        variables: dict:
            dictionary assigning values to the parameters of the circuit.

        Returns
        -------
        None

        """
        # this is here because cirq cant take numpy arrays correctly
        if isinstance(variables, dict):
            variables = {k: to_float(v) for k, v in variables.items()}

        if self.sympy_to_tq is not None:
            self.resolver = cirq.ParamResolver({k: v(variables) for k, v in self.sympy_to_tq.items()})
        else:
            self.resolver = None

    def retrieve_device(self, device):
        """
        Retrieve a cirq.Device object for circuit execution (emulated).

        Parameters
        ----------
        device:
            a cirq.Device or string pointing to a named cirq device.
        Returns
        -------
        cirq.Device or None:
            the device on which to execute cirq circuits.
        """
        if isinstance(device, str):
            return getattr(cirq_google, device)
        else:
            if device is None:
                return device
            if isinstance(device, cirq.Device):
                return device
            else:
                raise TequilaException('Unable to retrieve requested device, {}, in cirq'.format(str(device)))

    def check_device(self, device):
        """
        Verify if a device is valid.
        Parameters
        ----------
        device:
            a cirq.Device or the name of a known cirq device.
        Returns
        -------
        None

        """
        if device is None:
            return
        if isinstance(device, cirq.Device):
            return
        else:
            assert isinstance(device, str)
            if device.lower() in ['sycamore', 'sycamore23']:
                pass
            else:
                raise TequilaException('requested device {} could not be found!'.format(device))


class BackendExpectationValueCirq(BackendExpectationValue):
    """
    See BackendExpectationValue for details.
    """
    BackendCircuitType = BackendCircuitCirq
