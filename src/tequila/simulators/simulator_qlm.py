import typing
from dataclasses import dataclass

import numpy as np
from qat.lang.AQASM import (PH, RX, RY, RZ, SWAP, AbstractGate, H, I, Program,
                            QRoutine, X, Y, Z, build_gate)
from qat.lang.AQASM.qint import QInt
from tequila.circuit._gates_impl import QGateImpl
from tequila.circuit.circuit import QCircuit
from tequila.simulators.simulator_base import (BackendCircuit,
                                               BackendExpectationValue)
from tequila.utils.bitstrings import BitNumbering, BitString
from tequila.utils.exceptions import TequilaException
from tequila.utils.misc import to_float
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction


MY_QLM = True
try:
    from qat.qpus import LinAlg
    MY_QLM = False
except ImportError:
    from qat.qpus import PyLinalg


def get_statevector(result) -> np.ndarray:
    """
    Creates a statevector out of a simulation result.

    Used when the simulator does not give a statevector.

    Parameters
    ----------
    result:
        the result from the simulation.

    Returns
    -------
    ndarray:
        the statevector corresponding to the result.
    """
    if result.statevector:
        return result.statevector
    qubits = result[0].qregs[0].length
    statevector = np.zeros((2 ** qubits,), dtype=complex)
    for sample in result:
        statevector[sample._state] = sample.amplitude
    return statevector


def initialize_state(initial_state: int, n_qubits: int):
    """
    Creates a custom gate for setting the classical initial state before computation.

    Parameters
    ----------
    initial_state:
        the initial state as an integer.
    q_qubits:
        how many qubits the gate effects. For intended use all qubits.
    Returns:
    ----------
        an abstract gate that uses X gates to create the wanted initial state.
    """
    @build_gate("INIT", [])
    def initialize():
        routine = QRoutine()
        qints = routine.new_wires(n_qubits, QInt)
        qints.set_value(initial_state)
        return routine
    return initialize


@dataclass
class RawCircuit:
    """
    A helper class to use the qlm program object more like circuits of other simulators.
    """
    program: str
    qubits: float
    measure: typing.List[int] = None

    def to_circ(self, link):
        return self.program.to_circ(link=link)

    def apply(self, gate, *args):
        self.program.apply(gate, *[self.qubits[arg] for arg in args])

    def new_var(self, name):
        return self.program.new_var(float, name)


class TequilaQLMException(TequilaException):
    def __str__(self):
        return "Error in MyQLM backend: " + self.message


class BackendCircuitQLM(BackendCircuit):
    """
    A class representing circuits that can be run on QLM simulators.

    See BackendCircuit for documentation on inherited attributes and methods.

    Attributes
    ----------
    numbering:
        tequila object for qubit order resolution.
    op_lookup: dict:
        dictionary mapping strings (tequila gate names) to PyAQASM gates.
    counter:
        counter used to make sure that all circuit variables have different names.
    tq_to_pars:
        dictionary for mapping tequila Variables to QLM variables.
    pars_to_tq:
        dictionary for mapping QLM variables to tequila Variables.
    pars_for_job:
        dictionary containing variable names as keys and their current values as values.
    qpu:
        a simulator from QLM that is to be used (if none PyLinalg(MyQLM) or LinAlg(QLM) is used).
    map_exact_qubits:
        if the cubits should be mapped exactly (possiblt leavin unused qubits in the simulation).

    Methods
    -------
    """
    compiler_arguments = {
        "multitarget": True,
        "multicontrol": False,
        "trotterized": True,
        "generalized_rotation": True,
        "exponential_pauli": True,
        "controlled_exponential_pauli": True,
        "hadamard_power": True,
        "controlled_power": True,
        "power": True,
        "toffoli": False,
        "controlled_phase": False,
        "phase": False,
        "phase_to_z": False,
        "controlled_rotation": False,
        "swap": False,
        "cc_max": False,
        "ry_gate": False,
        "y_gate": False,
        "ch_gate": False
    }

    numbering = BitNumbering.MSB

    def __init__(self, abstract_circuit: QCircuit, variables, noise=None, device=None,
                 qubit_map=None, optimize_circuit=True, qpu=None, map_exact_qubits=False,
                 *args, **kwargs):
        """
        Parameters
        ----------
        abstract_circuit: QCircuit:
            the circuit to be compiled for QLM.
        variables: dict:
            values for the variables of abstract_circuit.
        noise: optional:
            not supported.
        device: optiona:
            not supported.
        qubit_map: optional:
            a qubit map which maps the abstract qubits in the abstract_circuit to the qubits on the backend
            there is no need to initialize the corresponding backend types
            the dictionary should simply be {int:int} (preferred) or {int:name}
            if None the default will map to qubits 0 ... n_qubits -1 in the backend.
        optimize_circuit: optional: bool:
            currently does nothing.
        qpu: optional:
            a simulator from QLM that is to be used (if none PyLinalg(MyQLM) or LinAlg(QLM) is used).
        map_exact_qubits: optional:
            if the cubits should be mapped exactly (possiblt leavin unused qubits in the simulation).
        args
        kwargs
        """

        self.op_lookup = {
            'I': I,
            'X': X,
            'Y': Y,
            'Z': Z,
            'H': H,
            'Rx': RX,
            'Ry': RY,
            'Rz': RZ,
            'Phase': PH,
            'SWAP': SWAP
        }

        self.tq_to_pars = {}
        self.pars_for_job = {}
        self.counter = 0
        self.qpu = qpu
        self.map_exact_qubits = map_exact_qubits

        if noise:
            raise TequilaQLMException(
                "Tequila noise not supported. Noise can be used by giving a QPU from QLM as paramaetr 'qpu'."
            )
        if device:
            raise TequilaQLMException(
                "Use of device not suppoted. Device can be imitated by giving a QPu from QLM as parameter 'qpu'."
            )

        super().__init__(abstract_circuit=abstract_circuit, variables=variables, noise=noise, device=device,
                         qubit_map=qubit_map, optimize_circuit=optimize_circuit, *args, **kwargs)

        if len(self.tq_to_pars.keys()) is None:
            self.pars_to_tq = None
            self.pars_for_job = None
        else:
            self.pars_to_tq = {v: k for k, v in self.tq_to_pars.items()}
            self.pars_for_job = {next(iter(k.get_variables())): to_float(v(variables)) for k, v in self.pars_to_tq.items()}
            print(self.pars_for_job)

    def initialize_circuit(self, *args, **kwargs) -> RawCircuit:
        """
        returns an empy RawCircuit object representing an empty QLM program.

        Parameters
        ----------
        args
        kwargs

        Returns
        -------
        RawCircuit:
            an empy RawCircuit object representing an empty QLM program.
        """
        program = Program()
        qubits = program.qalloc(self.n_qubits)
        initializer = AbstractGate("INIT", [])
        program.apply(initializer(), qubits)
        return RawCircuit(program=program, qubits=qubits)

    def add_parametrized_gate(self, gate: QGateImpl, circuit: RawCircuit, *args, **kwargs):
        """
        add a parametrized gate to a circuit.

        Parameters
        ----------
        gate: QGateImpl:
            the  gate to apply to the circuit.
        circuit: RawCircuit:
            the circuit, to apply the gate to.
        args
        kwargsw

        Returns
        -------
        None

        """
        operator = self.op_lookup[gate.name]
        if gate.extract_variables():
            try:
                parameter = self.tq_to_pars[gate.parameter]
            except KeyError:
                parameter_name = f'{self._name_variable_objective(gate.parameter)}_{self.counter}'.replace(" ", "$")
                parameter = circuit.new_var(parameter_name)
                self.tq_to_pars[gate.parameter] = parameter
                self.counter += 1
        else:
            parameter = float(gate.parameter)
        operator = operator(parameter)
        if gate.is_controlled():
            for _ in gate.control:
                operator = operator.ctrl()
        target = self.qubit_map[gate.target[0]].number
        control = [self.qubit_map[qubit].number for qubit in gate.control]
        circuit.apply(operator, *control, target)

    def add_basic_gate(self, gate: QGateImpl, circuit: RawCircuit, *args, **kwargs):
        """
        add an unparametrized gate to a circuit.

        Parameters
        ----------
        gate: QGateImpl:
            the  gate to apply to the circuit.
        circuit: RawCircuit:
            the circuit, to apply the gate to.
        args
        kwargs

        Returns
        -------
        None

        """
        operator = self.op_lookup[gate.name]
        if gate.is_controlled():
            for _ in gate.control:
                operator = operator.ctrl()
        target = self.qubit_map[gate.target[0]].number
        control = [self.qubit_map[qubit].number for qubit in gate.control]
        circuit.apply(operator, *control, target)

    def add_measurement(self, circuit: RawCircuit, target_qubits, *args, **kwargs):
        """
        add a measurement to a circuit.

        Parameters
        ----------
        circuit: RawCircuit:
            the circuit, to apply measurement to.

        args
        kwargs

        Returns
        -------
        None

        """
        circuit.measure = target_qubits
        return circuit

    def do_sample(self, samples, circuit: RawCircuit, noise=None, abstract_qubits=None,
                  initial_state=0, *args, **kwargs) -> QubitWaveFunction:
        """
        Helper function for performing sampling.

        Parameters
        ----------
        circuit: RawCircuit:
            the circuit from which to sample.
        samples:
            the number of samples to take.
        noise:
            not supported.
        abstract_qubits:
            list qubit indexes to be measured.
        initial_state: int:
            integer indicating the starting state for self.circuit.
        args
        kwargs

        Returns
        -------
        QubitWaveFunction:
            the result of sampling.
        """
        if not isinstance(initial_state, int):
            raise TequilaQLMException("Initial state has to be given as an integer")
        if noise:
            raise TequilaQLMException(
                "Tequila noise not supported. Noise can be used by giving a QPU from QLM as paramaetr 'qpu'."
            )
        if abstract_qubits is None:
            abstract_qubits = circuit.measure
        abstract_qubits = sorted(list(abstract_qubits))
        abstract_qubits = [self.qubit_map[qubit].number for qubit in abstract_qubits]

        real_circuit = circuit.to_circ(link=[initialize_state(initial_state, self.n_qubits)])
        job = real_circuit.to_job(qubits=abstract_qubits, nbshots=samples)
        job = job(**self.pars_for_job)

        if self.qpu:
            return self.convert_measurements(self.qpu.submit(job))

        if MY_QLM:
            return self.convert_measurements(PyLinalg().submit(job))

        return self.convert_measurements(LinAlg().submit(job))

    def do_simulate(self, variables, initial_state=0, *args, **kwargs) -> QubitWaveFunction:
        """
        Helper function for performing simulation.

        Parameters
        ----------
        variables:
            variables to pass to the circuit for simulation.
        initial_state: int:
            integer indicating the starting state for self.circuit.
        args
        kwargs

        Returns
        -------
        QubitWaveFunction:
            the result of simulation.
        """
        if not isinstance(initial_state, int):
            raise TequilaQLMException("Initial state has to be given as an integer")
        self.update_variables(variables=variables)

        circuit = self.circuit.to_circ(link=[initialize_state(initial_state, self.n_qubits)])
        job = circuit.to_job()
        job = job(**self.pars_for_job)

        if MY_QLM:
            result = PyLinalg().submit(job)
            statevector = get_statevector(result)
            return QubitWaveFunction.from_array(arr=statevector, numbering=self.numbering)

        result = LinAlg().submit(job)
        return QubitWaveFunction.from_array(arr=result.statevector, numbering=self.numbering)

    def update_variables(self, variables):
        """
        Update circuit variables for use in simulation or sampling.

        Parameters
        ----------
        variables:
             a new set of variables for use in the circuit.

        Returns
        -------
        None
        """
        if self.pars_to_tq is not None:
            self.pars_for_job = {next(iter(k.get_variables())): to_float(v(variables)) for k, v in self.pars_to_tq.items()}
        else:
            self.pars_for_job = None

    def convert_measurements(self, backend_result) -> QubitWaveFunction:
        """
        map backend results to QubitWaveFunction

        Parameters
        ----------
        backend_result:
            the result returned directly from a QLM simulation.
        Returns
        -------
        QubitWaveFunction:
            measurements converted into wave function form.
        """
        result = QubitWaveFunction()
        shots = int(backend_result.meta_data["nbshots"])
        nbits = backend_result[0].qregs[0].length
        for sample in backend_result:
            converted_key = BitString.from_bitstring(other=BitString.from_int(integer=sample._state, nbits=nbits))
            result._state[converted_key] = round(sample.probability * shots)
        return result

    def make_qubit_map(self, qubits: dict):
        """
        Build the mapping between abstract qubits.

        Parameters
        ----------
        qubits:
            the qubits to map onto.
            If given as a dictionary, the map is already defined
            If given as a list the map will be those qubits mapped to 0 .... n_qubit-1 of the backend
        Returns
        -------
        Dict
            the dictionary that maps the qubits of the abstract circuits to an ordered sequence of integers.
            keys are the abstract qubit integers values are the backend qubits
            those are data structures which contain name and instance
            where number is the qubit identifier and instance the instance of the backend qubit
            if the backend does not require a special object for qubits the instance should be the same as number
        """
        if self.map_exact_qubits:
            qubits = range(self._input_args["abstract_circuit"].n_qubits)
        return super().make_qubit_map(qubits)


class BackendExpectationValueQLM(BackendExpectationValue):
    BackendCircuitType = BackendCircuitQLM
