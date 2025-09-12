import cudaq
import numpy as np
from cudaq import spin
import numbers
import numpy
from tequila import TequilaException
from tequila.utils.bitstrings import BitNumbering, reverse_int_bits
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila.simulators.simulator_base import BackendCircuit, BackendExpectationValue
from typing import Union, Optional

"""
Developer Note:
    Cudaq does not have objects for circuits and gates. Instead it uses quantum kernels. These only allow the input of
    arguments of certain types (and are therefore quite picky), i.e. ints, floats or a list[int], list[float].
    Due to this restriction the circuit is a list of these primitive types and the used gates are encoded.
"""


class TequilaCudaqException(TequilaException):
    def __str__(self):
        return "Error in cudaq (cuda-quantum) backend:" + self.message


class BackendCircuitCudaq(BackendCircuit):
    """
    Class representing circuits compiled to cudaq (cuda-quantum of NVIDIA).
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
        "swap": True,
        "multitarget": True,
        "controlled_rotation": True,  # needed for gates depending on variables
        "generalized_rotation": True,
        "exponential_pauli": True,
        "controlled_exponential_pauli": True,
        "phase": True,
        "power": True,
        "hadamard_power": True,
        "controlled_power": True,
        "controlled_phase": True,
        "toffoli": True,
        "phase_to_z": True,
        "cc_max": True,
    }
    numbering = BitNumbering.LSB
    supports_generic_initialization = True

    def __init__(self, abstract_circuit, noise=None, *args, **kwargs):
        """

        Parameters
        ----------
        abstract_circuit: QCircuit:
            the circuit to compile to cudaq
        noise: optional:
            noise to apply to the circuit.
        args
        kwargs
        """

        # gates encodings
        self.op_lookup = {
            # primitives
            "I": None,
            "X": 1,
            "Y": 2,
            "Z": 3,
            "H": 4,
            # simple 1-qubit rotations
            "Rx": 5,
            "Ry": 6,
            "Rz": 7,
            # phase gates
            "S": 8,
            "T": 9,
            # controlled primitives
            "CNOT": 10,
            "Cx": 11,
            "Cy": 12,
            "Cz": 13,
            # controlled rotations
            "CRx": 14,
            "CRy": 15,
            "CRz": 16,
            # other types of gates
            "SWAP": 17,
            "Measure": 18,
            "Exp-Pauli": None,
        }

        # instantiate a cudaq circuit as a list
        self.circuit = self.initialize_circuit()
        self.measurements = None
        self.variables = []
        super().__init__(abstract_circuit=abstract_circuit, noise=noise, *args, **kwargs)
        self.has_noise = False

    @cudaq.kernel
    def state_modifier_from_initial_state(  ## create a state based on an EXISTING PREVIOUS STATE i.e. |1010010>
        number_of_qubits: int,
        gate_encodings: list[int],
        target_qubits: list[int],
        angles: list[float],
        control_qubits: list[int],
        iteration_length: int,
        inital_state: cudaq.State,
    ):
        """
        This function applies a circuit to an EXISTING QUANTUM STATE with a given number of qubits,
        i.e. |10110>
        - unlike "state_modifier" this function needs a special preperation of the gates to apply to the state
        and therefore works with prepare_state_from_integer

        The function collects the gates to apply, which are stored in a list (cudaq's circuit "object")
        applies them in the given order to the state.

        These circuits support:
            1. single-qubit gates like: X, Y, Z, H, S, T as well as controlled variantes of them with up
            to one control-qubit
            2. parametrized single qubit gates like: Rx, Ry, Rz with a given angle
        """

        # create a quantum state based on a given initial state
        s = cudaq.qvector(inital_state)

        for index in range(iteration_length):
            encoding = gate_encodings[index]
            target = target_qubits[index]
            angle = angles[index]
            control = control_qubits[index]

            # x gate
            if encoding == 1:
                if control != -1:
                    x.ctrl(s[control], s[target])
                else:
                    x(s[target])
            # y gate
            elif encoding == 2:
                if control != -1:
                    y.ctrl(s[control], s[target])
                else:
                    y(s[target])
            # z gate
            elif encoding == 3:
                if control != -1:
                    z.ctrl(s[control], s[target])
                else:
                    z(s[target])
            # h gate
            elif encoding == 4:
                if control != -1:
                    h.ctrl(s[control], s[target])
                else:
                    h(s[target])
            # Rx gate
            elif encoding == 5:
                if control != -1:
                    pass
                else:
                    rx(angle, s[target])
            # Rx gate
            elif encoding == 6:
                if control != -1:
                    pass
                else:
                    ry(angle, s[target])
            # Rx gate
            elif encoding == 7:
                if control != -1:
                    pass
                else:
                    rz(angle, s[target])
            # S gate
            elif encoding == 8:
                if control != -1:
                    s.ctrl(s[control], s[target])
                else:
                    s(s[target])
            # T gate
            elif encoding == 9:
                if control != -1:
                    t.ctrl(s[control], s[target])
                else:
                    t(s[target])

    def prepare_circuit_for_state_modifier(self):
        """
        this function decomposes the circuit elements for later usage in state_modifier, which uses the
        cudaq annotation @cudaq.kernel.

        Has to be done this way since other functions for expectation value for example like cudaq.observe have special syntax
        and accept parameters and the state modifier itself
        """

        # prepare parameters for usage in kernal since "self. " access doesnt work within kernels
        number_of_qubits = self.n_qubits

        circuit = None
        if isinstance(self, BackendCircuitCudaq):
            circuit = self.circuit
        elif isinstance(self, BackendExpectationValueCudaq):
            circuit = self.U.circuit

        if circuit is None:
            raise ValueError("wrong attribute access in function - prepare_circuit_for_state_modifier")

        # get single lists from dict
        gate_encodings = circuit["gate_encodings"]
        target_qubits = circuit["target_qubits"]
        # Cast from FixedVariable to float to avoid CudaQ MLIR type conversion problems
        angles = [float(angle) for angle in circuit["angles"]]
        control_qubits = circuit["control_qubits"]

        iteration_length = None

        if len(gate_encodings) == len(target_qubits) == len(angles) == len(control_qubits):
            iteration_length = len(gate_encodings)
        else:
            raise ValueError("length of params lists in prepare_circuit_for_modifier has sto match")

        if iteration_length is None:
            raise ValueError("iter length from prepare_circuit shall not be None")

        return (number_of_qubits, gate_encodings, target_qubits, angles, control_qubits, iteration_length)

    def do_simulate(self, variables, initial_state, *args, **kwargs):
        """
        Helper function to perform simulation.

        - performs simulation in the following order:
            1. create a quantum state based on a given input integer
            2. apply a given quantum circuit on this state
            3. gate the amplitudes of the resulting wave function (wfn)
               after application of the circuit


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

        # prepare the circuit to apply onto the state made from an integer
        (number_of_qubits, gate_encodings, target_qubits, angles, control_qubits, iteration_length) = (
            BackendCircuitCudaq.prepare_circuit_for_state_modifier(self)
        )

        # apply state modifier (circuit) onto state and get amplitudes
        vector = cudaq.get_state(
            self.state_modifier_from_initial_state,
            number_of_qubits,
            gate_encodings,
            target_qubits,
            angles,
            control_qubits,
            iteration_length,
            initialize_state(initial_state, n_qubits=self.n_qubits),
        )

        wfn = QubitWaveFunction.from_array(array=np.array(vector), numbering=self.numbering)

        return wfn

    def initialize_circuit(self, *args, **kwargs):
        """
        return an empty circuit.
        for cudaq return an empty dict as the main data structure containing 4 key parameters:
        1. encodings of gates as integers
        2. indices of target qubits to which the gates are applied
        3. angles in case of parametrized gates
        4. index of a potential control qubit

        Parameters
        ----------
        args
        kwargs

        Returns
        -------
        """

        # as for now circuit supports only CNOT, single qubit gates and single rotations
        circuit = {"gate_encodings": [], "target_qubits": [], "angles": [], "control_qubits": []}

        return circuit

    def add_parametrized_gate(self, gate, circuit, variables, *args, **kwargs):
        """
        add a parametrized gate.
        -   for this fetch params like 1. angle 2. gate encoding 3. target qubits
            and store them in the corresponding lists within the circuit (as a dict)

        Parameters
        ----------
        gate: QGateImpl:
            the gate to add to the circuit.
        circuit:
            the circuit to which the gate is to be added
        variables:
            dict that tells values of variables; needed IFF the gate is an ExpPauli gate.
        args
        kwargs

        Returns
        -------
        None
        """
        gate_encoding = self.op_lookup[gate.name]

        # target qubits as a list
        target_qubits = [self.qubit(t) for t in gate.target]

        if len(target_qubits) != 1:
            raise ValueError(" at most 1 target qubits is supported, NOT MORE ")

        # save all control qubits into a list
        control_qubits = []
        if gate.is_controlled():
            for control in gate.control:
                control_qubits.append(self.qubit(control))

        # more than one control currently not supported
        if len(control_qubits) > 1:
            raise ValueError("at most 1 control qubit is supported for cudaq, NOT MORE ")

        # extract information about angle - one per gate
        angle = gate.parameter(variables)

        # if the gate has one control qubit append its index to the list of controls
        if len(control_qubits) == 1:
            circuit["control_qubits"].append(control_qubits[0])
        elif len(control_qubits) == 0:
            circuit["control_qubits"].append(-1)

        # append the angle of the gate
        circuit["angles"].append(angle)
        circuit["gate_encodings"].append(gate_encoding)
        circuit["target_qubits"].append(target_qubits[0])

    def add_basic_gate(self, gate, circuit, *args, **kwargs):
        """
        add an unparametrized gate to the circuit.
        Parameters
        -   add a gate with 1. control 2. target 3. encoding
            since these gates are not parametrized save an angle of 0.0 into the angles list

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

        # fetch the gate encoding
        gate_encoding = self.op_lookup[gate.name]

        # target qubits as a list
        target_qubits = [self.qubit(t) for t in gate.target]

        if len(target_qubits) != 1:
            raise ValueError(" at most 1 target qubits is supported, NOT MORE ")

        # save all control qubits into a list
        control_qubits = []
        if gate.is_controlled():
            for control in gate.control:
                control_qubits.append(self.qubit(control))

        # more than one control currently not supported
        if len(control_qubits) > 1:
            raise ValueError("at must 1 control qubit is supported, NOT MORE ")

        # if the gate has one control qubit append its index to the list of the controls
        if len(control_qubits) == 1:
            circuit["control_qubits"].append(control_qubits[0])
        elif len(control_qubits) == 0:
            circuit["control_qubits"].append(-1)

        circuit["target_qubits"].append(target_qubits[0])
        circuit["gate_encodings"].append(gate_encoding)

        # since primitive gates have to angle add zero to the list of angles
        # so all 4 lists have the same length for iteration
        circuit["angles"].append(0.0)


class BackendExpectationValueCudaq(BackendExpectationValue):
    """
    Class representing Expectation Values compiled for Cudaq.

    Overrides some methods of BackendExpectationValue, which should be seen for details.
    """

    use_mapping = True
    BackendCircuitType = BackendCircuitCudaq

    def simulate(self, variables, initial_state: Union[int, QubitWaveFunction], *args, **kwargs) -> np.array:
        """
        Perform simulation of this expectationvalue.
        Parameters
        ----------
        variables:
            variables, to be supplied to the underlying circuit.
        args
        kwargs

        Returns
        -------
        numpy.array:
            the result of simulation as an array.
        """
        # fast return if possible
        if self.H is None:
            return np.asarray([0.0])
        elif len(self.H) == 0:
            return np.asarray([0.0])
        elif isinstance(self.H, numbers.Number):
            return np.asarray[self.H]

        # update the variables for correct expectation value evaluation
        self.U.update_variables(variables)

        # prepare circuit to apply onto state
        (number_of_qubits, gate_encodings, target_qubits, angles, control_qubits, iteration_length) = (
            BackendCircuitCudaq.prepare_circuit_for_state_modifier(self)
        )

        # array containing results of exp. value simulation
        resulting_expectation_values = []

        # go over all given hamiltonians
        for hamiltonian in self.H:
            # compute expectation value between hamiltonian and state
            expectation_value = cudaq.observe(
                BackendCircuitCudaq.state_modifier_from_initial_state,
                hamiltonian,
                number_of_qubits,
                gate_encodings,
                target_qubits,
                angles,
                control_qubits,
                iteration_length,
                initialize_state(initial_state, n_qubits=self.n_qubits),
            ).expectation()

            # store exp. val in results array
            resulting_expectation_values.append(expectation_value)

        return np.asarray(resulting_expectation_values)

    def XX__XX__initialize_hamiltonian_old_implementation(self, hamiltonians):
        """
        Convert reduced hamiltonians to native Cudaq types for efficient expectation value evaluation.

        Parameters
        ----------
        hamiltonians:
            an interable set of hamiltonian objects.

        Returns
        -------
        list:
            initialized hamiltonian objects.

        """
        # Map logical qubit labels to consecutive indices (0,1,2,...)
        # this turns cases like [0,2,5] into [0,1,2] for cudaq
        qubit_map = {}
        for i, q in enumerate(self.U.abstract_circuit.qubits):
            qubit_map[q] = i

        list_of_initialized_hamiltonians = []
        # assemble hamiltonian with cudaq "spin" objects
        for hamiltonian in hamiltonians:
            hamiltonian_as_spin = 0  # Initialize per Hamiltonian
            for paulistring in hamiltonian.paulistrings:
                term = 1  # Start with identity
                for qubit, gate in paulistring.items():
                    mapped_qubit = qubit_map[qubit]
                    if gate == "X":
                        term *= spin.x(mapped_qubit)
                    elif gate == "Y":
                        term *= spin.y(mapped_qubit)
                    elif gate == "Z":
                        term *= spin.z(mapped_qubit)
                term *= paulistring._coeff  # Apply coefficient
                hamiltonian_as_spin += term  # Accumulate terms
            list_of_initialized_hamiltonians.append(hamiltonian_as_spin)
        return list_of_initialized_hamiltonians

    def initialize_hamiltonian(self, hamiltonians):
        """
        Convert reduced hamiltonians to native Cudaq types for efficient expectation value evaluation.

        Parameters
        ----------
        hamiltonians:
            an interable set of hamiltonian objects.

        Returns
        -------
        list:
            initialized hamiltonian objects.

        """
        # Map logical qubit labels to consecutive indices
        qubit_map = {q: i for i, q in enumerate(self.U.abstract_circuit.qubits)}
        list_of_initialized_hamiltonians = []

        # assemble hamiltonian with cudaq "spin" objects
        for hamiltonian in hamiltonians:
            hamiltonian_as_spin = None
            for paulistring in hamiltonian.paulistrings:
                # store the spin operators in a list
                ops = []
                for qubit, gate in paulistring.items():
                    mapped_qubit = qubit_map[qubit]
                    if gate == "X":
                        ops.append(spin.x(mapped_qubit))
                    elif gate == "Y":
                        ops.append(spin.y(mapped_qubit))
                    elif gate == "Z":
                        ops.append(spin.z(mapped_qubit))
                # If no operators, skip to next paulistring
                if ops:
                    # start with the first spin operator (instead of identity "1")
                    term = ops[0]
                    for op in ops[1:]:
                        term *= op
                    term *= paulistring._coeff
                else:
                    # If no operators, create a "zero operator" (identity with zero coefficient)
                    term = spin.z(0) * 0.0 + paulistring._coeff
                if hamiltonian_as_spin is None:
                    hamiltonian_as_spin = term
                else:
                    hamiltonian_as_spin += term
            # If no paulistrings, set to "zero operator" (not float)
            if hamiltonian_as_spin is None:
                hamiltonian_as_spin = spin.z(0) * 0.0
            list_of_initialized_hamiltonians.append(hamiltonian_as_spin)
        return list_of_initialized_hamiltonians


def initialize_state(
    state: Union[int, QubitWaveFunction], n_qubits: Optional[int] = None, numbering: BitNumbering = BitNumbering.LSB
) -> cudaq.State:
    if isinstance(state, int):
        if n_qubits is None:
            raise ValueError("Need to specify n_qubits for initializing CudaQ basis state")
        amplitudes = np.zeros(2**n_qubits, dtype=cudaq.complex())
        # Reverse because Tequila uses MSB bit ordering by default, while CudaQ uses LSB
        # TODO: It would be better to handle this somewhere else, e.g. by passing a BitString object
        #   in simulator_api (which stores its ordering), and then convert to the backend specific
        #   bit-ordering before passing this to the backends.
        if numbering == BitNumbering.LSB:
            index = reverse_int_bits(state, n_qubits)
        amplitudes[index] = 1
        return cudaq.State.from_data(amplitudes)
    elif isinstance(state, QubitWaveFunction):
        return cudaq.State.from_data(state.to_array(out_numbering=numbering).astype(cudaq.complex()))
    else:
        raise TypeError("State must be an int or QubitWaveFunction")
