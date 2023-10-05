import qulacs
import numbers, numpy
import warnings

from tequila import TequilaException, TequilaWarning
from tequila.utils.bitstrings import BitNumbering, BitString, BitStringLSB
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila.simulators.simulator_base import BackendCircuit, BackendExpectationValue, QCircuit, change_basis
from tequila.utils.keymap import KeyMapRegisterToSubregister

"""
Developer Note:
    Qulacs uses different Rotational Gate conventions: Rx(angle) = exp(i angle/2 X) instead of exp(-i angle/2 X)
    And the same for MultiPauli rotational gates
    The angles are scaled with -1.0 to keep things consistent with the rest of tequila
"""

class TequilaQulacsException(TequilaException):
    def __str__(self):
        return "Error in qulacs backend:" + self.message

class BackendCircuitQulacs(BackendCircuit):
    """
    Class representing circuits compiled to qulacs.
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
        "controlled_rotation": True, # needed for gates depending on variables
        "generalized_rotation": True,
        "exponential_pauli": False,
        "controlled_exponential_pauli": True,
        "phase": True,
        "power": True,
        "hadamard_power": True,
        "controlled_power": True,
        "controlled_phase": True,
        "toffoli": False,
        "phase_to_z": True,
        "cc_max": False
    }

    numbering = BitNumbering.LSB

    def __init__(self, abstract_circuit, noise=None, *args, **kwargs):
        """

        Parameters
        ----------
        abstract_circuit: QCircuit:
            the circuit to compile to qulacs
        noise: optional:
            noise to apply to the circuit.
        args
        kwargs
        """
        self.op_lookup = {
            'I': qulacs.gate.Identity,
            'X': qulacs.gate.X,
            'Y': qulacs.gate.Y,
            'Z': qulacs.gate.Z,
            'H': qulacs.gate.H,
            'Rx': (lambda c: c.add_parametric_RX_gate, qulacs.gate.RX),
            'Ry': (lambda c: c.add_parametric_RY_gate, qulacs.gate.RY),
            'Rz': (lambda c: c.add_parametric_RZ_gate, qulacs.gate.RZ),
            'SWAP': qulacs.gate.SWAP,
            'Measure': qulacs.gate.Measurement,
            'Exp-Pauli': None
        }
        self.measurements = None
        self.variables = []
        super().__init__(abstract_circuit=abstract_circuit, noise=noise, *args, **kwargs)
        self.has_noise=False
        if noise is not None:

            warnings.warn("Warning: noise in qulacs module will be dropped. Currently only works for qulacs version 0.5 or lower", TequilaWarning)

            self.has_noise=True
            self.noise_lookup = {
                'bit flip': [qulacs.gate.BitFlipNoise],
                'phase flip': [lambda target, prob: qulacs.gate.Probabilistic([prob],[qulacs.gate.Z(target)])],
                'phase damp': [lambda target, prob: qulacs.gate.DephasingNoise(target,(1/2)*(1-numpy.sqrt(1-prob)))],
                'amplitude damp': [qulacs.gate.AmplitudeDampingNoise],
                'phase-amplitude damp': [qulacs.gate.AmplitudeDampingNoise,
                                         lambda target, prob: qulacs.gate.DephasingNoise(target,(1/2)*(1-numpy.sqrt(1-prob)))
                                         ],
                'depolarizing': [lambda target,prob: qulacs.gate.DepolarizingNoise(target,3*prob/4)]
            }

            self.circuit=self.add_noise_to_circuit(noise)

    def initialize_state(self, n_qubits:int=None) -> qulacs.QuantumState:
        if n_qubits is None:
            n_qubits = self.n_qubits
        return qulacs.QuantumState(n_qubits)

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
        for k, angle in enumerate(self.variables):
            self.circuit.set_parameter(k, angle(variables))

    def do_simulate(self, variables, initial_state, *args, **kwargs):
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
        state = self.initialize_state(self.n_qubits)
        lsb = BitStringLSB.from_int(initial_state, nbits=self.n_qubits)
        state.set_computational_basis(BitString.from_binary(lsb.binary).integer)
        self.circuit.update_quantum_state(state)

        wfn = QubitWaveFunction.from_array(arr=state.get_vector(), numbering=self.numbering)
        return wfn

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


        for k in backend_result:
            converted_key = BitString.from_binary(BitStringLSB.from_int(integer=k, nbits=self.n_qubits).binary)
            if converted_key in result._state:
                result._state[converted_key] += 1
            else:
                result._state[converted_key] = 1

        if target_qubits is not None:
            mapped_target = [self.qubit_map[q].number for q in target_qubits]
            mapped_full = [self.qubit_map[q].number for q in self.abstract_qubits]
            keymap = KeyMapRegisterToSubregister(subregister=mapped_target, register=mapped_full)
            result = result.apply_keymap(keymap=keymap)

        return result

    def do_sample(self, samples, circuit, noise_model=None, initial_state=0, *args, **kwargs) -> QubitWaveFunction:
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
        state = self.initialize_state(self.n_qubits)
        lsb = BitStringLSB.from_int(initial_state, nbits=self.n_qubits)
        state.set_computational_basis(BitString.from_binary(lsb.binary).integer)
        circuit.update_quantum_state(state)
        sampled = state.sampling(samples)
        return self.convert_measurements(backend_result=sampled, target_qubits=self.measurements)

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
        qulacs.ParametricQuantumCircuit
        """
        return qulacs.ParametricQuantumCircuit(self.n_qubits)

    def add_exponential_pauli_gate(self, gate, circuit, variables, *args, **kwargs):
        """
        Add a native qulacs Exponential Pauli gate to a circuit.
        Parameters
        ----------
        gate: ExpPauliGateImpl:
            the gate to add
        circuit:
            the qulacs circuit, to which the gate is to be added.
        variables:
            dict containing values of the parameters appearing in the pauli gate.
        args
        kwargs

        Returns
        -------
        None
        """
        assert not gate.is_controlled()
        convert = {'x': 1, 'y': 2, 'z': 3}
        pind = [convert[x.lower()] for x in gate.paulistring.values()]
        qind = [self.qubit(x) for x in gate.paulistring.keys()]
        if len(gate.extract_variables()) > 0:
            self.variables.append(-gate.parameter * gate.paulistring.coeff)
            circuit.add_parametric_multi_Pauli_rotation_gate(qind, pind,
                                                             -gate.parameter(variables) * gate.paulistring.coeff)
        else:
            circuit.add_multi_Pauli_rotation_gate(qind, pind, -gate.parameter(variables) * gate.paulistring.coeff)

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
            dict that tells values of variables; needed IFF the gate is an ExpPauli gate.
        args
        kwargs

        Returns
        -------
        None
        """
        op = self.op_lookup[gate.name]
        if gate.name == 'Exp-Pauli':
            self.add_exponential_pauli_gate(gate, circuit, variables)
            return
        else:
            if len(gate.extract_variables()) > 0:
                op = op[0]
                self.variables.append(-gate.parameter)
                op(circuit)(self.qubit(gate.target[0]), -gate.parameter(variables=variables))
                if gate.is_controlled():
                    raise TequilaQulacsException("Gates which depend on variables can not be controlled! Gate was:\n{}".format(gate))
                return
            else:
                op = op[1]
                qulacs_gate = op(self.qubit(gate.target[0]), -gate.parameter(variables=variables))
        if gate.is_controlled():
            qulacs_gate = qulacs.gate.to_matrix_gate(qulacs_gate)
            for c in gate.control:
                qulacs_gate.add_control_qubit(self.qubit(c), 1)
        circuit.add_gate(qulacs_gate)

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
        qulacs_gate = op(*[self.qubit(t) for t in gate.target])
        if gate.is_controlled():
            qulacs_gate = qulacs.gate.to_matrix_gate(qulacs_gate)
            for c in gate.control:
                qulacs_gate.add_control_qubit(self.qubit(c), 1)

        circuit.add_gate(qulacs_gate)

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
        self.measurements = sorted(target_qubits)
        return circuit


    def add_noise_to_circuit(self,noise_model):
        """
        Apply noise from a NoiseModel to a circuit.
        Parameters
        ----------
        noise_model: NoiseModel:
            the noisemodel to apply to the circuit.

        Returns
        -------
        qulacs.ParametrizedQuantumCircuit:
            self.circuit, with noise added on.
        """
        c=self.circuit
        n=noise_model
        g_count=c.get_gate_count()
        new=self.initialize_circuit()
        for i in range(g_count):
            g=c.get_gate(i)
            new.add_gate(g)
            qubits=g.get_target_index_list() + g.get_control_index_list()
            for noise in n.noises:
                if len(qubits) == noise.level:
                    for j,channel in enumerate(self.noise_lookup[noise.name]):
                        for q in qubits:
                            chan=channel(q,noise.probs[j])
                            new.add_gate(chan)
        return new

    def optimize_circuit(self, circuit, max_block_size: int = 4, silent: bool = True, *args, **kwargs):
        """
        reduce circuit depth using the native qulacs optimizer.
        Parameters
        ----------
        circuit
        max_block_size: int: Default = 4:
            the maximum block size for use by the qulacs internal optimizer.
        silent: bool:
            whether or not to print the resullt of having optimized.
        args
        kwargs

        Returns
        -------
        qulacs.QuantumCircuit:
            optimized qulacs circuit.

        """
        old = circuit.calculate_depth()
        opt = qulacs.circuit.QuantumCircuitOptimizer()
        opt.optimize(circuit, max_block_size)
        if not silent:
            print("qulacs: optimized circuit depth from {} to {} with max_block_size {}".format(old,
                                                                                                circuit.calculate_depth(),
                                                                                                max_block_size))
        return circuit

class BackendExpectationValueQulacs(BackendExpectationValue):
    """
    Class representing Expectation Values compiled for Qulacs.

    Ovverrides some methods of BackendExpectationValue, which should be seen for details.
    """
    use_mapping = True
    BackendCircuitType = BackendCircuitQulacs

    def simulate(self, variables, *args, **kwargs) -> numpy.array:
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
            return numpy.asarray([0.0])
        elif len(self.H) == 0:
            return numpy.asarray([0.0])
        elif isinstance(self.H, numbers.Number):
            return numpy.asarray[self.H]

        self.U.update_variables(variables)
        state = self.U.initialize_state(self.n_qubits)
        self.U.circuit.update_quantum_state(state)
        result = []
        for H in self.H:
            if isinstance(H, numbers.Number):
                result.append(H) # those are accumulated unit strings, e.g 0.1*X(3) in wfn on qubits 0,1
            else:
                result.append(H.get_expectation_value(state))

        return numpy.asarray(result)

    def initialize_hamiltonian(self, hamiltonians):
        """
        Convert reduced hamiltonians to native Qulacs types for efficient expectation value evaluation.
        Parameters
        ----------
        hamiltonians:
            an interable set of hamiltonian objects.

        Returns
        -------
        list:
            initialized hamiltonian objects.

        """

        # map the reduced operators to the potentially smaller qubit system
        qubit_map = {}
        for i,q in enumerate(self.U.abstract_circuit.qubits):
            qubit_map[q] = i

        result = []
        for H in hamiltonians:
            qulacs_H = qulacs.Observable(self.n_qubits)
            for ps in H.paulistrings:
                string = ""
                for k, v in ps.items():
                    string += v.upper() + " " + str(qubit_map[k])
                qulacs_H.add_operator(ps.coeff, string)
            result.append(qulacs_H)
        return result

    def sample(self, variables, samples, *args, **kwargs) -> numpy.array:
        """
        Sample this Expectation Value.
        Parameters
        ----------
        variables:
            variables, to supply to the underlying circuit.
        samples: int:
            the number of samples to take.
        args
        kwargs

        Returns
        -------
        numpy.ndarray:
            the result of sampling as a number.
        """
        self.update_variables(variables)
        state = self.U.initialize_state(self.n_qubits)
        self.U.circuit.update_quantum_state(state)
        result = []
        for H in self._reduced_hamiltonians: # those are the hamiltonians which where non-used qubits are already traced out
            E = 0.0
            if H.is_all_z() and not self.U.has_noise:
                E = super().sample(samples=samples, variables=variables, *args, **kwargs)
            else:
                for ps in H.paulistrings:
                    # change basis, measurement is destructive so the state will be copied
                    # to avoid recomputation (except when noise was required)
                    bc = QCircuit()
                    for idx, p in ps.items():
                        bc += change_basis(target=idx, axis=p)
                    qbc = self.U.create_circuit(abstract_circuit=bc, variables=None)
                    Esamples = []
                    for sample in range(samples):
                        if self.U.has_noise and sample>0:
                            state = self.U.initialize_state(self.n_qubits)
                            self.U.circuit.update_quantum_state(state)
                            state_tmp = state
                        else:
                            state_tmp = state.copy()
                        if len(bc.gates) > 0:  # otherwise there is no basis change (empty qulacs circuit does not work out)
                            qbc.update_quantum_state(state_tmp)
                        ps_measure = 1.0
                        for idx in ps.keys():
                            assert idx in self.U.abstract_qubits # assert that the hamiltonian was really reduced
                            M = qulacs.gate.Measurement(self.U.qubit(idx), self.U.qubit(idx))
                            M.update_quantum_state(state_tmp)
                            measured = state_tmp.get_classical_value(self.U.qubit(idx))
                            ps_measure *= (-2.0 * measured + 1.0)  # 0 becomes 1 and 1 becomes -1
                        Esamples.append(ps_measure)
                    E += ps.coeff * sum(Esamples) / len(Esamples)
            result.append(E)
        return numpy.asarray(result)
