from tequila.simulators.simulator_base import BackendExpectationValue, BackendCircuit
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila.utils import TequilaException
from tequila.hamiltonian import paulis, PauliString
from tequila.circuit._gates_impl import ExponentialPauliGateImpl, QGateImpl, RotationGateImpl, QubitHamiltonian
from tequila import BitNumbering
from tequila.circuit import compile_circuit

import hashlib
import numpy
import os
import spex_tequila

numbering = BitNumbering.LSB

class TequilaSpexException(TequilaException):
    pass

def extract_pauli_dict(ps):
        # Extract a dict {qubit: 'X'/'Y'/'Z'} from ps.
        # ps is PauliString or QubitHamiltonian
        if isinstance(ps, PauliString):
            return dict(ps.items())
        elif isinstance(ps, QubitHamiltonian):
            if len(ps.paulistrings) == 1:
                return dict(ps.paulistrings[0].items())
            else:
                raise TequilaSpexException("Rotation gate generator with multiple PauliStrings is not supported.")
        else:
            raise TequilaSpexException(f"Unexpected generator type: {type(ps)}")

def circuit_hash(abstract_circuit):
    sha = hashlib.md5()
    if abstract_circuit is None:
        return None
    for g in abstract_circuit.gates:
        gate_str = f"{type(g).__name__}:{g.name}:{g.target}:{g.control}:{g.generator}\n"
        sha.update(gate_str.encode('utf-8'))
    return sha.hexdigest()

class BackendCircuitSpex(BackendCircuit):
    compiler_arguments = {
        "multitarget": True,
        "multicontrol": True,
        "trotterized": True,
        "generalized_rotation": True,
        "exponential_pauli": False,
        "controlled_exponential_pauli": True,
        "hadamard_power": True,
        "controlled_power": True,
        "power": True,
        "toffoli": True,
        "controlled_phase": True,
        "phase": True,
        "phase_to_z": True,
        "controlled_rotation": True,
        "swap": True,
        "cc_max": True,
        "ry_gate": True,
        "y_gate": True,
        "ch_gate": True
    }


    def __init__(self, 
                 abstract_circuit=None, 
                 variables=None, 
                 num_threads=-1, 
                 amplitude_threshold=1e-14, 
                 angle_threshold=1e-14,
                 *args, **kwargs):
        
        self._cached_circuit_hash = None
        self._cached_circuit = []

        self.num_threads = num_threads
        self.amplitude_threshold = amplitude_threshold
        self.angle_threshold = angle_threshold

        super().__init__(abstract_circuit=abstract_circuit, variables=variables, *args, **kwargs)


    def initialize_circuit(self, *args, **kwargs):
        return []
    

    def create_circuit(self, abstract_circuit=None, variables=None, *args, **kwargs):
        if abstract_circuit is None:
            abstract_circuit = self.abstract_circuit

        new_hash = circuit_hash(abstract_circuit)

        if (new_hash is not None) and (new_hash == self._cached_circuit_hash):
            return self._cached_circuit
        
        circuit = super().create_circuit(abstract_circuit=abstract_circuit, variables=variables, *args, **kwargs)

        self._cached_circuit_key = abstract_circuit
        self._cached_circuit = circuit
        self._cached_circuit_hash = new_hash
        return circuit


    def add_basic_gate(self, gate, circuit, *args, **kwargs):
        exp_term = spex_tequila.ExpPauliTerm()
        if isinstance(gate, ExponentialPauliGateImpl):
            if self.angle_threshold != None and abs(gate.parameter) < self.angle_threshold:
                return
            exp_term.pauli_map = extract_pauli_dict(gate.paulistring)
            exp_term.angle = gate.parameter
            circuit.append(exp_term)

        elif isinstance(gate, RotationGateImpl):
            if self.angle_threshold != None and abs(gate.parameter) < self.angle_threshold:
                return
            exp_term.pauli_map = extract_pauli_dict(gate.generator)
            exp_term.angle = gate.parameter
            circuit.append(exp_term)

        elif isinstance(gate, QGateImpl):
            for ps in gate.make_generator(include_controls=True).paulistrings:
                angle = numpy.pi * ps.coeff
                if self.angle_threshold != None and abs(angle) < self.angle_threshold:
                    continue
                exp_term = spex_tequila.ExpPauliTerm()
                exp_term.pauli_map = dict(ps.items())
                exp_term.angle = angle
                circuit.append(exp_term)

        else:
            raise TequilaSpexException(f"Unsupported gate object type: {type(gate)}. "
                                       "All gates should be compiled to exponential pauli or rotation gates.")



    def add_parametrized_gate(self, gate, circuit, *args, **kwargs):
        exp_term = spex_tequila.ExpPauliTerm()
        if isinstance(gate, ExponentialPauliGateImpl):
            if self.angle_threshold != None and abs(gate.parameter) < self.angle_threshold:
                return
            exp_term.pauli_map = extract_pauli_dict(gate.paulistring)
            exp_term.angle = gate.parameter
            circuit.append(exp_term)

        elif isinstance(gate, RotationGateImpl):
            if self.angle_threshold != None and abs(gate.parameter) < self.angle_threshold:
                return
            exp_term.pauli_map = extract_pauli_dict(gate.generator)
            exp_term.angle = gate.parameter
            circuit.append(exp_term)
        
        elif isinstance(gate, QGateImpl):
            for ps in gate.make_generator(include_controls=True).paulistrings:
                if self.angle_threshold != None and abs(gate.parameter) < self.angle_threshold:
                    print("used")
                    continue
                exp_term = spex_tequila.ExpPauliTerm()
                exp_term.pauli_map = dict(ps.items())
                exp_term.angle = gate.parameter
                circuit.append(exp_term)

        else:
            raise TequilaSpexException(f"Unsupported gate type: {type(gate)}. "
                                       "Only Exponential Pauli and Rotation gates are allowed after compilation.")


    def do_simulate(self, variables, initial_state=0, *args, **kwargs) -> QubitWaveFunction:
        """
        Simulates the circuit and returns the final qubit state.

        Args:
            variables: Variables to adjust the circuit (not used for fixed gates).
            initial_state: The initial state of the qubit.

        Returns:
            QubitWaveFunction: The final state after applying the circuit.
        """
        n_qubits = self.n_qubits
        # Prepare the initial state
        if isinstance(initial_state, int):
            if initial_state == 0:
                state = spex_tequila.initialize_zero_state(n_qubits)
            else:
                state = {initial_state: 1.0 + 0j}
        else:
            # initial_state is already a QubitWaveFunction
            state = initial_state.to_dictionary()

        final_state = spex_tequila.apply_U(self.circuit, state)

        if self.amplitude_threshold != None:
            for basis, amplitude in list(final_state.items()):
                if abs(amplitude) < self.amplitude_threshold:
                    del final_state[basis]

        wfn = QubitWaveFunction(n_qubits=n_qubits, numbering=numbering)
        for state, amplitude in final_state.items():
            wfn[state] = amplitude
        return wfn


class BackendExpectationValueSpex(BackendExpectationValue):
    """
    Backend for computing expectation values using the spex_tequila C++ module.
    """
    BackendCircuitType = BackendCircuitSpex

    def __init__(self, *args,
                 num_threads=-1,
                 amplitude_threshold=1e-14, 
                 angle_threshold=1e-14,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.num_threads = num_threads
        self.amplitude_threshold = amplitude_threshold
        self.angle_threshold = angle_threshold
    
        if isinstance(self.U, BackendCircuitSpex):
            self.U.num_threads = num_threads
            self.U.amplitude_threshold = amplitude_threshold
            self.U.angle_threshold = angle_threshold

    def initialize_hamiltonian(self, hamiltonians):
        """
        Initializes the Hamiltonian terms for the simulation.

        Args:
            hamiltonians: A list of Hamiltonian objects.

        Returns:
            tuple: A converted list of (pauli_string, coefficient) tuples.
        """
        # Convert Tequila Hamiltonians into a list of (pauli_string, coeff) tuples for spex_tequila.
        converted = []
        for H in hamiltonians:
            terms = []
            for ps in H.paulistrings:
                # Construct Pauli string like "X(0)Y(1)"
                pauli_map = dict(ps.items()) 
                term = spex_tequila.ExpPauliTerm()
                term.pauli_map = pauli_map 
                terms.append((term, ps.coeff))                    
            converted.append(terms)
        return tuple(converted)


    def simulate(self, variables, initial_state=0, *args, **kwargs):
        """
        Computes the expectation value by simulating the circuit U and evaluating ⟨ψ|H|ψ⟩.

        Args:
            variables: Variables to adjust the circuit (not used for fixed gates).
            initial_state: The initial state of the qubit.

        Returns:
            numpy.ndarray: The computed expectation values for the Hamiltonian terms.
        """

        # variables as dict, variable map for var to gates

        self.update_variables(variables)
        n_qubits = self.U.n_qubits

        # Prepare the initial state
        if isinstance(initial_state, int):
            if initial_state == 0:
                state = spex_tequila.initialize_zero_state(n_qubits)
            else:
                state = {initial_state: 1.0 + 0j}
        else:
            # initial_state is a QubitWaveFunction
            state = initial_state.to_dictionary()

        final_state = spex_tequila.apply_U(self.U.circuit, state)

        if self.amplitude_threshold != None:
            for basis, amplitude in list(final_state.items()):
                if abs(amplitude) < self.amplitude_threshold:
                    del final_state[basis]

        if "SPEX_NUM_THREADS" in os.environ:
            self.num_threads = int(os.environ["SPEX_NUM_THREADS"])
        elif "OMP_NUM_THREADS" in os.environ:
            self.num_threads = int(os.environ["OMP_NUM_THREADS"])

        # Calculate the expectation value for each Hamiltonian
        results = []
        for H_terms in self.H:
            val = spex_tequila.expectation_value_parallel(final_state, final_state, H_terms, num_threads=-1)
            results.append(val.real)
        return numpy.array(results)


    def sample(self, variables, samples, initial_state=0, *args, **kwargs):
        return super().sample(variables=variables, samples=samples, initial_state=initial_state, *args, **kwargs)

    def sample_paulistring(self, samples: int, paulistring, variables, initial_state=0, *args, **kwargs):
        return super().sample_paulistring(samples, paulistring, variables, initial_state=initial_state, *args, **kwargs)