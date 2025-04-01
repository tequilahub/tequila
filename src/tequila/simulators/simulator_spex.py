from tequila.simulators.simulator_base import BackendExpectationValue, BackendCircuit
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila.utils import TequilaException
from tequila.hamiltonian import PauliString
from tequila.circuit._gates_impl import ExponentialPauliGateImpl, QGateImpl, RotationGateImpl, QubitHamiltonian
from tequila.circuit.gates import QubitExcitationImpl
from tequila import BitNumbering


import hashlib
import numpy
import os
import spex_tequila
import gc

numbering = BitNumbering.MSB

class TequilaSpexException(TequilaException):
    """Custom exception for SPEX simulator errors"""
    pass

def extract_pauli_dict(ps):
    """
    Extract qubit:operator mapping from PauliString/QubitHamiltonian
    Args:
        ps: PauliString or single-term QubitHamiltonian
    Returns:
        dict: {qubit: 'X'/'Y'/'Z'}
    """

    if isinstance(ps, PauliString):
        return dict(ps.items())
    if isinstance(ps, QubitHamiltonian) and len(ps.paulistrings) == 1:
        return dict(ps.paulistrings[0].items())
    raise TequilaSpexException("Unsupported generator type")

def circuit_hash(abstract_circuit, variables=None):
    """
    Create MD5 hash for circuit caching
    Uses gate types, targets, controls and generators for uniqueness
    """
    sha = hashlib.md5()
    if abstract_circuit is None:
        return None
    for g in abstract_circuit.gates:
        gate_str = f"{type(g).__name__}:{g.name}:{g.target}:{g.control}:{g.generator}:{getattr(g, 'parameter', None)}\n"
        sha.update(gate_str.encode('utf-8'))
    if variables:
        for key, value in sorted(variables.items()):
            sha.update(f"{key}:{value}\n".encode('utf-8'))
    return sha.hexdigest()

class BackendCircuitSpex(BackendCircuit):
    """SPEX circuit implementation using sparse state representation"""

    # Circuit compilation configuration
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
                 compress_qubits=True,
                 *args, **kwargs):
        
        # Circuit chaching
        self.circuit_cache = {}

        # Performance parameters
        self.num_threads = num_threads
        self.amplitude_threshold = amplitude_threshold
        self.angle_threshold = angle_threshold

        # State compression
        self.compress_qubits = compress_qubits
        self.n_qubits_compressed = None
        self.hamiltonians = None
        
        super().__init__(abstract_circuit=abstract_circuit, variables=variables, *args, **kwargs)

    @property
    def n_qubits(self):
        """Get number of qubits after compression (if enabled)"""
        used = set()
        if hasattr(self, "circuit") and self.circuit:
            for term in self.circuit:
                used.update(term.pauli_map.keys())
        
        if self.abstract_circuit is not None and hasattr(self.abstract_circuit, "gates"):
            for gate in self.abstract_circuit.gates:
                if hasattr(gate, "target"):
                    if isinstance(gate.target, (list, tuple)):
                        used.update(gate.target)
                    else:
                        used.add(gate.target)
                if hasattr(gate, "control") and gate.control:
                    if isinstance(gate.control, (list, tuple)):
                        used.update(gate.control)
                    else:
                        used.add(gate.control)
        computed = max(used) + 1 if used else 0
        return max(super().n_qubits, computed)

    def initialize_circuit(self, *args, **kwargs):
        return []
    
    def create_circuit(self, abstract_circuit=None, variables=None, *args, **kwargs):
        """Compile circuit with caching using MD5 hash"""
        if abstract_circuit is None:
            abstract_circuit = self.abstract_circuit

        key = circuit_hash(abstract_circuit, variables)

        if key in self.circuit_cache:
            return self.circuit_cache[key]
        
        circuit = super().create_circuit(abstract_circuit=abstract_circuit, variables=variables, *args, **kwargs)

        self.circuit_cache[key] = circuit

        return circuit
    
    def compress_qubit_indices(self):
        """
        Optimize qubit indices by mapping used qubits to contiguous range
        Reduces memory usage by eliminating unused qubit dimensions
        """
        if not self.compress_qubits or not (hasattr(self, "circuit") and self.circuit):
            return

        # Collect all qubits used in circuit and Hamiltonians
        used_qubits = set()
        for term in self.circuit:
            used_qubits.update(term.pauli_map.keys())
        for ham in self.hamiltonians:
            for term, _ in ham:
                used_qubits.update(term.pauli_map.keys())

        if not used_qubits:
            self.n_qubits_compressed = 0
            return

        # Create qubit mapping and remap all terms
        qubit_map = {old: new for new, old in enumerate(sorted(used_qubits))}
        
        for term in self.circuit:
            term.pauli_map = {qubit_map[old]: op for old, op in term.pauli_map.items()}

        if self.hamiltonians is not None:    
            for ham in self.hamiltonians:
                for term, _ in ham:
                    term.pauli_map = {qubit_map[old]: op for old, op in term.pauli_map.items()}

        self.n_qubits_compressed = len(used_qubits)

    def update_variables(self, variables, *args, **kwargs):
        if variables is None:
            variables = {}
        super().update_variables(variables)
        self.circuit = self.create_circuit(abstract_circuit=self.abstract_circuit, variables=variables)

    def assign_parameter(self, param, variables):
        if isinstance(param, (int, float, complex)):
            return float(param)
        if isinstance(param, str):
            if param in variables:
                return float(variables[param])
            else:
                raise TequilaSpexException(f"Variable '{param}' not found in variables")
        if callable(param):
            result = param(variables)
            return float(result)
        
        raise TequilaSpexException(f"Can't assign parameter '{param}'.")


    def add_basic_gate(self, gate, circuit, *args, **kwargs):
        """Convert Tequila gates to SPEX exponential Pauli terms"""
        exp_term = spex_tequila.ExpPauliTerm()
        if isinstance(gate, ExponentialPauliGateImpl):
            if self.angle_threshold is not None and abs(gate.parameter) < self.angle_threshold:
                return
            exp_term.pauli_map = extract_pauli_dict(gate.paulistring)
            exp_term.angle = gate.parameter
            circuit.append(exp_term)

        elif isinstance(gate, RotationGateImpl):
            if self.angle_threshold is not None and abs(gate.parameter) < self.angle_threshold:
                return
            exp_term.pauli_map = extract_pauli_dict(gate.generator)
            exp_term.angle = gate.parameter
            circuit.append(exp_term)

        elif isinstance(gate, QubitExcitationImpl):
            compiled_gate = gate.compile(exponential_pauli=True)
            for sub_gate in compiled_gate.abstract_circuit.gates:
                self.add_basic_gate(sub_gate, circuit, *args, **kwargs)

        elif isinstance(gate, QGateImpl):
            if gate.name.lower() in ["x","y","z"]:
                # Convert standard gates to Pauli rotations
                for ps in gate.make_generator(include_controls=True).paulistrings:
                    angle = numpy.pi * ps.coeff
                    if self.angle_threshold is not None and abs(angle) < self.angle_threshold:
                        continue
                    exp_term = spex_tequila.ExpPauliTerm()
                    exp_term.pauli_map = dict(ps.items())
                    exp_term.angle = angle
                    circuit.append(exp_term)
            elif gate.name.lower() in ["h", "hadamard"]:
                assert len(gate.target)==1
                target = gate.target[0]
                for ps in ["-0.25*Y({q})", "Z({q})", "0.25*Y({q})"]:
                    ps = QubitHamiltonian(ps.format(q=gate.target[0])).paulistrings[0]
                    angle = numpy.pi * ps.coeff
                    exp_term = spex_tequila.ExpPauliTerm()
                    exp_term.pauli_map = dict(ps.items())
                    exp_term.angle = angle
                    circuit.append(exp_term)
            else:
                raise TequilaSpexException("{} not supported. Only x,y,z,h".format(gate.name.lower()))

        else:
            raise TequilaSpexException(f"Unsupported gate object type: {type(gate)}. "
                                       "All gates should be compiled to exponential pauli or rotation gates.")



    def add_parametrized_gate(self, gate, circuit, *args, **kwargs):
        """Convert Tequila parametrized gates to SPEX exponential Pauli terms"""
        exp_term = spex_tequila.ExpPauliTerm()
        if isinstance(gate, ExponentialPauliGateImpl):
            angle = self.assign_parameter(gate.parameter, kwargs.get("variables", {}))
            if self.angle_threshold is not None and abs(angle) < self.angle_threshold:
                return
            exp_term.pauli_map = extract_pauli_dict(gate.paulistring)
            exp_term.angle = angle
            circuit.append(exp_term)

        elif isinstance(gate, RotationGateImpl):
            angle = self.assign_parameter(gate.parameter, kwargs.get("variables", {}))
            if self.angle_threshold is not None and abs(angle) < self.angle_threshold:
                return
            exp_term.pauli_map = extract_pauli_dict(gate.generator)
            exp_term.angle = angle
            circuit.append(exp_term)

        elif isinstance(gate, QubitExcitationImpl):
            compiled_gate = gate.compile(exponential_pauli=True)
            for sub_gate in compiled_gate.abstract_circuit.gates:
                self.add_parametrized_gate(sub_gate, circuit, *args, **kwargs)
        
        elif isinstance(gate, QGateImpl):
            for ps in gate.make_generator(include_controls=True).paulistrings:
                if self.angle_threshold is not None and abs(gate.parameter) < self.angle_threshold:
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
        Simulate circuit and return final state
        Args:
            initial_state: Starting state (int or QubitWaveFunction)
        Returns:
            QubitWaveFunction: Sparse state representation
        """

        if self.compress_qubits and self.n_qubits_compressed is not None and self.n_qubits_compressed > 0:
            n_qubits = self.n_qubits_compressed
        else:
            n_qubits = self.n_qubits

        # Initialize state
        if isinstance(initial_state, (int, numpy.integer)):
            if initial_state == 0:
                state = spex_tequila.initialize_zero_state(n_qubits)
            else:
                state = {initial_state: 1.0 + 0j}
        else:
            # initial_state is already a QubitWaveFunction
            state = {k: v for k, v in initial_state.raw_items()}

        # Apply circuit with amplitude thresholding, -1.0 disables threshold in spex_tequila
        threshold = self.amplitude_threshold if self.amplitude_threshold is not None else -1.0
        final_state = spex_tequila.apply_U(self.circuit, state, threshold, n_qubits)

        wfn_MSB = QubitWaveFunction(n_qubits=n_qubits, numbering=BitNumbering.MSB)
        for state, amplitude in final_state.items():
            wfn_MSB[state] = amplitude

        del final_state
        gc.collect()

        return wfn_MSB
    
    def simulate(self, variables, initial_state=0, *args, **kwargs) -> QubitWaveFunction:
        """Override simulate to avoid automatic mapping by KeyMapSubregisterToRegister"""
        self.update_variables(variables)
        result = self.do_simulate(variables=variables, initial_state=initial_state, *args, **kwargs)
        return result


class BackendExpectationValueSpex(BackendExpectationValue):
    """SPEX expectation value calculator using sparse simulations"""
    BackendCircuitType = BackendCircuitSpex

    def __init__(self, *args,
                 num_threads=-1,
                 amplitude_threshold=1e-14, 
                 angle_threshold=1e-14,
                 compress_qubits=True,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.num_threads = num_threads
        self.amplitude_threshold = amplitude_threshold
        self.angle_threshold = angle_threshold
    
        # Configure circuit parameters
        if isinstance(self.U, BackendCircuitSpex):
            self.U.num_threads = num_threads
            self.U.amplitude_threshold = amplitude_threshold
            self.U.angle_threshold = angle_threshold
            self.U.compress_qubits = compress_qubits

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

        if isinstance(self.U, BackendCircuitSpex):
            self.U.hamiltonians = converted

        return tuple(converted)


    def simulate(self, variables, initial_state=0, *args, **kwargs):
        """
        Calculate expectation value through sparse simulation
        Returns:
            numpy.ndarray: Expectation values for each Hamiltonian term
        """

        # Prepare simulation
        self.update_variables(variables)
        if self.U.compress_qubits:
            self.U.compress_qubit_indices()

        if self.U.compress_qubits and self.U.n_qubits_compressed is not None and self.U.n_qubits_compressed > 0:
            n_qubits = self.U.n_qubits_compressed
        else:
            n_qubits = self.U.n_qubits

        # Prepare the initial state
        if isinstance(initial_state, int):
            if initial_state == 0:
                state = spex_tequila.initialize_zero_state(n_qubits)
            else:
                state = {initial_state: 1.0 + 0j}
        else:
            # initial_state is a QubitWaveFunction
            state = {k: v for k, v in initial_state.raw_items()}

        self.U.circuit = [t for t in self.U.circuit if abs(t.angle) >= self.U.angle_threshold]

        threshold = self.amplitude_threshold if self.amplitude_threshold is not None else -1.0
        final_state = spex_tequila.apply_U(self.U.circuit, state, threshold, n_qubits)
        del state

        if "SPEX_NUM_THREADS" in os.environ:
            self.num_threads = int(os.environ["SPEX_NUM_THREADS"])
        elif "OMP_NUM_THREADS" in os.environ:
            self.num_threads = int(os.environ["OMP_NUM_THREADS"])

        # Calculate the expectation value for each Hamiltonian
        results = []
        for H_terms in self.H:
            val = spex_tequila.expectation_value_parallel(final_state, final_state, H_terms, n_qubits, num_threads=-1)
            results.append(val.real)
        
        del final_state
        gc.collect()
        
        return numpy.array(results)
