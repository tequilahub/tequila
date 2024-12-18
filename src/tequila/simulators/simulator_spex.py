from tequila.simulators.simulator_base import BackendExpectationValue, BackendCircuit
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila.utils import TequilaException
from tequila.hamiltonian import paulis, PauliString
from tequila.circuit._gates_impl import ExponentialPauliGateImpl, QGateImpl, RotationGateImpl, QubitHamiltonian
from tequila import BitNumbering
from tequila.circuit import compile_circuit

import numpy
import spex_tequila

numbering = BitNumbering.LSB

class TequilaSpexException(TequilaException):
    pass

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


    def initialize_circuit(self, *args, **kwargs):
        return []
    
    
    def extract_pauli_dict(self, ps):
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


    def add_basic_gate(self, gate, circuit, *args, **kwargs):
        if isinstance(gate, ExponentialPauliGateImpl):
            ps = gate.paulistring
            angle = gate.parameter
            pauli_dict = self.extract_pauli_dict(ps)
            spex_pauli_string = "".join([f"{p.upper()}({q})" for q, p in pauli_dict.items()])
            if not spex_pauli_string:
                spex_pauli_string = "I"
            circuit.append((spex_pauli_string, angle))

        elif isinstance(gate, RotationGateImpl):
            ps = gate.generator
            angle = gate.parameter
            pauli_dict = self.extract_pauli_dict(ps)
            spex_pauli_string = "".join([f"{p.upper()}({q})" for q, p in pauli_dict.items()])
            if not spex_pauli_string:
                spex_pauli_string = "I"
            circuit.append((spex_pauli_string, angle))

        elif isinstance(gate, QGateImpl):
            for ps in gate.make_generator(include_controls=True).paulistrings:
                angle = numpy.pi * ps.coeff
                pauli_dict = self.extract_pauli_dict(ps)
                spex_pauli_string = "".join([f"{p.upper()}({q})" for q, p in pauli_dict.items()])
                if not spex_pauli_string:
                    spex_pauli_string = "I"
                circuit.append((spex_pauli_string, angle))

        else:
            raise TequilaSpexException(f"Unsupported gate object type: {type(gate)}. "
                                       "All gates should be compiled to exponential pauli or rotation gates.")


    def add_parametrized_gate(self, gate, circuit, *args, **kwargs):
        if isinstance(gate, ExponentialPauliGateImpl):
            ps = gate.paulistring
            angle = gate.parameter
            pauli_dict = self.extract_pauli_dict(ps)
            spex_pauli_string = "".join([f"{p.upper()}({q})" for q, p in pauli_dict.items()])
            if not spex_pauli_string:
                spex_pauli_string = "I"
            circuit.append((spex_pauli_string, angle))

        elif isinstance(gate, RotationGateImpl):
            ps = gate.generator
            angle = gate.parameter
            pauli_dict = self.extract_pauli_dict(ps)
            spex_pauli_string = "".join([f"{p.upper()}({q})" for q, p in pauli_dict.items()])
            if not spex_pauli_string:
                spex_pauli_string = "I"
            circuit.append((spex_pauli_string, angle))
        
        elif isinstance(gate, QGateImpl):
            for ps in gate.make_generator(include_controls=True).paulistrings:
                angle = gate.parameter
                pauli_dict = self.extract_pauli_dict(ps)
                spex_pauli_string = "".join([f"{p.upper()}({q})" for q, p in pauli_dict.items()])
                if not spex_pauli_string:
                    spex_pauli_string = "I"
                circuit.append((spex_pauli_string, angle))

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

        # Apply the Exponential Pauli Term gates
        U_terms = []
        for pauli_str, angle in self.circuit:
            U_terms.append(spex_tequila.ExpPauliTerm(pauli_str, angle))

        final_state = spex_tequila.apply_U(U_terms, state)

        wfn = QubitWaveFunction(n_qubits=n_qubits, numbering=numbering)
        for state, amplitude in final_state.items():
            wfn[state] = amplitude
        return wfn


class BackendExpectationValueSpex(BackendExpectationValue):
    """
    Backend for computing expectation values using the spex_tequila C++ module.
    """

    BackendCircuitType = BackendCircuitSpex

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
                spex_ps = "".join([f"{p.upper()}({q})" for q, p in dict(ps.items()).items()])
                if not spex_ps:
                    spex_ps = "I"                
                terms.append((spex_ps, ps.coeff))                    
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

        # Convert circuit gates into Exponential Pauli Terms
        U_terms = []
        for pauli_str, angle in self.U.circuit:
            U_terms.append(spex_tequila.ExpPauliTerm(pauli_str, angle))

        # Apply U to the initial state
        final_state = spex_tequila.apply_U(U_terms, state)

        # Calculate the expectation value for each Hamiltonian
        results = []
        for H_terms in self.H:
            val = spex_tequila.expectation_value_parallel(final_state, final_state, H_terms)
            results.append(val.real)
        return numpy.array(results)


    def sample(self, variables, samples, initial_state=0, *args, **kwargs):
        return super().sample(variables=variables, samples=samples, initial_state=initial_state, *args, **kwargs)

    def sample_paulistring(self, samples: int, paulistring, variables, initial_state=0, *args, **kwargs):
        return super().sample_paulistring(samples, paulistring, variables, initial_state=initial_state, *args, **kwargs)