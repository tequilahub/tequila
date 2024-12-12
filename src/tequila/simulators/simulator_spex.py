from tequila.simulators.simulator_base import BackendExpectationValue, BackendCircuit
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila.utils import TequilaException
from tequila.hamiltonian import paulis, PauliString
from tequila.circuit._gates_impl import ExponentialPauliGateImpl

import numpy
import spex_tequila

class TequilaSpexException(TequilaException):
    pass

class BackendCircuitSpex(BackendCircuit):
    compiler_arguments = {
        "trotterized": False,
        "swap": False,
        "multitarget": False,
        "controlled_rotation": False,
        "generalized_rotation": False,
        "exponential_pauli": True,
        "controlled_exponential_pauli": True,
        "phase": False,
        "power": False,
        "hadamard_power": False,
        "controlled_power": False,
        "controlled_phase": False,
        "toffoli": False,
        "phase_to_z": True,
        "cc_max": False
    }


    def initialize_circuit(self, *args, **kwargs):
        return []

    def add_basic_gate(self, gate, circuit, *args, **kwargs):
        """
        Adds a basic Exponential Pauli gate to the circuit.

        Args:
            gate: The gate object to be added.
            circuit: The list representing the circuit.
        """

        ps = gate.paulistring
        angle = gate.parameter


        # Ensure that ps is a PauliString
        if isinstance(ps, PauliString):
            pauli_dict = dict(ps.items())
        else:
            raise TequilaSpexException(f"Unexpected paulistring type in add_basic_gate: {type(ps)}")

        # Create the spex_pauli_string in the correct format
        spex_pauli_string = "".join([f"{p.upper()}({q})" for q, p in pauli_dict.items()])

        # Add the gate to the circuit
        circuit.append((spex_pauli_string, angle))



    def add_parametrized_gate(self, gate, circuit, *args, **kwargs):
        """
        Adds a parametrized Exponential Pauli gate to the circuit.

        Args:
            gate: The gate object to be added.
            circuit: The list representing the circuit.
        """

        # Check if gate is an ExponentialPauliGateImpl
        if isinstance(gate, ExponentialPauliGateImpl):
            ps = gate.paulistring
            angle = gate.parameter

            # Ensure that ps is a PauliString
            if isinstance(ps, PauliString):
                pauli_dict = dict(ps.items())
            elif isinstance(ps, dict):
                pauli_dict = ps
            else:
                raise TequilaSpexException(f"Unexpected paulistring type: {type(ps)}")

            # Create the spex_pauli_string in the correct format
            spex_pauli_string = "".join([f"{p.upper()}({q})" for q, p in pauli_dict.items()])

            # Add the gate to the circuit
            circuit.append((spex_pauli_string, angle))
        else:
            # If the gate is not an Exponential Pauli gate, raise an error
            raise TequilaSpexException(f"Unsupported gate type: {type(gate)}. Only Exponential Pauli gates are allowed.")


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

        # Create the QubitWaveFunction from the final state
        wfn = QubitWaveFunction.from_dictionary(final_state, n_qubits=n_qubits, numbering=self.numbering)
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
            val = spex_tequila.expectation_value(final_state, final_state, H_terms)
            results.append(val.real)
        return numpy.array(results)


    def sample(self, variables, samples, initial_state=0, *args, **kwargs):
        return super().sample(variables=variables, samples=samples, initial_state=initial_state, *args, **kwargs)

    def sample_paulistring(self, samples: int, paulistring, variables, initial_state=0, *args, **kwargs):
        return super().sample_paulistring(samples, paulistring, variables, initial_state=initial_state, *args, **kwargs)
