from openvqe.circuit.circuit import QCircuit
import cirq
import numpy


def make_cirq_circuit(circuit: QCircuit) -> cirq.Circuit:
    """
    Converts a QCircuit object to a cirq.Circuit with qubits on a line
    :param circuit: The QCircuit object which shall be converted
    :return:
    """

    if isinstance(circuit, cirq.Circuit):
        return circuit

    n_qubits = circuit.max_qubit()
    qubits = [cirq.LineQubit(i) for i in range(n_qubits)]
    result = cirq.Circuit()
    for g in circuit.gates:

        if g.is_parametrized() and g.control is not None:
            # here we need recompilation
            rc = circuit.compile_controlled_rotation_gate(g)
            result += make_cirq_circuit(rc)
        else:
            tmp = cirq.Circuit()
            gate = None

            if g.name.upper() == "CNOT":
                gate = (cirq.CNOT(target=qubits[g.target[0]], control=qubits[g.control[0]]))
            else:
                if g.is_parametrized():
                    gate = getattr(cirq, g.name)(rads=g.angle)
                else:
                    gate = getattr(cirq, g.name)

                gate=gate.on(*[qubits[t] for t in g.target])

                if g.control is not None:
                        gate = gate.controlled_by(*[qubits[t] for t in g.control])

            tmp.append(gate)
            result += tmp
    return result


def show(circuit: QCircuit):
    """
    Conenience to use the cirq ASCII printout
    """
    print(make_cirq_circuit(circuit))


def simulate(circuit: QCircuit, initial_state=0, silent=True):
    """
    Simulates the circuit with cirq
    """
    simulator = cirq.Simulator()
    circuit = make_cirq_circuit(circuit)
    result = simulator.simulate(program=circuit, initial_state=initial_state)

    if not silent:
        print("Simulated circuit:")
        print(circuit)
        print("\nResulting State is:")
        print("|State>=", result.dirac_notation())

    return result

def expectation_value_cirq(final_state, hamiltonian, n_qubits):
    """
    Function from Philip to compute expectation values with cirq
    :param final_state:
    :param hamiltonian:
    :param n_qubits:
    :return:
    """

    # rewrite the hamiltonian as a sum of pauli strings:
    qubits = [cirq.LineQubit(i) for i in range(n_qubits)]

    coeffs = []
    list_of_h_terms = []
    for qubit_terms, coefficient in hamiltonian.terms.items():
        h_terms = []

        for i in range(n_qubits):
            h_terms.append(cirq.I)

        for tensor_term in qubit_terms:
            if tensor_term[1] == 'Z':
                h_terms[tensor_term[0]] = cirq.Z
            elif tensor_term[1] == 'Y':
                h_terms[tensor_term[0]] = cirq.Y
            elif tensor_term[1] == 'X':
                h_terms[tensor_term[0]] = cirq.X

        list_of_h_terms.append(cirq.PauliString({qubits[i]: h_terms[i] for i in range(n_qubits)}))
        coeffs.append(coefficient)

    # calculate the expectation value of the Hamiltonian:
    qubit_map = {qubits[i]: i for i in range(n_qubits)}
    len_ham = len(coeffs)
    energies = []
    for i in range(len_ham):
        num_qubits = final_state.shape[0].bit_length() - 1
        ket = numpy.reshape(numpy.copy(final_state), (2,) * num_qubits)
        for qubit, pauli in list_of_h_terms[i].items():
            buffer = numpy.empty(ket.shape, dtype=final_state.dtype)
            args = cirq.protocols.ApplyUnitaryArgs(
                target_tensor=ket,
                available_buffer=buffer,
                axes=(qubit_map[qubit],)
            )
            ket = cirq.protocols.apply_unitary(pauli, args)
        ket = numpy.reshape(ket, final_state.shape)
        arr = numpy.dot(numpy.transpose(final_state.conj()), ket) * coeffs[i]
        energies.append(arr)
    expectation_value = numpy.real(numpy.sum(energies))
    # print(expectation_value)

    return expectation_value
