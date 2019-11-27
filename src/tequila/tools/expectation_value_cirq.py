import cirq
import numpy

def expectation_value_cirq(final_state, hamiltonian, n_qubits):
    """
    Function from Philip to compute expectation values with cirq using this currently for consistency tests
    :param final_state:
    :param hamiltonian:
    :param n_qubits:
    :return:
    """

    # rewrite the hamiltonian as a sum of pauli strings:
    qubits = [cirq.LineQubit(i) for i in range(n_qubits)]

    coeffs = []
    list_of_h_terms = []
    for qubit_terms, coefficient in hamiltonian.items():
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