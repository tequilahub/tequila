from openvqe import HamiltonianQC, AnsatzUCC, ParametersQC, ParametersUCC
from openfermionpsi4._psi4_conversion_functions import parse_psi4_ccsd_amplitudes
from openvqe.ansatz.ansatz_ucc import ManyBodyAmplitudes
from openvqe.ansatz.ansatz_ucc import AnsatzUCC
import openvqe
import cirq
import numpy


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


if __name__ == "__main__":
    print("Demo for closed-shell UCC with psi4-CCSD trial state and first order Trotter decomposition")

    print("First get the Hamiltonian:")
    parameters_qc = ParametersQC(geometry="data/h2.xyz", basis_set="sto-3g")
    parameters_qc.transformation = "JW"
    parameters_qc.psi4.delete_output = False
    parameters_qc.psi4.delete_input = False
    parameters_qc.psi4.run_ccsd = True
    parameters_qc.filename = "psi4"
    hqc = HamiltonianQC(parameters_qc)
    print("parameters=", hqc.parameters)
    print("The Qubit Hamiltonian is:\n", hqc())

    print("Parse Guess CCSD amplitudes from the PSI4 calculation")
    # get initial amplitudes from psi4
    filename = parameters_qc.filename
    print("filename=", filename + ".out")
    print("n_electrons=", hqc.n_electrons())
    print("n_orbitals=", hqc.n_orbitals())

    # @todo get a functioning guess factory which does not use this parser
    singles, doubles = parse_psi4_ccsd_amplitudes(number_orbitals=hqc.n_orbitals() * 2,
                                                  n_alpha_electrons=hqc.n_electrons() // 2,
                                                  n_beta_electrons=hqc.n_electrons() // 2,
                                                  psi_filename=filename + ".out")

    amplitudes = ManyBodyAmplitudes(one_body=singles, two_body=doubles)

    print("Construct the AnsatzUCC class")

    parameters_ucc = ParametersUCC(backend="cirq", decomposition='trotter', trotter_steps=1)
    print(parameters_ucc)
    ucc = AnsatzUCC(parameters=parameters_ucc, hamiltonian=hqc)
    circuit = ucc(angles=amplitudes)

    print("created the following circuit:")
    print(circuit)

    print("run the circuit:")
    simulator = cirq.Simulator()
    result = simulator.simulate(program=circuit)
    print("resulting state is:")
    print("|psi>=", result.dirac_notation(decimals=5))
    print("type(result):", type(result))
    print("type(final_state)", type(result.final_simulator_state.state_vector))
    print("final_state.state_vector:\n", result.final_simulator_state.state_vector)
    print("Evaluate energy:")

    qubits = ucc.backend_handler.qubits  # [cirq.GridQubit(i, 0) for i in range(5)]
    test = cirq.X(qubits[0]) * cirq.Y(qubits[2])
    print(type(test))
    print(test)

    energy = expectation_value_cirq(final_state=result.final_simulator_state.state_vector, hamiltonian=hqc(),
                                    n_qubits=hqc.n_qubits())
    print("energy=", energy)
