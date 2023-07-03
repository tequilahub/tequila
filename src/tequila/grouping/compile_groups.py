from tequila.grouping.binary_rep import BinaryHamiltonian
from tequila.grouping.fermionic_methods import get_fermion_wise, do_fff, do_svd
from tequila.grouping.fermionic_functions import n_elec
from tequila.utils import TequilaException
from openfermion import reverse_jordan_wigner
import tequila as tq
import numpy as np
import numpy.linalg as npl
import copy

def compile_commuting_parts(H, unitary_circuit="improved", *args, **kwargs):
    """
    Compile the commuting parts of a QubitHamiltonian
    Into a list of All-Z Hamiltonians and corresponding unitary rotations
    Parameters
    ----------
    H: the tq.QubitHamiltonian

    Returns
    -------
        A list of tuples containing all-Z Hamiltonian and corresponding Rotations
    """
    if "options" in kwargs and not(kwargs["options"] is None) and "method" in kwargs["options"]:
        method = kwargs["options"]["method"]
    else:
        method = "rlf"

    def method_class(method):
        if method == "lf" or method == "rlf" or method == "si" or method == "ics":
            return "qubit"
        elif method == "lr" or method == "fff-lr":
            return "fermionic"
    
    if method_class(method) == 'qubit':
        if unitary_circuit is None or unitary_circuit.lower() == "improved":
            # @ Zack
            result, suggested_samples = _compile_commuting_parts_zb(H, *args, **kwargs)
        else:
            # original implementation of Thomson (T.C. Yen)
            binary_H = BinaryHamiltonian.init_from_qubit_hamiltonian(H)
            commuting_parts, suggested_samples = binary_H.commuting_groups(*args, **kwargs)
            result = [cH.get_qubit_wise() for cH in commuting_parts]
        return result, suggested_samples
    
    elif method_class(method) == 'fermionic':
        options = kwargs["options"]
        if "n_el" in options:
            n_el = int(options["n_el"])
        elif "mol_name" in options: 
            n_el = n_elec(options["mol_name"])
        else:
            raise TequilaException("Please specify either {mol_name:} or {n_el:} in the dictionary provided as the keyward argument (optimize_measurements) to function (ExpectationValue).")

        if "reverse_H_transf" in options:
            reverse_H_transf = options["reverse_H_transf"]
        else:
            print("Using default reverse_H_transf, i.e., reverse_jordan_wigner.")
            reverse_H_transf = _reverse_jordan_wigner
        
        h_ferm = reverse_H_transf(H)

        if method == 'lr':
            u_ops, cartan_obt, cartan_tbt, suggested_samples = do_svd(h_ferm, n_el)
            result = [get_fermion_wise(cartan_obt, u_ops[0])]
            for i in range(len(cartan_tbt)):
                result.append(get_fermion_wise(cartan_tbt[i], u_ops[i+1]))
            return result, suggested_samples
        elif method == 'fff-lr':
            
            all_ops, u_ops, cartan_obt, cartan_tbt, suggested_samples = do_fff(h_ferm, n_el, options=options, metric_estim=False)
            result = [get_fermion_wise(cartan_obt, u_ops[0])]
            for i in range(len(cartan_tbt)):
                result.append(get_fermion_wise(cartan_tbt[i], u_ops[i+1]))
            return result, suggested_samples

def _reverse_jordan_wigner(H):
    '''
    Default funcion used by fermionic methods to obtain the H in the second quantized form.
    The user can specify a different reverse function if H was obtained from the fermionic 
    Hamiltonian using a mapping different than the JordanWigner transformation.
    '''
    return reverse_jordan_wigner(H.to_openfermion())

def _compile_commuting_parts_zb(H, *args, **kwargs):
    # @ Zack add main function here and rest in this file
    # should return list of commuting Hamiltonians in Z-Form and Circuits
    # i.e. result = [(H,U), (H,U), ...]

    binary_H = BinaryHamiltonian.init_from_qubit_hamiltonian(H)
    commuting_parts, suggested_samples = binary_H.commuting_groups(*args, **kwargs)
    circuits = []
    qubit_wise_commuting = []
    for i in range(len(commuting_parts)):
        qwc, U = commuting_parts[i].get_qubit_wise()
        qubit_wise_commuting.append(qwc)
        openqasmcode = tq.export_open_qasm(U)
        new_circuit = tq.import_open_qasm(openqasmcode)
        circuits.append(new_circuit)

    Tableaus, phase_stab, phase_destab, circuits, Tests = Tableau_algorithm(circuits)

    rotations = []
    for i in range(len(commuting_parts)):
        rotations.append(tuple((qubit_wise_commuting[i], circuits[i])))

    return rotations, suggested_samples


def Tableau_algorithm(circuits):
    """Implements the tabluea algorithm for a unitary stabilizer circuit (thomsons circuit) and then
    synthesizes an improved circuit by using the canonical form theorem as laid out in
     "Improved Simulation of Stabilizer Circuits" """
    final_circuits = []
    final_tableaus = []
    final_stabilizer_phases = []
    final_destabilizer_phases = []
    Tests = []

    for U in circuits:
        ## if circuit is empty, do nothing and return no circuit -- already all Z
        if len(U.gates) == 0:
            final_circuits.append(U)
            final_tableaus.append(None)
            final_stabilizer_phases.append(None)
            final_destabilizer_phases.append(None)
            Tests.append(U)

        # If there is a circuit, run the tableau algorithm to bring to canonical form
        # where the algorithm consists of 11 blocks of gates : H-C-P-C-P-C-H-P-C-P-C
        # From https://arxiv.org/abs/quant-ph/0406196
        else:
            num_qubits = U.n_qubits
            Tableau, Phase_stabilizer, Phase_destabilizer, Circuit = initial_tableau(U, U.n_qubits)

            Tableau, Phase_stabilizer, Phase_destabilizer, Q0 = first_round_hadamard(Tableau, Phase_stabilizer,
                                                                                     Phase_destabilizer)  # H
            Tableau, Phase_stabilizer, Phase_destabilizer, Q1 = first_round_cnot(Tableau, Phase_stabilizer,
                                                                                 Phase_destabilizer)  # C
            Tableau, Phase_stabilizer, Phase_destabilizer, Q2 = first_round_phase(Tableau, Phase_stabilizer,
                                                                                  Phase_destabilizer)  # P
            Tableau, Phase_stabilizer, Phase_destabilizer, Q3 = second_round_cnot(Tableau, Phase_stabilizer,
                                                                                  Phase_destabilizer)  # C
            Tableau, Phase_stabilizer, Phase_destabilizer, Q4 = second_round_phase(Tableau, Phase_stabilizer,
                                                                                   Phase_destabilizer)  # P
            Tableau, Phase_stabilizer, Phase_destabilizer, Q5 = third_round_cnot(Tableau, Phase_stabilizer,
                                                                                 Phase_destabilizer)  # C
            Tableau, Phase_stabilizer, Phase_destabilizer, Q6 = second_round_hadamard(Tableau, Phase_stabilizer,
                                                                                      Phase_destabilizer)  # H
            Tableau, Phase_stabilizer, Phase_destabilizer, Q7 = third_round_phase(Tableau, Phase_stabilizer,
                                                                                  Phase_destabilizer)  # P
            Tableau, Phase_stabilizer, Phase_destabilizer, Q8 = fourth_round_cnot(Tableau, Phase_stabilizer,
                                                                                  Phase_destabilizer)  # C
            Tableau, Phase_stabilizer, Phase_destabilizer, Q9 = fourth_round_phase(Tableau, Phase_stabilizer,
                                                                                   Phase_destabilizer)  # P
            Tableau, Phase_stabilizer, Phase_destabilizer, Q10 = final_round_cnot(Tableau, Phase_stabilizer,
                                                                                  Phase_destabilizer)  # C
            # now the code will optimize the CNOT segments
            # as put forth in https://arxiv.org/abs/quant-ph/0302002

            # optimizing CNOT segments
            # optimizing first CNOT round
            if not len(Q1) == 0:
                CNOT_segment_1 = CNOT_matrix(Q1, num_qubits)
                CNOT_segment_1 = optimize_circuit(CNOT_segment_1)
            else:
                CNOT_segment_1 = []

            ##optimizing the second CNOTround
            if not len(Q3) == 0:
                CNOT_segment_2 = CNOT_matrix(Q3, num_qubits)
                CNOT_segment_2 = optimize_circuit(CNOT_segment_2)
            else:
                CNOT_segment_2 = []

            ##optimizing the third CNOT round
            if not len(Q5) == 0:
                CNOT_segment_3 = CNOT_matrix(Q5, num_qubits)
                CNOT_segment_3 = optimize_circuit(CNOT_segment_3)
            else:
                CNOT_segment_3 = []

            ##optimizing the fourth CNOT round
            if not len(Q8) == 0:
                CNOT_segment_4 = CNOT_matrix(Q8, num_qubits)
                CNOT_segment_4 = optimize_circuit(CNOT_segment_4)
            else:
                CNOT_segment_4 = []

            ##optimizing the third CNOT round
            if not len(Q10) == 0:
                CNOT_segment_5 = CNOT_matrix(Q10, num_qubits)
                CNOT_segment_5 = optimize_circuit(CNOT_segment_5)
            else:
                CNOT_segment_5 = []

            # Putting together the final circuit
            # putting together final circuit
            inverse_circuit = tq.QCircuit()

            for i in Q0:
                for j in i.gates:
                    inverse_circuit += j  # H

            inverse_circuit += CNOT_segment_1  # C

            for i in Q2:
                for j in i.gates:
                    inverse_circuit += j  # P

            inverse_circuit += CNOT_segment_2  # C

            for i in Q4:
                for j in i.gates:
                    inverse_circuit += j  # P

            inverse_circuit += CNOT_segment_3  # C

            for i in Q6:
                for j in i.gates:
                    inverse_circuit += j  # H

            for i in Q7:
                for j in i.gates:
                    inverse_circuit += j  # P

            inverse_circuit += CNOT_segment_4  # C

            for i in Q9:
                for j in i.gates:
                    inverse_circuit += j  # P

            inverse_circuit += CNOT_segment_5  # C

            # Actually synthesized inverse circuit, so need to reverse gates

            final_circuit = tq.circuit.QCircuit()
            for gate in reversed(inverse_circuit.gates):
                final_circuit += gate

            final_tableaus.append(Tableau)
            final_stabilizer_phases.append(Phase_stabilizer)
            final_destabilizer_phases.append(Phase_destabilizer)
            final_circuits.append(final_circuit)
            Tests.append(Circuit)

    return (final_tableaus, final_stabilizer_phases, final_destabilizer_phases, final_circuits, Tests)


def CNOT_matrix(list_of_gates, num_qubits):
    """This function represents the CNOT segments as a single unitary matrix, and this matrix will be used for optimizing
    CNOT gate depth as put forth by igor patel and hayes - Efficient Synthesis of Linear Reversible Circuits"""
    matrices = []
    ##the CNOT circuit soptimization does not work with odd number of qubits, so making the CNOT matrix
    # have an even dimension seems to be a workaround
    if not (num_qubits % 2 == 0):
        num_qubits += 1
    for U in reversed(list_of_gates):
        gate = np.identity(num_qubits)
        for i in U.gates:
            target_qubit = i.target[0]
            control_qubit = i.control[0]
        gate[target_qubit, control_qubit] = 1
        matrices.append(gate)
    C_NOT_matrix = matrices[0]
    for i, j in enumerate(matrices):
        if i == 0:
            continue
        else:
            C_NOT_matrix = np.matmul(C_NOT_matrix, j) % 2

    return (C_NOT_matrix)


def optimize_circuit(CNOT_matrix):
    """CNOT circuit optimization as put forth in paper by igor, patel and hayes -
    Efficient Synthesis of Linear Reversible Circuits"""
    CNOT_matrix, circuit1 = Lwr_CNOT_Synth(CNOT_matrix)
    CNOT_matrix = np.transpose(CNOT_matrix)
    CNOT_matrix, circuit2 = Lwr_CNOT_Synth(CNOT_matrix)
    optimized_circuit = tq.QCircuit()
    ## putting together the two circuits

    # the CNOT gates in circuit two have the control and target qubits swapped
    for U in (circuit2):
        control_qubit = U[1]
        target_qubit = U[0]
        optimized_circuit += tq.gates.CNOT(control_qubit, target_qubit)

    ## the order of CNOT gates in circuit one are reversed
    for U in reversed(circuit1):
        control_qubit = U[0]
        target_qubit = U[1]
        optimized_circuit += tq.gates.CNOT(control_qubit, target_qubit)

    return (optimized_circuit)


def Lwr_CNOT_Synth(C_NOT_matrix):
    """CNOT circuit optimization as put forth in paper by igor, patel and hayes"""
    num_qubits = len(C_NOT_matrix[0, :])
    iter = 2
    m = 0
    n = np.shape(C_NOT_matrix)[0]
    circ = []
    partitions = np.hsplit(C_NOT_matrix, n / iter)
    for sec in range(len(partitions)):
        # eliminate duplicate rows
        for j in range(m, m + iter):
            if np.all(C_NOT_matrix[j, m:m + iter] == 0):
                break
            for k in range(0, num_qubits):
                if k < m + iter:
                    continue
                if np.array_equal(C_NOT_matrix[j, m:m + iter], C_NOT_matrix[k, m:m + iter]):
                    target_qubit = k
                    control_qubit = j
                    C_NOT_matrix[target_qubit, :] = (C_NOT_matrix[control_qubit, :] + C_NOT_matrix[target_qubit, :]) % 2
                    circ.append([control_qubit, target_qubit])
        # put ones on the diagonal and eliminate ones on rows below
        for j in range(m, m + iter):
            diag_one = 1
            if C_NOT_matrix[j, j] == 0:
                diag_one = 0
            for row in range(j + 1, num_qubits):
                if C_NOT_matrix[row, j] == 1:
                    if diag_one == 0:
                        target_qubit = j
                        control_qubit = row
                        C_NOT_matrix[target_qubit, :] = (C_NOT_matrix[target_qubit, :] + C_NOT_matrix[control_qubit,
                                                                                         :]) % 2
                        circ.append([control_qubit, target_qubit])
                        diag_one = 1
                    target_qubit = row
                    control_qubit = j
                    C_NOT_matrix[target_qubit, :] = (C_NOT_matrix[target_qubit, :] + C_NOT_matrix[control_qubit, :]) % 2
                    circ.append([control_qubit, target_qubit])

        m += iter
    return (C_NOT_matrix, circ)

def is_canonical(tableau, phase_stab, phase_destab):
    dim = len(tableau)
    eye = np.identity(dim)
    if np.all(np.equal(tableau, eye)) and np.all((phase_stab == 0)) and np.all((phase_destab == 0)):
        return True
    else:
        return False

def initial_tableau(circuit, number_of_qubits):
    """Goes through Thomsons circuit (expressed only in H,CNOT and S gates) to update the standard initial tableau
    as a 2*n X 2*n binary matrix. The final tableau is a binary representation of the circuit"""
    n = number_of_qubits
    a = np.identity(2 * n)
    x_destab = a[0:n, 0:n]
    z_destab = a[0:n, n:2 * n]
    x_stab = a[n:2 * n, 0:n]
    z_stab = a[n:2 * n, n:2 * n]
    phase_destabilizer = np.zeros(n)
    phase_stabilizer = np.zeros(n)
    test_circuit = tq.circuit.QCircuit()

    for U in (circuit.gates):

        if U.name == 'Ry':
            if U.parameter > 0:
                # decomposing Ry as S^+ * H * S* H * S (we can only use clifford gates S,H,CNOT)
                target_qubit = U.target[0]
                test_circuit += tq.gates.S(target_qubit)
                for i in range(0, n):
                    phase_destabilizer[i] = int(phase_destabilizer[i]) ^ (
                            int(x_destab[i, target_qubit]) * int(z_destab[i, target_qubit]))
                    phase_stabilizer[i] = int(phase_stabilizer[i]) ^ (
                            int(x_stab[i, target_qubit]) * int(z_stab[i, target_qubit]))
                    z_stab[i, target_qubit] = int(z_stab[i, target_qubit]) ^ int(x_stab[i, target_qubit])
                    z_destab[i, target_qubit] = int(z_destab[i, target_qubit]) ^ int(x_destab[i, target_qubit])

                test_circuit += tq.gates.H(target_qubit)
                for i in range(0, n):
                    phase_destabilizer[i] = int(phase_destabilizer[i]) ^ (
                            int(x_destab[i, target_qubit]) * int(z_destab[i, target_qubit]))
                    phase_stabilizer[i] = int(phase_stabilizer[i]) ^ (
                            int(x_stab[i, target_qubit]) * int(z_stab[i, target_qubit]))
                    x_destab[i, target_qubit], z_destab[i, target_qubit] = z_destab[i, target_qubit], x_destab[
                        i, target_qubit]
                    z_stab[i, target_qubit], x_stab[i, target_qubit] = x_stab[i, target_qubit], z_stab[i, target_qubit]

                for k in range(3):
                    test_circuit += tq.gates.S(target_qubit)
                    for i in range(0, n):
                        phase_destabilizer[i] = int(phase_destabilizer[i]) ^ (
                                int(x_destab[i, target_qubit]) * int(z_destab[i, target_qubit]))
                        phase_stabilizer[i] = int(phase_stabilizer[i]) ^ (
                                int(x_stab[i, target_qubit]) * int(z_stab[i, target_qubit]))
                        z_stab[i, target_qubit] = int(z_stab[i, target_qubit]) ^ int(x_stab[i, target_qubit])
                        z_destab[i, target_qubit] = int(z_destab[i, target_qubit]) ^ int(x_destab[i, target_qubit])

                test_circuit += tq.gates.H(target_qubit)
                for i in range(0, n):
                    phase_destabilizer[i] = int(phase_destabilizer[i]) ^ (
                            int(x_destab[i, target_qubit]) * int(z_destab[i, target_qubit]))
                    phase_stabilizer[i] = int(phase_stabilizer[i]) ^ (
                            int(x_stab[i, target_qubit]) * int(z_stab[i, target_qubit]))
                    x_destab[i, target_qubit], z_destab[i, target_qubit] = z_destab[i, target_qubit], x_destab[
                        i, target_qubit]
                    z_stab[i, target_qubit], x_stab[i, target_qubit] = x_stab[i, target_qubit], z_stab[i, target_qubit]

                for k in range(3):
                    test_circuit += tq.gates.S(target_qubit)
                    for i in range(0, n):
                        phase_destabilizer[i] = int(phase_destabilizer[i]) ^ (
                                int(x_destab[i, target_qubit]) * int(z_destab[i, target_qubit]))
                        phase_stabilizer[i] = int(phase_stabilizer[i]) ^ (
                                int(x_stab[i, target_qubit]) * int(z_stab[i, target_qubit]))
                        z_stab[i, target_qubit] = int(z_stab[i, target_qubit]) ^ int(x_stab[i, target_qubit])
                        z_destab[i, target_qubit] = int(z_destab[i, target_qubit]) ^ int(x_destab[i, target_qubit])

            else:
                target_qubit = U.target[0]
                test_circuit += tq.gates.S(target_qubit)
                for i in range(0, n):
                    phase_destabilizer[i] = int(phase_destabilizer[i]) ^ (
                            int(x_destab[i, target_qubit]) * int(z_destab[i, target_qubit]))
                    phase_stabilizer[i] = int(phase_stabilizer[i]) ^ (
                            int(x_stab[i, target_qubit]) * int(z_stab[i, target_qubit]))
                    z_stab[i, target_qubit] = int(z_stab[i, target_qubit]) ^ int(x_stab[i, target_qubit])
                    z_destab[i, target_qubit] = int(z_destab[i, target_qubit]) ^ int(x_destab[i, target_qubit])

                test_circuit += tq.gates.H(target_qubit)
                for i in range(0, n):
                    phase_destabilizer[i] = int(phase_destabilizer[i]) ^ (
                            int(x_destab[i, target_qubit]) * int(z_destab[i, target_qubit]))
                    phase_stabilizer[i] = int(phase_stabilizer[i]) ^ (
                            int(x_stab[i, target_qubit]) * int(z_stab[i, target_qubit]))
                    x_destab[i, target_qubit], z_destab[i, target_qubit] = z_destab[i, target_qubit], x_destab[
                        i, target_qubit]
                    z_stab[i, target_qubit], x_stab[i, target_qubit] = x_stab[i, target_qubit], z_stab[i, target_qubit]

                test_circuit += tq.gates.S(target_qubit)
                for i in range(0, n):
                    phase_destabilizer[i] = int(phase_destabilizer[i]) ^ (
                            int(x_destab[i, target_qubit]) * int(z_destab[i, target_qubit]))
                    phase_stabilizer[i] = int(phase_stabilizer[i]) ^ (
                            int(x_stab[i, target_qubit]) * int(z_stab[i, target_qubit]))
                    z_stab[i, target_qubit] = int(z_stab[i, target_qubit]) ^ int(x_stab[i, target_qubit])
                    z_destab[i, target_qubit] = int(z_destab[i, target_qubit]) ^ int(x_destab[i, target_qubit])

                test_circuit += tq.gates.H(target_qubit)
                for i in range(0, n):
                    phase_destabilizer[i] = int(phase_destabilizer[i]) ^ (
                            int(x_destab[i, target_qubit]) * int(z_destab[i, target_qubit]))
                    phase_stabilizer[i] = int(phase_stabilizer[i]) ^ (
                            int(x_stab[i, target_qubit]) * int(z_stab[i, target_qubit]))
                    x_destab[i, target_qubit], z_destab[i, target_qubit] = z_destab[i, target_qubit], x_destab[
                        i, target_qubit]
                    z_stab[i, target_qubit], x_stab[i, target_qubit] = x_stab[i, target_qubit], z_stab[
                        i, target_qubit]

                for k in range(3):
                    test_circuit += tq.gates.S(target_qubit)
                    for i in range(0, n):
                        phase_destabilizer[i] = int(phase_destabilizer[i]) ^ (
                                int(x_destab[i, target_qubit]) * int(z_destab[i, target_qubit]))
                        phase_stabilizer[i] = int(phase_stabilizer[i]) ^ (
                                int(x_stab[i, target_qubit]) * int(z_stab[i, target_qubit]))
                        z_stab[i, target_qubit] = int(z_stab[i, target_qubit]) ^ int(x_stab[i, target_qubit])
                        z_destab[i, target_qubit] = int(z_destab[i, target_qubit]) ^ int(x_destab[i, target_qubit])

        if U.name == 'Phase':
            target_qubit = U.target[0]
            test_circuit += tq.gates.S(target_qubit)
            for i in range(0, n):
                phase_destabilizer[i] = int(phase_destabilizer[i]) ^ (
                        int(x_destab[i, target_qubit]) * int(z_destab[i, target_qubit]))
                phase_stabilizer[i] = int(phase_stabilizer[i]) ^ (
                        int(x_stab[i, target_qubit]) * int(z_stab[i, target_qubit]))
                z_stab[i, target_qubit] = int(z_stab[i, target_qubit]) ^ int(x_stab[i, target_qubit])
                z_destab[i, target_qubit] = int(z_destab[i, target_qubit]) ^ int(x_destab[i, target_qubit])

        # decomposing Rx as H*S^3*H, (we can only use clifford gates S,H,CNOT)
        if U.name == 'Rx':
            if U.parameter > 0:

                target_qubit = U.target[0]
                test_circuit += tq.gates.H(target_qubit)
                for i in range(0, n):
                    phase_destabilizer[i] = int(phase_destabilizer[i]) ^ (
                            int(x_destab[i, target_qubit]) * int(z_destab[i, target_qubit]))
                    phase_stabilizer[i] = int(phase_stabilizer[i]) ^ (
                            int(x_stab[i, target_qubit]) * int(z_stab[i, target_qubit]))
                    x_destab[i, target_qubit], z_destab[i, target_qubit] = z_destab[i, target_qubit], x_destab[
                        i, target_qubit]
                    z_stab[i, target_qubit], x_stab[i, target_qubit] = x_stab[i, target_qubit], z_stab[i, target_qubit]

                test_circuit += tq.gates.S(target_qubit)
                for i in range(0, n):
                    phase_destabilizer[i] = int(phase_destabilizer[i]) ^ (
                            int(x_destab[i, target_qubit]) * int(z_destab[i, target_qubit]))
                    phase_stabilizer[i] = int(phase_stabilizer[i]) ^ (
                            int(x_stab[i, target_qubit]) * int(z_stab[i, target_qubit]))
                    z_stab[i, target_qubit] = int(z_stab[i, target_qubit]) ^ int(x_stab[i, target_qubit])
                    z_destab[i, target_qubit] = int(z_destab[i, target_qubit]) ^ int(x_destab[i, target_qubit])

                test_circuit += tq.gates.H(target_qubit)
                for i in range(0, n):
                    phase_destabilizer[i] = int(phase_destabilizer[i]) ^ (
                            int(x_destab[i, target_qubit]) * int(z_destab[i, target_qubit]))
                    phase_stabilizer[i] = int(phase_stabilizer[i]) ^ (
                            int(x_stab[i, target_qubit]) * int(z_stab[i, target_qubit]))
                    x_destab[i, target_qubit], z_destab[i, target_qubit] = z_destab[i, target_qubit], x_destab[
                        i, target_qubit]
                    z_stab[i, target_qubit], x_stab[i, target_qubit] = x_stab[i, target_qubit], z_stab[i, target_qubit]

            else:
                target_qubit = U.target[0]
                test_circuit += tq.gates.H(target_qubit)
                for i in range(0, n):
                    phase_destabilizer[i] = int(phase_destabilizer[i]) ^ (
                            int(x_destab[i, target_qubit]) * int(z_destab[i, target_qubit]))
                    phase_stabilizer[i] = int(phase_stabilizer[i]) ^ (
                            int(x_stab[i, target_qubit]) * int(z_stab[i, target_qubit]))
                    x_destab[i, target_qubit], z_destab[i, target_qubit] = z_destab[i, target_qubit], x_destab[
                        i, target_qubit]
                    z_stab[i, target_qubit], x_stab[i, target_qubit] = x_stab[i, target_qubit], z_stab[i, target_qubit]

                for j in range(3):
                    test_circuit += tq.gates.S(target_qubit)
                    for i in range(0, n):
                        phase_destabilizer[i] = int(phase_destabilizer[i]) ^ (
                                int(x_destab[i, target_qubit]) * int(z_destab[i, target_qubit]))
                        phase_stabilizer[i] = int(phase_stabilizer[i]) ^ (
                                int(x_stab[i, target_qubit]) * int(z_stab[i, target_qubit]))
                        z_stab[i, target_qubit] = int(z_stab[i, target_qubit]) ^ int(x_stab[i, target_qubit])
                        z_destab[i, target_qubit] = int(z_destab[i, target_qubit]) ^ int(x_destab[i, target_qubit])

                test_circuit += tq.gates.H(target_qubit)
                for i in range(0, n):
                    phase_destabilizer[i] = int(phase_destabilizer[i]) ^ (
                            int(x_destab[i, target_qubit]) * int(z_destab[i, target_qubit]))
                    phase_stabilizer[i] = int(phase_stabilizer[i]) ^ (
                            int(x_stab[i, target_qubit]) * int(z_stab[i, target_qubit]))
                    x_destab[i, target_qubit], z_destab[i, target_qubit] = z_destab[i, target_qubit], x_destab[
                        i, target_qubit]
                    z_stab[i, target_qubit], x_stab[i, target_qubit] = x_stab[i, target_qubit], z_stab[i, target_qubit]

        if U.name == 'Rz':
            if U.parameter > 0:
                target_qubit = U.target[0]
                test_circuit += tq.gates.S(target_qubit)
                for i in range(0, n):
                    phase_destabilizer[i] = int(phase_destabilizer[i]) ^ (
                            int(x_destab[i, target_qubit]) * int(z_destab[i, target_qubit]))
                    phase_stabilizer[i] = int(phase_stabilizer[i]) ^ (
                            int(x_stab[i, target_qubit]) * int(z_stab[i, target_qubit]))
                    z_stab[i, target_qubit] = int(z_stab[i, target_qubit]) ^ int(x_stab[i, target_qubit])
                    z_destab[i, target_qubit] = int(z_destab[i, target_qubit]) ^ int(x_destab[i, target_qubit])
            else:
                target_qubit = U.target[0]
                for k in range(0, 3):
                    test_circuit += tq.gates.S(target_qubit)
                    for i in range(0, n):
                        phase_destabilizer[i] = int(phase_destabilizer[i]) ^ (
                                int(x_destab[i, target_qubit]) * int(z_destab[i, target_qubit]))
                        phase_stabilizer[i] = int(phase_stabilizer[i]) ^ (
                                int(x_stab[i, target_qubit]) * int(z_stab[i, target_qubit]))
                        z_stab[i, target_qubit] = int(z_stab[i, target_qubit]) ^ int(x_stab[i, target_qubit])
                        z_destab[i, target_qubit] = int(z_destab[i, target_qubit]) ^ int(x_destab[i, target_qubit])

        if U.name == 'H':

            target_qubit = U.target[0]
            test_circuit += tq.gates.H(target_qubit)
            for i in range(0, n):
                phase_destabilizer[i] = int(phase_destabilizer[i]) ^ (
                        int(x_destab[i, target_qubit]) * int(z_destab[i, target_qubit]))
                phase_stabilizer[i] = int(phase_stabilizer[i]) ^ (
                        int(x_stab[i, target_qubit]) * int(z_stab[i, target_qubit]))
                x_destab[i, target_qubit], z_destab[i, target_qubit] = z_destab[i, target_qubit], x_destab[
                    i, target_qubit]
                z_stab[i, target_qubit], x_stab[i, target_qubit] = x_stab[i, target_qubit], z_stab[i, target_qubit]

        if U.name == 'X':
            target_qubit = U.target[0]
            control_qubit = U.control[0]
            test_circuit += tq.gates.CNOT(control_qubit, target_qubit)
            for i in range(0, n):
                phase_destabilizer[i] = int(phase_destabilizer[i]) ^ (
                        (int(x_destab[i, control_qubit]) * int(z_destab[i, target_qubit]))
                        * (int(x_destab[i, target_qubit]) ^ int(z_destab[i, control_qubit]) ^ 1))
                phase_stabilizer[i] = int(phase_stabilizer[i]) ^ (
                        (int(x_stab[i, control_qubit]) * int(z_stab[i, target_qubit]))
                        * (int(x_stab[i, target_qubit]) ^ int(z_stab[i, control_qubit]) ^ 1))
                x_stab[i, target_qubit] = int(x_stab[i, target_qubit]) ^ int(x_stab[i, control_qubit])
                x_destab[i, target_qubit] = int(x_destab[i, target_qubit]) ^ int(x_destab[i, control_qubit])
                z_stab[i, control_qubit] = int(z_stab[i, control_qubit]) ^ int(z_stab[i, target_qubit])
                z_destab[i, control_qubit] = int(z_destab[i, control_qubit]) ^ int(z_destab[i, target_qubit])

    destabilizer = np.concatenate((x_destab, z_destab), axis=1)
    stabilizer = np.concatenate((x_stab, z_stab), axis=1)
    tableau = np.concatenate((destabilizer, stabilizer), axis=0)
    return (tableau, phase_stabilizer, phase_destabilizer, test_circuit)


def first_round_hadamard(tableau, phase_stabilizer, phase_destabilizer):
    """Hadamard gates makes the X stabilizer matrix of the tableau have full rank """
    num_qubits = int(len(tableau[0, :]) / 2)
    circ = []
    x_destab = tableau[0:num_qubits, 0:num_qubits]
    z_destab = tableau[0:num_qubits, num_qubits:2 * num_qubits]
    z_stab = tableau[num_qubits:2 * num_qubits, num_qubits:2 * num_qubits]
    x_stab = tableau[num_qubits:2 * num_qubits, 0:num_qubits]
    rank = npl.matrix_rank(x_stab)

    if is_canonical(tableau, phase_stabilizer, phase_destabilizer):
        return (tableau, phase_stabilizer, phase_destabilizer, circ)
    else:
        if rank != len(x_stab[0, :]):
            y = copy.deepcopy(x_stab)
            row_echelon_x = RREF_binary(y)
            row_echelon_x = np.transpose(row_echelon_x)
            # finding dependent rows, these are the bits to flip with the hadamard gates
            for k in range(len(row_echelon_x[:, 0])):
                for column, l in enumerate(row_echelon_x[k, :]):
                    if l == 1:
                        for m in range(k + 1, len(row_echelon_x[:, 0])):
                            if row_echelon_x[m, column] == 1:
                                row_echelon_x[m, :] = (row_echelon_x[k, :] + row_echelon_x[m, :]) % 2
            columns = np.all((row_echelon_x == 0), axis=1)
            bits_to_flip = []
            for k in range(len(columns)):
                if columns[k]:
                    bits_to_flip.append(k)
            for target_qubit in bits_to_flip:
                circ.append(tq.gates.H(target=target_qubit))
                for i in range(0, num_qubits):
                    phase_destabilizer[i] = int(phase_destabilizer[i]) ^ (
                            int(x_destab[i, target_qubit]) * int(z_destab[i, target_qubit]))
                    phase_stabilizer[i] = int(phase_stabilizer[i]) ^ (
                            int(x_stab[i, target_qubit]) * int(z_stab[i, target_qubit]))
                    x_destab[i, target_qubit], z_destab[i, target_qubit] = z_destab[i, target_qubit], x_destab[
                        i, target_qubit]
                    z_stab[i, target_qubit], x_stab[i, target_qubit] = x_stab[i, target_qubit], z_stab[i, target_qubit]
                destabilizer = np.concatenate((x_destab, z_destab), axis=1)
                stabilizer = np.concatenate((x_stab, z_stab), axis=1)
                tableau = np.concatenate((destabilizer, stabilizer), axis=0)
        else:
            destabilizer = np.concatenate((x_destab, z_destab), axis=1)
            stabilizer = np.concatenate((x_stab, z_stab), axis=1)
            tableau = np.concatenate((destabilizer, stabilizer), axis=0)

        return (tableau, phase_stabilizer, phase_destabilizer, circ)


def first_round_cnot(tableau, phase_stabilizer, phase_destabilizer):
    """CNOT gates make the X stabilizer matrix the identity matrix"""
    num_qubits = int(len(tableau[0, :]) / 2)
    x_destab = tableau[0:num_qubits, 0:num_qubits]
    z_destab = tableau[0:num_qubits, num_qubits:2 * num_qubits]
    z_stab = tableau[num_qubits:2 * num_qubits, num_qubits:2 * num_qubits]
    x_stab = tableau[num_qubits:2 * num_qubits, 0:num_qubits]
    circ = []

    if is_canonical(tableau, phase_stabilizer, phase_destabilizer):
        return (tableau, phase_stabilizer, phase_destabilizer, circ)
    else:
        for i in range(num_qubits):
            # making matrix upper triangular
            ##adding ones on diagonal if 0
            diag_one = 1
            if x_stab[i, i] == 0:
                diag_one = 0
            for j in range(i + 1, num_qubits):
                if x_stab[i, j] == 1:
                    if diag_one == 0:
                        target_qubit = i
                        control_qubit = j
                        circ.append(tq.gates.CNOT(control_qubit, target_qubit))
                        for k in range(0, num_qubits):
                            phase_destabilizer[k] = int(phase_destabilizer[k]) ^ (
                                    (int(x_destab[k, control_qubit]) * int(z_destab[k, target_qubit])) \
                                    * (int(x_destab[k, target_qubit]) ^ int(z_destab[k, control_qubit]) ^ 1))
                            phase_stabilizer[k] = int(phase_stabilizer[k]) ^ (
                                    (int(x_stab[k, control_qubit]) * int(z_stab[k, target_qubit])) \
                                    * (int(x_stab[k, target_qubit]) ^ int(z_stab[k, control_qubit]) ^ 1))
                            x_stab[k, target_qubit] = int(x_stab[k, target_qubit]) ^ int(x_stab[k, control_qubit])
                            x_destab[k, target_qubit] = int(x_destab[k, target_qubit]) ^ int(x_destab[k, control_qubit])
                            z_stab[k, control_qubit] = int(z_stab[k, control_qubit]) ^ int(z_stab[k, target_qubit])
                            z_destab[k, control_qubit] = int(z_destab[k, control_qubit]) ^ int(
                                z_destab[k, target_qubit])

                        # if one on diagnal guassian elimination on rows below
                        diag_one = 1

                        # guassian elimination on lower rows
                    target_qubit = j
                    control_qubit = i

                    circ.append(tq.gates.CNOT(control_qubit, target_qubit))
                    for k in range(0, num_qubits):
                        phase_destabilizer[k] = int(phase_destabilizer[k]) ^ (
                                (int(x_destab[k, control_qubit]) * int(z_destab[k, target_qubit])) \
                                * (int(x_destab[k, target_qubit]) ^ int(z_destab[k, control_qubit]) ^ 1))
                        phase_stabilizer[k] = int(phase_stabilizer[k]) ^ (
                                (int(x_stab[k, control_qubit]) * int(z_stab[k, target_qubit])) \
                                * (int(x_stab[k, target_qubit]) ^ int(z_stab[k, control_qubit]) ^ 1))
                        x_stab[k, target_qubit] = int(x_stab[k, target_qubit]) ^ int(x_stab[k, control_qubit])
                        x_destab[k, target_qubit] = int(x_destab[k, target_qubit]) ^ int(x_destab[k, control_qubit])
                        z_stab[k, control_qubit] = int(z_stab[k, control_qubit]) ^ int(z_stab[k, target_qubit])
                        z_destab[k, control_qubit] = int(z_destab[k, control_qubit]) ^ int(z_destab[k, target_qubit])


        # making matrix identity
        for i in reversed(range(num_qubits)):
            for j in range(i - 1, -1, -1):
                if x_stab[i, j] == 1:
                    target_qubit = j
                    control_qubit = i
                    circ.append(tq.gates.CNOT(control_qubit, target_qubit))
                    for k in range(num_qubits):
                        phase_destabilizer[k] = int(phase_destabilizer[k]) ^ (
                                (int(x_destab[k, control_qubit]) * int(z_destab[k, target_qubit])) \
                                * (int(x_destab[k, target_qubit]) ^ int(z_destab[k, control_qubit]) ^ 1))
                        phase_stabilizer[k] = int(phase_stabilizer[k]) ^ (
                                (int(x_stab[k, control_qubit]) * int(z_stab[k, target_qubit])) \
                                * (int(x_stab[k, target_qubit]) ^ int(z_stab[k, control_qubit]) ^ 1))
                        x_stab[k, target_qubit] = int(x_stab[k, target_qubit]) ^ int(x_stab[k, control_qubit])
                        x_destab[k, target_qubit] = int(x_destab[k, target_qubit]) ^ int(x_destab[k, control_qubit])
                        z_stab[k, control_qubit] = int(z_stab[k, control_qubit]) ^ int(z_stab[k, target_qubit])
                        z_destab[k, control_qubit] = int(z_destab[k, control_qubit]) ^ int(z_destab[k, target_qubit])

        destabilizer = np.concatenate((x_destab, z_destab), axis=1)
        stabilizer = np.concatenate((x_stab, z_stab), axis=1)
        tableau = np.concatenate((destabilizer, stabilizer), axis=0)
        return (tableau, phase_stabilizer, phase_destabilizer, circ)



def first_round_phase(tableau, phase_stabilizer, phase_destabilizer):
    """This phase round adds a diagonal matrix D to the Z stabilizer matrix such that Z + D = M*M' for some
    invertible M"""
    num_qubits = int(len(tableau[0, :]) / 2)
    x_destab = tableau[0:num_qubits, 0:num_qubits]
    z_destab = tableau[0:num_qubits, num_qubits:2 * num_qubits]
    z_stab = tableau[num_qubits:2 * num_qubits, num_qubits:2 * num_qubits]
    x_stab = tableau[num_qubits:2 * num_qubits, 0:num_qubits]
    matrix = copy.deepcopy(z_stab)
    M = np.identity(num_qubits)
    circ = []

    if is_canonical(tableau, phase_stabilizer, phase_destabilizer):
        return (tableau, phase_stabilizer, phase_destabilizer, circ)
    else:
        for j in range(0, num_qubits):
            if j == 0:
                for i in range(1, num_qubits):
                    M[i, j] = matrix[i, j]
                continue
            for i in range(j + 1, num_qubits):
                result = []
                for k in range(0, j):
                    Sum = (M[i, k] * M[j, k]) % 2
                    result.append(Sum)
                final_sum = sum(result) % 2
                M[i, j] = (matrix[i, j] + final_sum) % 2
        matrix = np.matmul(M, np.transpose(M)) % 2

        bits_to_flip = []
        for i in range(int(len(matrix[0, :]))):
            if matrix[i, i] != z_stab[i, i]:
                bits_to_flip.append(i)
            elif matrix[i, i] == z_stab[i, i]:
                continue
        for target_qubit in bits_to_flip:
            circ.append(tq.gates.S(target=target_qubit))
            for i in range(0, num_qubits):
                phase_destabilizer[i] = int(phase_destabilizer[i]) ^ (
                        int(x_destab[i, target_qubit]) * int(z_destab[i, target_qubit]))
                phase_stabilizer[i] = int(phase_stabilizer[i]) ^ (
                        int(x_stab[i, target_qubit]) * int(z_stab[i, target_qubit]))
                z_stab[i, target_qubit] = int(z_stab[i, target_qubit]) ^ int(x_stab[i, target_qubit])
                z_destab[i, target_qubit] = int(z_destab[i, target_qubit]) ^ int(x_destab[i, target_qubit])
        destabilizer = np.concatenate((x_destab, z_destab), axis=1)
        stabilizer = np.concatenate((x_stab, z_stab), axis=1)
        tableau = np.concatenate((destabilizer, stabilizer), axis=0)
        return (tableau, phase_stabilizer, phase_destabilizer, circ)



def second_round_cnot(tableau, phase_stabilizer, phase_destabilizer):
    """performs a cholesky decompostion of the symmetric Z = D + M*M' stabilizer matrix """
    num_qubits = int(len(tableau[0, :]) / 2)
    x_destab = tableau[0:num_qubits, 0:num_qubits]
    z_destab = tableau[0:num_qubits, num_qubits:2 * num_qubits]
    z_stab = tableau[num_qubits:2 * num_qubits, num_qubits:2 * num_qubits]
    x_stab = tableau[num_qubits:2 * num_qubits, 0:num_qubits]
    matrix = copy.deepcopy(z_stab)
    circ = []

    if is_canonical(tableau, phase_stabilizer, phase_destabilizer):
        return (tableau, phase_stabilizer, phase_destabilizer, circ)
    else:
        # Decomposing the symmetric z stabilizer matrix into lower triangular matrix
        M = np.identity(num_qubits)
        for j in range(0, num_qubits):
            if j == 0:
                for i in range(1, num_qubits):
                    M[i, j] = matrix[i, j]
                continue
            for i in range(j + 1, num_qubits):
                result = []
                for k in range(0, j):
                    Sum = (M[i, k] * M[j, k]) % 2
                    result.append(Sum)
                final_sum = sum(result) % 2
                M[i, j] = (matrix[i, j] + final_sum) % 2
        for i in range(1, num_qubits):
            ones = np.where(M[i, :] == 1)[0]
            for j, k in enumerate(ones):
                control_qubit = ones[-1]
                target_qubit = k
                if control_qubit != target_qubit:
                    for l in range(0, num_qubits):
                        phase_destabilizer[l] = int(phase_destabilizer[l]) ^ (
                                (int(x_destab[l, control_qubit]) * int(z_destab[l, target_qubit])) \
                                * (int(x_destab[l, target_qubit]) ^ int(z_destab[l, control_qubit]) ^ 1))
                        phase_stabilizer[l] = int(phase_stabilizer[l]) ^ (
                                (int(x_stab[l, control_qubit]) * int(z_stab[l, target_qubit])) \
                                * (int(x_stab[l, target_qubit]) ^ int(z_stab[l, control_qubit]) ^ 1))
                        x_stab[l, target_qubit] = int(x_stab[l, target_qubit]) ^ int(x_stab[l, control_qubit])
                        x_destab[l, target_qubit] = int(x_destab[l, target_qubit]) ^ int(x_destab[l, control_qubit])
                        z_stab[l, control_qubit] = int(z_stab[l, control_qubit]) ^ int(z_stab[l, target_qubit])
                        z_destab[l, control_qubit] = int(z_destab[l, control_qubit]) ^ int(z_destab[l, target_qubit])
                    circ.append(tq.gates.CNOT(control_qubit, target_qubit))
                else:
                    continue
        destabilizer = np.concatenate((x_destab, z_destab), axis=1)
        stabilizer = np.concatenate((x_stab, z_stab), axis=1)
        tableau = np.concatenate((destabilizer, stabilizer), axis=0)
        return (tableau, phase_stabilizer, phase_destabilizer, circ)



def second_round_phase(tableau, phase_stabilizer, phase_destabilizer):
    """phase on all the qubits eliminates the Z_stabilizer to the Zero Matrix, additional phases set all the stabilizer
    phase bits to zero"""
    num_qubits = int(len(tableau[0, :]) / 2)
    x_destab = tableau[0:num_qubits, 0:num_qubits]
    z_destab = tableau[0:num_qubits, num_qubits:2 * num_qubits]
    z_stab = tableau[num_qubits:2 * num_qubits, num_qubits:2 * num_qubits]
    x_stab = tableau[num_qubits:2 * num_qubits, 0:num_qubits]
    bits_to_flip = []
    circ = []

    if is_canonical(tableau, phase_stabilizer, phase_destabilizer):
        return (tableau, phase_stabilizer, phase_destabilizer, circ)
    else:
        for i in range(num_qubits):
            bits_to_flip.append(i)
        for target_qubit in bits_to_flip:
            circ.append(tq.gates.S(target_qubit))
            for i in range(num_qubits):
                phase_destabilizer[i] = int(phase_destabilizer[i]) ^ (
                        int(x_destab[i, target_qubit]) * int(z_destab[i, target_qubit]))
                phase_stabilizer[i] = int(phase_stabilizer[i]) ^ (
                        int(x_stab[i, target_qubit]) * int(z_stab[i, target_qubit]))
                z_stab[i, target_qubit] = int(z_stab[i, target_qubit]) ^ int(x_stab[i, target_qubit])
                z_destab[i, target_qubit] = int(z_destab[i, target_qubit]) ^ int(x_destab[i, target_qubit])
        # zeroing out remaining stabilizer phase bits
        # setting remaining phase bits to zero
        for i in range(num_qubits):
            flag = 1
            if phase_stabilizer[i] == 1:
                target_qubit = i
                flag = 0
            if flag == 0:
                for j in range(2):
                    circ.append(tq.gates.S(target_qubit))
                    for k in range(num_qubits):
                        phase_destabilizer[k] = int(phase_destabilizer[k]) ^ (
                                int(x_destab[k, target_qubit]) * int(z_destab[k, target_qubit]))
                        phase_stabilizer[k] = int(phase_stabilizer[k]) ^ (
                                int(x_stab[k, target_qubit]) * int(z_stab[k, target_qubit]))
                        z_stab[k, target_qubit] = int(z_stab[k, target_qubit]) ^ int(x_stab[k, target_qubit])
                        z_destab[k, target_qubit] = int(z_destab[k, target_qubit]) ^ int(x_destab[k, target_qubit])
        destabilizer = np.concatenate((x_destab, z_destab), axis=1)
        stabilizer = np.concatenate((x_stab, z_stab), axis=1)
        tableau = np.concatenate((destabilizer, stabilizer), axis=0)
        return (tableau, phase_stabilizer, phase_destabilizer, circ)


def third_round_cnot(tableau, phase_stabilizer, phase_destabilizer):
    """CNOT gates make the X stabilizer matrix the identity matrix"""
    num_qubits = int(len(tableau[0, :]) / 2)
    x_destab = tableau[0:num_qubits, 0:num_qubits]
    z_destab = tableau[0:num_qubits, num_qubits:2 * num_qubits]
    z_stab = tableau[num_qubits:2 * num_qubits, num_qubits:2 * num_qubits]
    x_stab = tableau[num_qubits:2 * num_qubits, 0:num_qubits]
    circ = []

    if is_canonical(tableau, phase_stabilizer, phase_destabilizer):
        return (tableau, phase_stabilizer, phase_destabilizer, circ)
    else:
        for i in range(len(x_stab[:, 0])):
            ones = np.where(x_stab[i, :] == 1)[0]
            if x_stab[i, i] == 1:
                control_qubit = i
                for target_qubit in ones:
                    if control_qubit == target_qubit:
                        continue
                    circ.append(tq.gates.CNOT(control_qubit, target_qubit))
                    for k in range(0, num_qubits):
                        phase_destabilizer[k] = int(phase_destabilizer[k]) ^ (
                                (int(x_destab[k, control_qubit]) * int(z_destab[k, target_qubit])) \
                                * (int(x_destab[k, target_qubit]) ^ int(z_destab[k, control_qubit]) ^ 1))
                        phase_stabilizer[k] = int(phase_stabilizer[k]) ^ (
                                (int(x_stab[k, control_qubit]) * int(z_stab[k, target_qubit])) \
                                * (int(x_stab[k, target_qubit]) ^ int(z_stab[k, control_qubit]) ^ 1))
                        x_stab[k, target_qubit] = int(x_stab[k, target_qubit]) ^ int(x_stab[k, control_qubit])
                        x_destab[k, target_qubit] = int(x_destab[k, target_qubit]) ^ int(x_destab[k, control_qubit])
                        z_stab[k, control_qubit] = int(z_stab[k, control_qubit]) ^ int(z_stab[k, target_qubit])
                        z_destab[k, control_qubit] = int(z_destab[k, control_qubit]) ^ int(z_destab[k, target_qubit])

        destabilizer = np.concatenate((x_destab, z_destab), axis=1)
        stabilizer = np.concatenate((x_stab, z_stab), axis=1)
        tableau = np.concatenate((destabilizer, stabilizer), axis=0)
        return (tableau, phase_stabilizer, phase_destabilizer, circ)



def second_round_hadamard(tableau, phase_stabilizer, phase_destabilizer):
    """"Apply Hadmard on all the bits puts the stabilizer matrix into canonical form"""
    num_qubits = int(len(tableau[0, :]) / 2)
    x_destab = tableau[0:num_qubits, 0:num_qubits]
    z_destab = tableau[0:num_qubits, num_qubits:2 * num_qubits]
    z_stab = tableau[num_qubits:2 * num_qubits, num_qubits:2 * num_qubits]
    x_stab = tableau[num_qubits:2 * num_qubits, 0:num_qubits]
    bits_to_flip = []
    circ = []

    if is_canonical(tableau, phase_stabilizer, phase_destabilizer):
        return (tableau, phase_stabilizer, phase_destabilizer, circ)
    else:
        for i in range(num_qubits):
            bits_to_flip.append(i)
        for target_qubit in bits_to_flip:
            circ.append(tq.gates.H(target=target_qubit))
            for i in range(0, num_qubits):
                phase_destabilizer[i] = int(phase_destabilizer[i]) ^ (
                        int(x_destab[i, target_qubit]) * int(z_destab[i, target_qubit]))
                phase_stabilizer[i] = int(phase_stabilizer[i]) ^ (
                        int(x_stab[i, target_qubit]) * int(z_stab[i, target_qubit]))
                x_destab[i, target_qubit], z_destab[i, target_qubit] = z_destab[i, target_qubit], x_destab[
                    i, target_qubit]
                z_stab[i, target_qubit], x_stab[i, target_qubit] = x_stab[i, target_qubit], z_stab[i, target_qubit]
        destabilizer = np.concatenate((x_destab, z_destab), axis=1)
        stabilizer = np.concatenate((x_stab, z_stab), axis=1)
        tableau = np.concatenate((destabilizer, stabilizer), axis=0)
        return (tableau, phase_stabilizer, phase_destabilizer, circ)



def third_round_phase(tableau, phase_stabilizer, phase_destabilizer):
    """Adds a diagonal matrix to the Z destabilizer matrix such that Zd + D = M*M' for some invertible M"""
    num_qubits = int(len(tableau[0, :]) / 2)
    x_destab = tableau[0:num_qubits, 0:num_qubits]
    z_destab = tableau[0:num_qubits, num_qubits:2 * num_qubits]
    z_stab = tableau[num_qubits:2 * num_qubits, num_qubits:2 * num_qubits]
    x_stab = tableau[num_qubits:2 * num_qubits, 0:num_qubits]
    matrix = copy.deepcopy(z_destab)
    M = np.identity(num_qubits)
    circ = []

    if is_canonical(tableau, phase_stabilizer, phase_destabilizer):
        return (tableau, phase_stabilizer, phase_destabilizer, circ)
    else:
        for j in range(0, num_qubits):
            if j == 0:
                for i in range(1, num_qubits):
                    M[i, j] = matrix[i, j]
                continue
            for i in range(j + 1, num_qubits):
                result = []
                for k in range(0, j):
                    Sum = (M[i, k] * M[j, k]) % 2
                    result.append(Sum)
                final_sum = sum(result) % 2
                M[i, j] = (matrix[i, j] + final_sum) % 2
        matrix = np.matmul(M, np.transpose(M)) % 2
        bits_to_flip = []
        for i in range(int(len(matrix[0, :]))):
            if matrix[i, i] != z_destab[i, i]:
                bits_to_flip.append(i)
            elif matrix[i, i] == z_destab[i, i]:
                continue
        for target_qubit in bits_to_flip:
            circ.append(tq.gates.S(target=target_qubit))
            for i in range(0, num_qubits):
                phase_destabilizer[i] = int(phase_destabilizer[i]) ^ (
                        int(x_destab[i, target_qubit]) * int(z_destab[i, target_qubit]))
                phase_stabilizer[i] = int(phase_stabilizer[i]) ^ (
                        int(x_stab[i, target_qubit]) * int(z_stab[i, target_qubit]))
                z_stab[i, target_qubit] = int(z_stab[i, target_qubit]) ^ int(x_stab[i, target_qubit])
                z_destab[i, target_qubit] = int(z_destab[i, target_qubit]) ^ int(x_destab[i, target_qubit])
        destabilizer = np.concatenate((x_destab, z_destab), axis=1)
        stabilizer = np.concatenate((x_stab, z_stab), axis=1)
        tableau = np.concatenate((destabilizer, stabilizer), axis=0)
        return (tableau, phase_stabilizer, phase_destabilizer, circ)



def fourth_round_cnot(tableau, phase_stabilizer, phase_destabilizer):
    """"Performs a binary cholesky decomposition on the Z destabilizer matrix"""
    num_qubits = int(len(tableau[0, :]) / 2)
    x_destab = tableau[0:num_qubits, 0:num_qubits]
    z_destab = tableau[0:num_qubits, num_qubits:2 * num_qubits]
    z_stab = tableau[num_qubits:2 * num_qubits, num_qubits:2 * num_qubits]
    x_stab = tableau[num_qubits:2 * num_qubits, 0:num_qubits]
    matrix = copy.deepcopy(z_destab)
    M = np.identity(num_qubits)
    circ = []

    if is_canonical(tableau, phase_stabilizer, phase_destabilizer):
        return (tableau, phase_stabilizer, phase_destabilizer, circ)
    else:
        for j in range(0, num_qubits):
            if j == 0:
                for i in range(1, num_qubits):
                    M[i, j] = matrix[i, j]
                continue
            for i in range(j + 1, num_qubits):
                result = []
                for k in range(0, j):
                    Sum = (M[i, k] * M[j, k]) % 2
                    result.append(Sum)
                final_sum = sum(result) % 2
                M[i, j] = (matrix[i, j] + final_sum) % 2
        for i in range(1, num_qubits):
            ones = np.where(M[i, :] == 1)[0]
            for j, k in enumerate(ones):
                control_qubit = ones[-1]
                target_qubit = k
                if control_qubit != target_qubit:
                    for l in range(0, num_qubits):
                        phase_destabilizer[l] = int(phase_destabilizer[l]) ^ (
                                (int(x_destab[l, control_qubit]) * int(z_destab[l, target_qubit])) \
                                * (int(x_destab[l, target_qubit]) ^ int(z_destab[l, control_qubit]) ^ 1))
                        phase_stabilizer[l] = int(phase_stabilizer[l]) ^ (
                                (int(x_stab[l, control_qubit]) * int(z_stab[l, target_qubit])) \
                                * (int(x_stab[l, target_qubit]) ^ int(z_stab[l, control_qubit]) ^ 1))
                        x_stab[l, target_qubit] = int(x_stab[l, target_qubit]) ^ int(x_stab[l, control_qubit])
                        x_destab[l, target_qubit] = int(x_destab[l, target_qubit]) ^ int(x_destab[l, control_qubit])
                        z_stab[l, control_qubit] = int(z_stab[l, control_qubit]) ^ int(z_stab[l, target_qubit])
                        z_destab[l, control_qubit] = int(z_destab[l, control_qubit]) ^ int(z_destab[l, target_qubit])
                    circ.append(tq.gates.CNOT(control_qubit, target_qubit))
                else:
                    continue
        destabilizer = np.concatenate((x_destab, z_destab), axis=1)
        stabilizer = np.concatenate((x_stab, z_stab), axis=1)
        tableau = np.concatenate((destabilizer, stabilizer), axis=0)
        return (tableau, phase_stabilizer, phase_destabilizer, circ)



def fourth_round_phase(tableau, phase_stabilizer, phase_destabilizer):
    """"Phase gates make the Z destabilizer the Zero matrix, additional phases set all the destabilizer
    phase bits to 0"""
    num_qubits = int(len(tableau[0, :]) / 2)
    x_destab = tableau[0:num_qubits, 0:num_qubits]
    z_destab = tableau[0:num_qubits, num_qubits:2 * num_qubits]
    z_stab = tableau[num_qubits:2 * num_qubits, num_qubits:2 * num_qubits]
    x_stab = tableau[num_qubits:2 * num_qubits, 0:num_qubits]
    bits_to_flip = []
    circ = []

    if is_canonical(tableau, phase_stabilizer, phase_destabilizer):
        return (tableau, phase_stabilizer, phase_destabilizer, circ)
    else:
        for i in range(num_qubits):
            bits_to_flip.append(i)
        for j in range(1):
            for target_qubit in bits_to_flip:
                circ.append(tq.gates.S(target_qubit))
                for i in range(num_qubits):
                    phase_destabilizer[i] = int(phase_destabilizer[i]) ^ (
                            int(x_destab[i, target_qubit]) * int(z_destab[i, target_qubit]))
                    phase_stabilizer[i] = int(phase_stabilizer[i]) ^ (
                            int(x_stab[i, target_qubit]) * int(z_stab[i, target_qubit]))
                    z_stab[i, target_qubit] = int(z_stab[i, target_qubit]) ^ int(x_stab[i, target_qubit])
                    z_destab[i, target_qubit] = int(z_destab[i, target_qubit]) ^ int(x_destab[i, target_qubit])

        # setting destabilizer bits to 0
        for i in range(num_qubits):
            flag = 1
            if phase_destabilizer[i] == 1:
                target_qubit = i
                flag = 0
            if flag == 0:
                for j in range(2):
                    circ.append(tq.gates.S(target_qubit))
                    for k in range(num_qubits):
                        phase_destabilizer[k] = int(phase_destabilizer[k]) ^ (
                                int(x_destab[k, target_qubit]) * int(z_destab[k, target_qubit]))
                        phase_stabilizer[k] = int(phase_stabilizer[k]) ^ (
                                int(x_stab[k, target_qubit]) * int(z_stab[k, target_qubit]))
                        z_stab[k, target_qubit] = int(z_stab[k, target_qubit]) ^ int(x_stab[k, target_qubit])
                        z_destab[k, target_qubit] = int(z_destab[k, target_qubit]) ^ int(x_destab[k, target_qubit])

        destabilizer = np.concatenate((x_destab, z_destab), axis=1)
        stabilizer = np.concatenate((x_stab, z_stab), axis=1)
        tableau = np.concatenate((destabilizer, stabilizer), axis=0)
        return (tableau, phase_stabilizer, phase_destabilizer, circ)



def final_round_cnot(tableau, phase_stabilizer, phase_destabilizer):
    """Final round of CNOT gates puts the Tableau into canonical form: a 2n X 2n identity matrix with all phase
    bits zero"""
    num_qubits = int(len(tableau[0, :]) / 2)
    x_destab = tableau[0:num_qubits, 0:num_qubits]
    z_destab = tableau[0:num_qubits, num_qubits:2 * num_qubits]
    z_stab = tableau[num_qubits:2 * num_qubits, num_qubits:2 * num_qubits]
    x_stab = tableau[num_qubits:2 * num_qubits, 0:num_qubits]
    circ = []

    if is_canonical(tableau, phase_stabilizer, phase_destabilizer):
        return (tableau, phase_stabilizer, phase_destabilizer, circ)
    else:
        for i in range(len(x_destab[:, 0])):
            ones = np.where(x_destab[i, :] == 1)[0]
            if x_destab[i, i] == 0:
                control_qubit = ones[-1]
                target_qubit = i
                circ.append(tq.gates.CNOT(control_qubit, target_qubit))
                for k in range(0, num_qubits):
                    phase_destabilizer[k] = int(phase_destabilizer[k]) ^ (
                            (int(x_destab[k, control_qubit]) * int(z_destab[k, target_qubit])) \
                            * (int(x_destab[k, target_qubit]) ^ int(z_destab[k, control_qubit]) ^ 1))
                    phase_stabilizer[k] = int(phase_stabilizer[k]) ^ (
                            (int(x_stab[k, control_qubit]) * int(z_stab[k, target_qubit])) \
                            * (int(x_stab[k, target_qubit]) ^ int(z_stab[k, control_qubit]) ^ 1))
                    x_stab[k, target_qubit] = int(x_stab[k, target_qubit]) ^ int(x_stab[k, control_qubit])
                    x_destab[k, target_qubit] = int(x_destab[k, target_qubit]) ^ int(x_destab[k, control_qubit])
                    z_stab[k, control_qubit] = int(z_stab[k, control_qubit]) ^ int(z_stab[k, target_qubit])
                    z_destab[k, control_qubit] = int(z_destab[k, control_qubit]) ^ int(z_destab[k, target_qubit])

                control_qubit, target_qubit = target_qubit, control_qubit

                circ.append(tq.gates.CNOT(control_qubit, target_qubit))
                for k in range(0, num_qubits):
                    phase_destabilizer[k] = int(phase_destabilizer[k]) ^ (
                            (int(x_destab[k, control_qubit]) * int(z_destab[k, target_qubit])) \
                            * (int(x_destab[k, target_qubit]) ^ int(z_destab[k, control_qubit]) ^ 1))
                    phase_stabilizer[k] = int(phase_stabilizer[k]) ^ (
                            (int(x_stab[k, control_qubit]) * int(z_stab[k, target_qubit])) \
                            * (int(x_stab[k, target_qubit]) ^ int(z_stab[k, control_qubit]) ^ 1))
                    x_stab[k, target_qubit] = int(x_stab[k, target_qubit]) ^ int(x_stab[k, control_qubit])
                    x_destab[k, target_qubit] = int(x_destab[k, target_qubit]) ^ int(x_destab[k, control_qubit])
                    z_stab[k, control_qubit] = int(z_stab[k, control_qubit]) ^ int(z_stab[k, target_qubit])
                    z_destab[k, control_qubit] = int(z_destab[k, control_qubit]) ^ int(z_destab[k, target_qubit])


            if x_destab[i, i] == 1:
                ones = np.where(x_destab[i, :] == 1)[0]
                control_qubit = i
                for target_qubit in ones:
                    if i == target_qubit:
                        continue
                    circ.append(tq.gates.CNOT(control_qubit, target_qubit))
                    for k in range(0, num_qubits):
                        phase_destabilizer[k] = int(phase_destabilizer[k]) ^ (
                                (int(x_destab[k, control_qubit]) * int(z_destab[k, target_qubit])) \
                                * (int(x_destab[k, target_qubit]) ^ int(z_destab[k, control_qubit]) ^ 1))
                        phase_stabilizer[k] = int(phase_stabilizer[k]) ^ (
                                (int(x_stab[k, control_qubit]) * int(z_stab[k, target_qubit])) \
                                * (int(x_stab[k, target_qubit]) ^ int(z_stab[k, control_qubit]) ^ 1))
                        x_stab[k, target_qubit] = int(x_stab[k, target_qubit]) ^ int(x_stab[k, control_qubit])
                        x_destab[k, target_qubit] = int(x_destab[k, target_qubit]) ^ int(x_destab[k, control_qubit])
                        z_stab[k, control_qubit] = int(z_stab[k, control_qubit]) ^ int(z_stab[k, target_qubit])
                        z_destab[k, control_qubit] = int(z_destab[k, control_qubit]) ^ int(z_destab[k, target_qubit])
        destabilizer = np.concatenate((x_destab, z_destab), axis=1)
        stabilizer = np.concatenate((x_stab, z_stab), axis=1)
        tableau = np.concatenate((destabilizer, stabilizer), axis=0)
        return (tableau, phase_stabilizer, phase_destabilizer, circ)



#
def REF_binary(matrix):
    A = matrix
    n_rows, n_cols = np.shape(A)
    # Compute row echelon form (REF)
    current_row = 0
    for j in range(n_cols):  # For each column
        if current_row >= n_rows:
            break

        pivot_row = current_row

        while pivot_row < n_rows and A[pivot_row, j] == 0:
            pivot_row += 1

        if pivot_row == n_rows:
            continue

        A[[current_row, pivot_row]] = A[[pivot_row, current_row]]

        pivot_row = current_row
        current_row += 1

        # Eliminate rows below
        for i in range(current_row, n_rows):
            if A[i, j] == 1:
                A[i] = (A[i] + A[pivot_row]) % 2
    return (A)


#
def RREF_binary(matrix):
    """Converts a list of matrices to reduced row echelon form (RREF)"""
    n_rows, n_cols = np.shape(matrix)
    A = matrix

    # Compute row echelon form (REF)
    current_row = 0
    for j in range(n_cols):
        if current_row >= n_rows:
            break

        pivot_row = current_row
        while pivot_row < n_rows and A[pivot_row, j] == 0:
            pivot_row += 1

        if pivot_row == n_rows:
            continue

        A[[current_row, pivot_row]] = A[[pivot_row, current_row]]

        pivot_row = current_row
        current_row += 1

        for i in range(current_row, n_rows):
            if A[i, j] == 1:
                A[i] = (A[i] + A[pivot_row]) % 2

    for i in reversed(range(current_row)):
        pivot_col = 0

        while pivot_col < n_cols and A[i, pivot_col] == 0:
            pivot_col += 1
        if pivot_col == n_cols:
            continue

        for j in range(i):
            if A[j, pivot_col] == 1:
                A[j] = (A[j] + A[i]) % 2

    for i in range(len(A)):
        A[i] = np.mod(A[i], 2)

    return A

    raise NotImplementedError
