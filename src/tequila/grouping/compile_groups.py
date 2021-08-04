from tequila.grouping.binary_rep import BinaryHamiltonian, BinaryPauliString
import tequila as tq
from tequila import TequilaException
from tequila.hamiltonian import QubitHamiltonian, paulis, PauliString
import tequila.grouping.binary_utils as bu
import numpy as np
import numpy.linalg as npl
import copy

def compile_commuting_parts(H, method="zb", *args, **kwargs):
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
    if method is None or method.lower() == "zb":
        # @ Zack
        return _compile_commuting_parts_zm(H, *args, **kwargs)
    else:
        # original implementation of Thomson (T.C. Yen)
        binary_H = BinaryHamiltonian.init_from_qubit_hamiltonian(H)
        commuting_parts = binary_H.commuting_groups()
        return [cH.get_qubit_wise() for cH in commuting_parts]

def _compile_commuting_parts_zb(H):
    # @ Zack add main function here and rest in this file
    # should return list of commuting Hamiltonians in Z-Form and Circuits
    # i.e. result = [(H,U), (H,U), ...]
    
    binary_H = BinaryHamiltonian.init_from_qubit_hamiltonian(H)
    commuting_parts = binary_H.commuting_groups()
    rotations = rotation_circuit(commuting_parts)
    
    return (rotations)


def rotation_circuit(commuting_groups):
    """returns all z paulis and corresponding circuit that transforms commuting to QWC """

    # generates initial binary matrices of commuting pauli words
    binaries = binary_groups(commuting_groups)
    binary_matrices = initial_binary_matrices(binaries)
    independent_matrices = []

    # reduces pauli binary matrices to the matrix of linearly independent paulis
    for i in range(len(binary_matrices)):
        independent_matrices.append(independent_paulis(binary_matrices[i]))
    Result = []

    # bringing stabilizer matrices to canonical form via the 7 block sequence:
    # H-C-P-C-P-C-H as discussed by Gottesman
    for i in range(len(independent_matrices)):
        num_qubits = int(len(independent_matrices[i][:, 0]) / 2)
        A, B = Hadamard_phase(independent_matrices[i])
        C, D = C_NOT(A)
        E = basis_extension(C)
        F, G = C_NOT(E)
        H, I = phase_round(F)
        J, K = second_round_cnot(H)
        L, M = second_round_phase(J)
        N, O = C_NOT(L)
        P, Q = final_hadamard(N)  #the Q stands for Quantum
        x_stab = np.split(binary_matrices[i], 2)[1]
        z_stab = np.split(binary_matrices[i], 2)[0]
        x_stab, z_stab = z_stab, x_stab
        stabilizer_matrix = np.concatenate((z_stab, x_stab))

        # optimizing CNOT segments
        #optimizing first CNOT round
        if not len(D + G) == 0:
            CNOT_segment_1 = CNOT_matrix(D + G, num_qubits)
            CNOT_segment_1 = optimize_circuit(CNOT_segment_1)
        else:
            CNOT_segment_1 = []

        ##optimizing the second CNOTround
        if not len(K) == 0:
            CNOT_segment_2 = CNOT_matrix(K, num_qubits)
            CNOT_segment_2 = optimize_circuit(CNOT_segment_2)
        else:
            CNOT_segment_2 = []

        ##optimizing the third CNOT round
        if not len(O) == 0:
            CNOT_segment_3 = CNOT_matrix(O, num_qubits)
            CNOT_segment_3 = optimize_circuit(CNOT_segment_3)
        else:
            CNOT_segment_3 = []

        # putting together final circuit
        final_circuit = tq.QCircuit()
        for j in B:
            for k in j.gates:
                final_circuit += k        # H

        final_circuit += CNOT_segment_1   # C

        for j in I:
            for k in j.gates:
                final_circuit += k       # P

        final_circuit += CNOT_segment_2  # C

        for j in M:
            for k in j.gates:
                final_circuit += k      # P

        final_circuit += CNOT_segment_3 # C

        for j in Q:
            for k in j.gates:
                final_circuit += k      # H

        ##converting the binary representation back to qubit form
        paulis=[]
        for i in range(len(stabilizer_matrix[0,:])):
            a=BinaryPauliString(stabilizer_matrix[:,i])
            paulis.append(a)
        H=BinaryHamiltonian(paulis)
        H=H.to_qubit_hamiltonian()

        Result.append(tuple((H, final_circuit)))
    
    return (Result)

def CNOT_matrix(list_of_gates,num_qubits):
    """This function represents the CNOT segments as a single unitary matrix, and this matrix will be used for optimizing
    CNOT gate depth as put forth by igor patel and hayes - Efficient Synthesis of Linear Reversible Circuits"""
    matrices=[]
    ##the CNOT circuit soptimization does not work with odd number of qubits, so making the CNOT matrix       have an even dimension seems to be a workaround (I dont think I have done anything illegal)
    if not (num_qubits % 2 == 0):
        num_qubits += 1
    for U in reversed(list_of_gates):
        gate = np.identity(num_qubits)
        for i in U.gates:
            target_qubit = i.target[0]
            control_qubit = i.control[0]
        gate[target_qubit, control_qubit] = 1
        matrices.append(gate)
    C_NOT_matrix=matrices[0]
    for i,j in enumerate(matrices):
        if i == 0:
            continue
        else:
            C_NOT_matrix=np.matmul(C_NOT_matrix,j) % 2
            
    return(C_NOT_matrix)

def optimize_circuit(CNOT_matrix):
    """CNOT circuit optimization as put forth in paper by igor, patel and hayes -
    Efficient Synthesis of Linear Reversible Circuits"""
    CNOT_matrix , circuit1 = Lwr_CNOT_Synth(CNOT_matrix)
    CNOT_matrix = np.transpose(CNOT_matrix)
    CNOT_matrix , circuit2 = Lwr_CNOT_Synth(CNOT_matrix)
    optimized_circuit = tq.QCircuit()
    ## putting together the two circuits

    #the CNOT gates in circuit two have the control and target qubits swapped
    for U in (circuit2):
        control_qubit = U[1]
        target_qubit = U[0]
        optimized_circuit += tq.gates.CNOT(control_qubit, target_qubit)

    ## the order of CNOT gates in circuit one are reversed
    for U in reversed(circuit1):
        control_qubit = U[0]
        target_qubit = U[1]
        optimized_circuit += tq.gates.CNOT(control_qubit, target_qubit)

    return(optimized_circuit)

def Lwr_CNOT_Synth(C_NOT_matrix):
    """CNOT circuit optimization as put forth in paper by igor, patel and hayes"""
    num_qubits = len(C_NOT_matrix[0,:])
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
    return(C_NOT_matrix, circ)

def binary_groups(commuting_groups):
    """goes through all the pauli words in the qubit wise commuting groups and converts them into their binary vector
    form"""
    binary_group = []
    for i in range(len(commuting_groups)):
        binary_group.append(commuting_groups[i].get_binary())
    return binary_group

def initial_binary_matrices(list_of_paulis):
    """generates the intial binary matrix form of each qubitwise commuting group.
    with these matrices we can do a series of transformations to bring them to the z- canoncial form.
    Must be passed through to function as a list to work"""
    list_of_binary_matrices = []
    for i in range(len(list_of_paulis)):
        for j in range(len(list_of_paulis[i])):
            bu.flip_first_second_half(list_of_paulis[i][j])
        list_of_binary_matrices.append(np.column_stack(list_of_paulis[i]))
    return(list_of_binary_matrices)

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
    return(A)

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

def independent_paulis(binary_pauli_matrix):
    """function returns the pivot columns and rows of the reduced row echelon form of the binary matrix representation
    of each pauli word. These pivot columns are linearly independent pauli words. These commuting paulis will form our
    stabilizer matrix in the circuit synthesis"""
    lin_indep_paulis=[]
    B = copy.deepcopy(binary_pauli_matrix)
    rref_matrix=RREF_binary(binary_pauli_matrix)
    n_rows, n_cols = np.shape(rref_matrix)
    A=rref_matrix
    b=B
    pivot_columns_original_matrix=[]
    for j in range(n_cols):  # For each column
        current_row = 0

        if current_row >= n_rows:
            break
        pivot_row = current_row

        while pivot_row < n_rows and A[pivot_row, j] == 0:
            pivot_row += 1

        if pivot_row == n_rows:
            continue
        current_row = pivot_row + 1
        a = 1 + pivot_row
        for k in range(current_row, n_rows):
            if A[k, j] == 1:
                break
            a += 1
        if a == n_rows:
            pivot_columns_original_matrix.append(b[:,j])
    pivot_rows = []
    for i in range(len(A[:,0])):
        for j in range(len(A[0,:])):
            if A[i][j] == 1:
                pivot_rows.append(A[i,:])
                break
    pivot_matrix = np.row_stack(pivot_rows)
    lin_indep_paulis = np.column_stack(pivot_columns_original_matrix)
    return lin_indep_paulis

def Hadamard_phase(matrix):
    """applies the hadamard phase in the circuit synthesis on the binary matrix representations of our paulis:
    makes the X stabilizer matrix full rank"""
    y=copy.deepcopy(matrix)
    x_matrix=np.split(y,2)[1]
    n_qubits=len(x_matrix[:,0])
    circ = []
    column_echelon_x=REF_binary(np.transpose(x_matrix))
    column_echelon_x=np.transpose(column_echelon_x)
    # finding dependent rows, these are the bits i flip with the hadamard gates
    for k in range(len(column_echelon_x[:,0])):
        for column, l in enumerate(column_echelon_x[k,:]):
            if l == 1:
                for m in range(k+1,len(column_echelon_x[:,0])):
                    if column_echelon_x[m,column] == 1:
                        column_echelon_x[m,:]=(column_echelon_x[k,:]+ column_echelon_x[m,:]) % 2
    rows = np.all((column_echelon_x == 0), axis=1)
    bits_to_flip=[]
    for k in range(len(rows)):
        if rows[k]:
            bits_to_flip.append(k)
    rank=npl.matrix_rank(column_echelon_x)
    #flipping bits
    if rank != len(x_matrix[0,:]):
        for k in bits_to_flip:
            circ.append(tq.gates.H(target=k))
            matrix[[k,k+n_qubits]]=matrix[[k+n_qubits,k]]
            
    return(matrix,circ)

def C_NOT(matrix):
    """uses CNOT gates to reduce the x_matrix to Identity matrix"""
    circuit=[]
    x_stab = np.split(matrix, 2)[1]
    rank = npl.matrix_rank(x_stab)
    num_qubits = len(x_stab[:, 0])
    num_paulis = int(len(x_stab[0, :]))
    z_stab = np.split(matrix, 2)[0]
    x_stab = np.transpose(x_stab)
    z_stab = np.transpose(z_stab)
    if rank != num_qubits:
        for i in (range(num_paulis)):
            ##adding ones on diagonal
            if x_stab[i, i] == 0:
                ones = np.where(x_stab[i, :] == 1)[0]
                target_qubit = i
                for control_qubit in ones:
                    if control_qubit < target_qubit:
                        continue
                    else:
                        circuit.append(tq.gates.CNOT(control_qubit, target_qubit))
                        for k in range(0, num_paulis):
                            x_stab[k, target_qubit] = int(x_stab[k, target_qubit]) ^ int(x_stab[k, control_qubit])
                            z_stab[k, control_qubit] = int(z_stab[k, control_qubit]) ^ int(z_stab[k, target_qubit])
                    break
                #gaussian elimation on lower rows
                control_qubit=i
                ones = np.where(x_stab[i, :] == 1)[0]
                for target_qubit in ones:
                    if target_qubit == control_qubit:
                        continue
                    circuit.append(tq.gates.CNOT(control_qubit, target_qubit))
                    for k in range(0, num_paulis):
                        x_stab[k, target_qubit] = int(x_stab[k, target_qubit]) ^ int(x_stab[k, control_qubit])
                        z_stab[k, control_qubit] = int(z_stab[k, control_qubit]) ^ int(z_stab[k, target_qubit])
            # if one one already on diagonal, performing guassian elimination on rest of rows
            if x_stab[i, i] == 1:
                ones = np.where(x_stab[i, 0:rank] == 1)[0]
                control_qubit = i
                for target_qubit in ones:
                    if i == target_qubit:
                        continue
                    circuit.append(tq.gates.CNOT(control_qubit, target_qubit))
                    for k in range(0, num_paulis):
                        x_stab[k, target_qubit] = int(x_stab[k, target_qubit]) ^ int(x_stab[k, control_qubit])
                        z_stab[k, control_qubit] = int(z_stab[k, control_qubit]) ^ int(z_stab[k, target_qubit])
                        
    elif rank == num_qubits:
        for i in range(len(x_stab[:, 0])):
            ones = np.where(x_stab[i, :] == 1)[0]
            # puttng ones on diagonal
            if x_stab[i, i] == 0:
                control_qubit = ones[-1]
                target_qubit = i
                circuit.append(tq.gates.CNOT(control_qubit, target_qubit))
                for k in range(0, num_paulis):
                    x_stab[k, target_qubit] = int(x_stab[k, target_qubit]) ^ int(
                        x_stab[k, control_qubit])
                    z_stab[k, control_qubit] = int(z_stab[k, control_qubit]) ^ int(
                        z_stab[k, target_qubit])
                control_qubit = i
                # gaussian elimination on rest of rows
                for target_qubit in ones:
                    if target_qubit == i:
                        continue
                    circuit.append(tq.gates.CNOT(control_qubit, target_qubit))
                    for k in range(0, num_paulis):
                        x_stab[k, target_qubit] = int(x_stab[k, target_qubit]) ^ int(
                            x_stab[k, control_qubit])
                        z_stab[k, control_qubit] = int(z_stab[k, control_qubit]) ^ int(
                            z_stab[k, target_qubit])
            # if one already on diagonal -> gausssian elimination
            if x_stab[i, i] == 1:
                ones = np.where(x_stab[i, :] == 1)[0]
                control_qubit = i
                for target_qubit in ones:
                    if i == target_qubit:
                        continue
                    circuit.append(tq.gates.CNOT(control_qubit, target_qubit))
                    for k in range(0, num_paulis):
                        x_stab[k, target_qubit] = int(x_stab[k, target_qubit]) ^ int(
                            x_stab[k, control_qubit])
                        z_stab[k, control_qubit] = int(z_stab[k, control_qubit]) ^ int(
                            z_stab[k, target_qubit])
                        
    x_stab = np.transpose(x_stab)
    z_stab = np.transpose(z_stab)
    stabilizer_matrix = np.concatenate((z_stab, x_stab))
    
    return (stabilizer_matrix, circuit)

def basis_extension(matrix):
    """Performs a basis extension as done in the Crawford et. al paper to make my binary matrices have the same number
    of columns as there are qubits, this step allows for the binary cholesky decomposition in next step"""
    
    #please dont ask me how this works because I wrote this and forgot to comment it 
    num_qbits=int(len(matrix[:,0])/2)
    matrix_c=copy.deepcopy(matrix)
    x_matrix=np.split(matrix_c,2)[1]
    z_matrix=np.split(matrix_c,2)[0]
    num_independents=int(len(z_matrix[0,:]))
    rank=npl.matrix_rank(x_matrix)
    if num_independents == num_qbits:
        extended_basis_matrix=np.concatenate((z_matrix,x_matrix))
    elif num_independents < num_qbits:
        rows=[]
        for l in range(num_independents,num_qbits):
            rows.append(l)
        if int(len(rows)) == int(num_qbits-num_independents):
            num_zeros=np.zeros((int(len(rows)),int(len(rows))) , dtype=np.int64)
        else:
            num_zeros=np.zeros([int(len(rows)),int(num_qbits-num_independents)] , dtype=np.int64)
        if num_independents == rows[-1]:
            D = (z_matrix[num_independents,:])
        elif rows[-1] > num_independents:
            D = (z_matrix[rows[0]:,:])
        if D.ndim == 1:
            D = np.concatenate((D,num_zeros[0]))
        else:
            D = np.concatenate((D,num_zeros),axis=1)
        z_matrix = np.column_stack((z_matrix,np.transpose(D)))
        eye=np.identity(num_qbits-num_independents)
        zeros=np.zeros((rank,len(z_matrix[0,:])-len(x_matrix[0,:])),dtype='int64')
        column_to_append=np.concatenate((zeros,eye))
        x_matrix=np.column_stack((x_matrix,column_to_append))
        extended_basis_matrix=np.concatenate((z_matrix,x_matrix))
        
    return (extended_basis_matrix)

def phase_round(stabilizer_matrix):
    """adds a diagonal matrix, D, to the Z stabilizer matrix, such that Z + D = M*M' for some invertible M"""
    x_stab=np.split(stabilizer_matrix,2)[1]
    z_stab=np.split(stabilizer_matrix,2)[0]
    num_qubits=int(len(stabilizer_matrix[:,0])/2)
    matrix = copy.deepcopy(z_stab)
    M=np.identity(num_qubits)
    circuit=[]

    # method for finding the invertible M discussed in the Gottesman paper - this loop implements the method
    for j in range(0,num_qubits):
        if j == 0:
            for i in range(1,num_qubits):
                M[i,j]=matrix[i,j]
            continue
        for i in range(j+1,num_qubits):
            result=[]
            for k in range(0,j):
                    Sum = (M[i,k]*M[j,k]) % 2
                    result.append(Sum)
            final_sum=sum(result) % 2
            M[i,j] = (matrix[i,j] + final_sum) % 2
    matrix=np.matmul(M,np.transpose(M)) % 2
    bits_to_flip=[]
    for i in range(int(len(matrix[0,:]))):
        if matrix[i,i] != z_stab[i,i]:
            bits_to_flip.append(i)
        elif matrix[i,i] == z_stab[i,i]:
            continue
    for target_qubit in bits_to_flip:
        circuit.append(tq.gates.S(target=target_qubit))
        for i in range(0,num_qubits):
            z_stab[target_qubit,i] = int(z_stab[target_qubit,i])^int(x_stab[target_qubit,i])
    stabilizer=np.concatenate((z_stab,x_stab))
    
    return(stabilizer,circuit)

def second_round_cnot(stabilizer_matrix):
    """performs a cholesky decompostion of the symmetric Z = D + M*M' stabilizer matrix """
    circuit=[]
    x_stab=np.split(stabilizer_matrix,2)[1]
    num_qubits=len(x_stab[:,0])
    num_paulis=int(len(x_stab[0,:]))
    z_stab=np.split(stabilizer_matrix,2)[0]
    x_stab=np.transpose(x_stab)
    z_stab=np.transpose(z_stab)
    matrix = copy.deepcopy(z_stab)
    M=np.identity(num_qubits)
    
    # method for finding the invertible M discussed in the Gottesman paper
    for j in range(0,num_qubits):
        if j == 0:
            for i in range(1,num_qubits):
                M[i,j]=matrix[i,j]
            continue
        for i in range(j+1,num_qubits):
            result=[]
            for k in range(0,j):
                    Sum = (M[i,k]*M[j,k]) % 2
                    result.append(Sum)
            final_sum=sum(result) % 2
            M[i,j] = (matrix[i,j] + final_sum) % 2
            
    for i in range(1,num_qubits):
        ones = np.where(M[i,:] == 1)[0]
        for j,k in enumerate(ones):
            control_qubit = ones[-1]
            target_qubit = k
            if control_qubit != target_qubit:
                for l in range(0,num_qubits):
                    x_stab[l,target_qubit] = int(x_stab[l,target_qubit]) ^ int(x_stab[l,control_qubit])
                    z_stab[l,control_qubit] = int(z_stab[l,control_qubit]) ^ int(z_stab[l,target_qubit])
                circuit.append(tq.gates.CNOT(control_qubit,target_qubit))
                continue
            else:
                continue
                
    x_stab=np.transpose(x_stab)
    z_stab=np.transpose(z_stab)
    stabilizer = np.concatenate((z_stab, x_stab))
    
    return(stabilizer,circuit)

def second_round_phase(stabilizer_matrix):
    """phase on all the qubits eliminates the Z_stabilizer to the Zero Matrix"""
    circuit=[]
    x_stab=np.split(stabilizer_matrix,2)[1]
    z_stab=np.split(stabilizer_matrix,2)[0]
    num_qubits=int(len(stabilizer_matrix[:,0])/2)
    bits_to_flip=[]
    
    ## add all qubits to bits to be phased
    for i in range(num_qubits):
        bits_to_flip.append(i)
    
    #phase the qubits
    for target_qubit in bits_to_flip:
        circuit.append(tq.gates.S(target_qubit))
        for i in range(num_qubits):
            z_stab[i,target_qubit] = int(z_stab[i,target_qubit])^int(x_stab[i,target_qubit])
            
    stabilizer=np.concatenate((z_stab,x_stab))
    
    return(stabilizer,circuit)

def final_hadamard(stabilizer_matrix):
    """the final round of gates in the synthesis. Hadamard gates applied to all qubits 
    makes the Z_stabilizer matrix the identity matrix and puts the
     stabilizer matrix into canonical form"""
    circuit = []
    num_qubits=int(len(stabilizer_matrix[:,0])/2)
    bits_to_flip=[]
    
    for i in range(num_qubits):
        bits_to_flip.append(i)
        
    for target_qubit in bits_to_flip:
        circuit.append(tq.gates.H(target_qubit))
        stabilizer_matrix[[target_qubit,target_qubit+num_qubits]]=stabilizer_matrix[[target_qubit+num_qubits,target_qubit]]
        
    return(stabilizer_matrix,circuit)


    
    
    raise NotImplementedError
