from tequila import TequilaException
import numpy as np


def get_lagrangian_subspace(binary_matrix):
    '''
    Given a list of vectors. 
    Find the lagrangian subspace that spans the vectors. 

    Return: A list of vectors that forms the lagrangian subspace and
    spans the space of the given matrix. 
    '''
    null_basis = binary_null_space(np.array(binary_matrix))
    lagrangian_basis = binary_symplectic_gram_schmidt(null_basis)
    for vec in lagrangian_basis:
        flip_first_second_half(vec)

    return lagrangian_basis


def flip_first_second_half(vector):
    '''
    Modify: Flip the first half the vector to second half, and vice versa. 
    '''
    dim = len(vector) // 2
    tmp = vector[:dim].copy()
    vector[:dim] = vector[dim:]
    vector[dim:] = tmp


def binary_null_space(binary_matrix):
    '''
    Return the basis of binary null space of the given binary matrix. 

    Return: a list of binary vectors that forms the null space of the given binary matrix.
    Modify: gauss_eliminated binary_matrix. 
    '''
    dim = len(binary_matrix[0, :])
    I = np.identity(dim)

    # Apply Gauss Elimination on [A; I]
    for i in range(dim):
        non_zero_row = np.where(binary_matrix[:, i] == 1)

        # if exists a non zero index
        if len(non_zero_row[0]) != 0:
            non_zero_row = non_zero_row[0][0]
            for j in range(i + 1, dim):
                if binary_matrix[non_zero_row, j] != 0:
                    binary_matrix[:, j] = (binary_matrix[:, i] +
                                           binary_matrix[:, j]) % 2
                    I[:, j] = (I[:, i] + I[:, j]) % 2

    null_basis = []
    # Find zero column index
    for i in range(dim):
        if all(binary_matrix[:, i] == 0):
            null_basis.append(I[:, i])
    return null_basis


def binary_symplectic_gram_schmidt(coisotropic_space):
    '''
    Accepts a list of binary vectors, basis which
    forms coisotropic space in 2*dim binary symplectic space (so len of list > dim)
    Use Symplectic Gram-Schmidt to find the lagrangian subspace within the coisotropic space
    that seperates the binary space into X, Y subspaces such that <x_i, y_i> = 1. 

    Return: A list of the basis vector of the lagrangian subspace X. 
    Modify: coisotropic space to isotropic space. 
    '''
    dim = len(coisotropic_space[0]) // 2
    symplectic_dim = len(coisotropic_space) - dim

    x_set = []
    # If is lagrangian already
    if symplectic_dim == 0:
        x_set = coisotropic_space
    else:
        for i in range(symplectic_dim):
            x_cur, y_cur = pick_symplectic_pair(coisotropic_space)
            binary_symplectic_orthogonalization(coisotropic_space, x_cur,
                                                y_cur)
            x_set.append(x_cur)
        x_set.extend(coisotropic_space)
    return x_set


def pick_symplectic_pair(coisotropic_space):
    '''
    Pick out one pair in the symplectic space such that 
    <sympX, sympY> = 1

    Return: A symplectic pair. 
    Modify: Pops the chosen pair from given list of vectors. 
    '''
    for i, ivec in enumerate(coisotropic_space):
        for j, jvec in enumerate(coisotropic_space):
            if i < j:
                if binary_symplectic_inner_product(ivec, jvec) == 1:
                    y = coisotropic_space.pop(j)
                    x = coisotropic_space.pop(i)
                    return x, y


def binary_symplectic_orthogonalization(space, x, y):
    '''
    Symplectically orthogonalize all vectors in space
    to the symplectic pair x and y

    Modify: The linear space of space is modified such that each basis vector 
    is symplectically orthogonal to x and y. 
    '''
    for i in range(len(space)):
        vec = space[i]
        x_overlap = binary_symplectic_inner_product(vec, x)
        y_overlap = binary_symplectic_inner_product(vec, y)
        vec = vec + y_overlap * x + x_overlap * y
        space[i] = vec % 2


def binary_symplectic_inner_product(a, b):
    '''
    Return the binary symplectic inner product between two binary vectors a and b. 

    Return: 0 or 1. 
    '''
    if not len(a) == len(b):
        raise TequilaException(
            'Two binary vectors given do not share same number of qubits. ')
    dim = len(a) // 2
    re = a[:dim] @ b[dim:] + b[:dim] @ a[dim:]

    return re % 2


def binary_solve(basis, target):
    '''
    Get the expansion of the target in the given basis in binary space. 
    '''
    coeff = np.zeros(len(basis))
    tsf_mat, pivot = binary_reduced_row_echelon(basis)
    for i, pivot_idx in enumerate(pivot):
        if target[int(pivot_idx)] == 1:
            coeff = (coeff + tsf_mat[:, i]) % 2
    return coeff


def binary_reduced_row_echelon(basis):
    '''
    Get a list of basis vectors. 
    Perfrom reduced row echelon and return the pivot and the transformation matrix such that
    np.array(basis) @ transformation_matrix = reduced_row_echelon_form
    '''
    num_basis = len(basis)
    dim = len(basis[0])

    # Initiate. No change.
    tsf_mat = np.identity(num_basis)
    reduced_basis = [vec.copy() for vec in basis]
    pivot = np.zeros(num_basis)

    for i, i_col in enumerate(reduced_basis):
        non_zero_row = np.where(i_col == 1)[0][0]
        pivot[i] = non_zero_row
        for j, j_col in enumerate(reduced_basis):
            if (i != j and j_col[non_zero_row] == 1):
                reduced_basis[j] = (j_col + i_col) % 2
                tsf_mat[:, j] = (tsf_mat[:, i] + tsf_mat[:, j]) % 2
    return tsf_mat, pivot


def binary_phase(self_binary, other_binary, n_qubit):
    '''
    Obtain the phase due to binary pauli string self * other. Get 0, 1, 2, 3 for 1, i, -1, -i.
    '''
    def get_phase_helper(this, other):
        '''
        Return the phase incured due to multiplying this * other on a single qubit. 
        '''
        identity = [0, 0]
        x = [1, 0]
        y = [1, 1]
        z = [0, 1]
        if this == identity or other == identity or this == other:
            return 0
        elif this == x:
            if other == y:
                return 1
            else:
                return 3
        elif this == y:
            if other == z:
                return 1
            else:
                return 3
        elif this == z:
            if other == x:
                return 1
            else:
                return 3

    phase = 0
    for i in range(n_qubit):
        self_cur_qub = [self_binary[i], self_binary[i + n_qubit]]
        other_cur_qub = [other_binary[i], other_binary[i + n_qubit]]
        phase += get_phase_helper(self_cur_qub, other_cur_qub)
    phase = phase % 4

    if phase == 0:
        return 1
    elif phase == 1:
        return 1j
    elif phase == 2:
        return -1
    else:
        return -1j


def gen_single_qubit_term(dim, qub, term):
    '''
    Generate single qubit term on the given qubit with given term (0, 1, 2 represents z, x, y) 

    Return: A binary vector representing the single qubit term specified. 
    '''
    word = np.zeros(dim * 2)
    if term == 0:
        word[qub + dim] = 1
    elif term == 1:
        word[qub] = 1
    elif term == 2:
        word[qub] = 1
        word[qub + dim] = 1
    return word

def term_commutes_with_group(term, group, condition):
    '''
    Returns if term commutes with a group of terms.
    '''
    commute = True
    for group_term in group:
        if condition == 'fc':
            commute = term.commute(group_term)
        elif condition == 'qwc':
            commute = term.qubit_wise_commute(group_term)
        else:
            raise TequilaException(f"There is no commutativity condition {condition}")
        if not (commute): break
    return commute

def sorted_insertion_grouping(terms, condition='fc'):
    '''
    Obtain a list of commuting Pauli operator groups using the sorted insertion algorithm.
    '''
    sorted_terms = sorted(terms, key=lambda x: np.abs(x.coeff), reverse=True)
    groups = [] #Initialize groups
    for term in sorted_terms:
        found_group = False
        for idx, group in enumerate(groups):
            commute = term_commutes_with_group(term, group, condition)
            if commute: # Add term if it commutes with the current group.
                groups[idx].append(term)
                found_group = True
                break 
        if not found_group: groups.append([term, ]) #Initiate new group that does not commute with any existing.
    return groups

def largest_first(terms, n, cg):
    """
    Color the graph using "largest first" heuristics with the given adjacency matrix
    Returns a dictionary with keys as colors (just numbers),
    and values as BinaryHamiltonian's
    Faster than RLF but not as good
    """
    rows = cg.sum(axis=0)
    ind = np.argsort(rows)[::-1]
    m = cg[ind,:][:,ind]
    colors = dict()
    c = np.zeros(n, dtype=int)
    k = 0 #color
    for i in range(n):
        neighbors = np.argwhere(m[i,:])
        colors_available = set(np.arange(1, k+1)) - set(c[[x[0] for x in neighbors]])
        term = terms[ind[i]]
        if not colors_available:
            k += 1
            c[i] = k
            colors[c[i]] = [term]
        else:
            c[i] = min(list(colors_available))
            colors[c[i]].append(term)
    return colors

def recursive_largest_first(terms, n, cg):
    """
    Color the graph using "recursive largest first" heuristics with the given adjacency matrix
    Returns a dictionary with keys as colors (just numbers),
    and values as BinaryHamiltonian's
    Produces better results than LF but is slower
    """
    def n_0(m, colored):
        m_colored = m[list(colored)]
        l = m_colored[-1]
        for i in range(len(m_colored)-1):
            l += m_colored[i]
        white_neighbors = np.argwhere(np.logical_not(l))
        return set([x[0] for x in white_neighbors]) - colored

    colors = dict()
    c = np.zeros(n, dtype=int)
    # so, the preliminary work is done
    uncolored = set(np.arange(n))
    colored = set()
    k = 0
    while uncolored:
        decode = np.array(list(uncolored))
        k += 1
        m = cg[:, decode][decode, :]
        v = np.argmax(m.sum(axis=1))
        colored_sub = {v}
        uncolored_sub = set(np.arange(len(decode))) - {v}
        n0 = n_0(m, colored_sub)#vertices that are not adjacent to any colored vertices
        n1 = uncolored_sub - n0
        while n0:
            m_uncolored = m[:,list(n1)][list(n0),:]
            v = list(n0)[np.argmax(m_uncolored.sum(axis=1))]
            colored_sub.add(v) #stable
            uncolored_sub -= {v} #stable
            n0 = n_0(m, colored_sub)
            n1 = uncolored_sub - n0 #stable
        indices = decode[list(colored_sub)]
        c[indices] = k  # stable
        colors[k] = [terms[i] for i in indices] # stable
        colored |= set(indices)
        uncolored = set(np.arange(n)) - colored
    return colors
