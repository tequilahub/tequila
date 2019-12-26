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
        x_set = coisotropic_space.deepcopy()
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
