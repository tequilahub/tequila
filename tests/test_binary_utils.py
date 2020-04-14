from tequila.grouping.binary_utils import get_lagrangian_subspace, binary_null_space, binary_symplectic_gram_schmidt, binary_symplectic_inner_product, binary_symplectic_orthogonalization, pick_symplectic_pair, largest_first, recursive_largest_first
import numpy as np
import itertools


def prepare_binary_matrix():
    matrix = []
    matrix.append(np.array([1, 1, 1, 1, 1, 1]))
    matrix.append(np.array([0, 0, 1, 0, 0, 0]))
    matrix.append(np.array([1, 1, 1, 1, 1, 0]))
    matrix.append(np.array([1, 0, 0, 0, 0, 0]))
    matrix.append(np.array([1, 1, 0, 0, 0, 0]))
    return matrix


def test_binary_symplectic_inner_product():
    a = np.array([1, 1, 0, 0])
    b = np.array([1, 1, 1, 1])
    c = np.array([1, 1, 1, 0])

    assert (binary_symplectic_inner_product(a, b) == 0)
    assert (binary_symplectic_inner_product(a, c) == 1)
    assert (binary_symplectic_inner_product(b, c) == 1)


def test_binary_symplectic_orthogonalization():
    matrix = prepare_binary_matrix()

    x = np.array([1, 1, 0, 0, 0, 0])
    y = np.array([0, 0, 0, 1, 0, 0])

    binary_symplectic_orthogonalization(matrix, x, y)

    # if all elements are orthogonalized.
    for vec in matrix:
        assert (binary_symplectic_inner_product(vec, x) == 0)
        assert (binary_symplectic_inner_product(vec, y) == 0)


def test_pick_symplectic_pair():
    matrix = prepare_binary_matrix()
    original_length = len(matrix)

    x, y = pick_symplectic_pair(matrix)

    # if is symplectic pair
    assert (binary_symplectic_inner_product(x, y) == 1)
    # if the symplectic pairs are picked out from the matrix
    assert (len(matrix) == original_length - 2)
    for vec in matrix:
        assert (not all(vec == x))
        assert (not all(vec == y))


def test_symplectic_gram_schmidt():
    matrix = prepare_binary_matrix()
    lagrangian_subspace = binary_symplectic_gram_schmidt(matrix)

    # if subspace is lagrangian
    assert (len(lagrangian_subspace) == len(matrix[0]) // 2)

    for ivec in lagrangian_subspace:
        for jvec in lagrangian_subspace:
            assert (binary_symplectic_inner_product(ivec, jvec) == 0)


def test_binary_null_space():
    matrix = np.array(prepare_binary_matrix())
    null_basis = binary_null_space(matrix.copy())

    # if gets null basis
    for vec in null_basis:
        assert (all((matrix @ vec.T) % 2 == 0))


def test_get_lagrangian_subspace():
    a = np.array([1, 1, 0, 0, 0, 0])
    b = np.array([1, 0, 0, 0, 0, 0])
    matrix = []
    matrix.append(a)
    matrix.append(b)
    lagrangian_subspace = get_lagrangian_subspace(matrix)

    # if lagrangian subspace spans given vectors
    assert (is_binary_basis(lagrangian_subspace, a))
    assert (is_binary_basis(lagrangian_subspace, b))

    # if subspace is lagrangian
    for ivec in matrix:
        for jvec in matrix:
            assert (binary_symplectic_inner_product(ivec, jvec) == 0)


def is_binary_basis(basis, vec):
    dim = len(basis)
    basis = np.array(basis)

    # Generate all combination
    lst = itertools.product([0, 1], repeat=dim)
    equal_lst = []
    for elem in lst:
        cur_comb = (elem @ basis) % 2
        equal_lst.append(all(cur_comb == vec))

    return any(equal_lst)


def test_graph_coloring():
    # anti-commutation graph has either 1, or 0. 0 on the diagonal
    num_elem = 5
    cg = np.random.randint(2, size=(num_elem, num_elem))
    np.fill_diagonal(cg, 1)
    for i in range(num_elem):
        for j in range(i + 1, num_elem):
            cg[j, i] = cg[i, j]

    lf_grouping = largest_first(np.arange(num_elem), num_elem, cg)
    rlf_grouping = recursive_largest_first(np.arange(num_elem), num_elem, cg)

    assert verify_graph_coloring(cg, lf_grouping)
    assert verify_graph_coloring(cg, rlf_grouping)


def verify_graph_coloring(cg, grouping):
    # Check if all terms within each group are disconnected
    for key in grouping.keys():
        cur_group = grouping[key]
        num_group = len(cur_group)
        for i in range(num_group):
            for j in range(i+1, num_group):
                idx1, idx2 = cur_group[i], cur_group[j]
                if cg[idx1][idx2] == 1:
                    return False
    return True
