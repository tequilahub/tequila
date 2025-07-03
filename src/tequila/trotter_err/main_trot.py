import numpy as np
import tequila as tq
import openfermion
import multiprocessing as mp
from scipy import sparse


def SpecNormComm(Op1, Op2, nqubs, Projector=None):
    """
    Returns: the spectral norm of the (hermitian) operator im*[Op1,Op2]=im*(Op1-Op2)
    Input:
    Op1,Op2: are either qubit or fermionic openfermion objects
    nqubs: number of qubits that define the Hamiltonian
    Projector: projector operator in sparse matrix form, useful when considering symmetries

    """

    SpOp1 = openfermion.get_sparse_operator(Op1, n_qubits=nqubs)
    SpOp2 = openfermion.get_sparse_operator(Op2, n_qubits=nqubs)

    Comm = 1j * (SpOp1 * SpOp2 - SpOp2 * SpOp1)

    if Projector != None:
        Comm = Projector * Comm

    spNorm = sparse.linalg.eigs(Comm, k=1, which="LM", return_eigenvectors=False)

    return np.abs(spNorm[0])


def compute_alpha_2(pair, list_frags, nqubs, projector):
    frag1 = list_frags[pair[0]]
    frag2 = list_frags[pair[1]]
    return SpecNormComm(frag1, frag2, nqubs, Projector=projector)


def EstTrotErr(ListFrags, nqubs, SymDict=None):
    """
    Returns: the sum of the spectral norm of (unique) commutators between Hamiltonian fragments listed in ListFrags. If SymDict is different from None
    we use the parameters contained in it (see below) to project the commutators to a symmetric manifold.
    Input:
    ListFrags: List of Hamiltonian fragments either in fermionic or qubit form.
    nqubs: number of qubits that are used to build the total Hamiltonian
    *SymDict entries****
    SymOps: list of operators that commute with all Hamiltonian fragments listed in ListFrags, either in Fermionic or Qubit form
    QNumbs: array whose ith entry contains the eigenvalue associated to the ith SymOps element of the symmetric manifold where
    the commutators are projected.
    """
    GaussProj = None
    if SymDict:
        # Build an (approximate) projector in sparse matrix form...
        ListSymOps = SymDict["SymOps"]
        ListQNums = SymDict["QNumbs"]
        NOps = len(ListQNums)

        SpSymOp = openfermion.linalg.get_sparse_operator(ListSymOps[0], n_qubits=nqubs)
        QNum = ListQNums[0]

        GaussProj = sparse.linalg.expm(-10 * (SpSymOp - QNum * sparse.csc_matrix(np.eye(2**nqubs))) ** 2)

        for i in range(1, NOps):
            SpSymOp = openfermion.linalg.get_sparse_operator(ListSymOps[i], n_qubits=nqubs)
            QNum = ListQNums[i]
            GaussProj = GaussProj * sparse.linalg.expm(
                -10 * (SpSymOp - QNum * sparse.csc_matrix(np.eye(2**nqubs))) ** 2
            )

    NFrags = len(ListFrags)
    idxs = []

    for i in range(NFrags):
        for j in range(i + 1, NFrags):
            idxs.append([i, j])

    Npairs = int(NFrags * (NFrags - 1) / 2)

    alpha_2 = 0.0

    for k in range(Npairs):
        Pair = idxs[k]
        alpha_2 += SpecNormComm(ListFrags[Pair[0]], ListFrags[Pair[1]], nqubs, Projector=GaussProj)

    return 2 * alpha_2


def compute_alpha_2(pair, list_frags, nqubs, gauss_proj):
    frag1 = list_frags[pair[0]]
    frag2 = list_frags[pair[1]]
    return SpecNormComm(frag1, frag2, nqubs, Projector=gauss_proj)


# Experimental parallel version of Trotter error calculation....
def EstTrotErrParal(ListFrags, nqubs, SymDict=None, pool=None):
    """
    Parallel version of EstTrotErr function
    Returns: the sum of the spectral norm of (unique) commutators between Hamiltonian fragments listed in ListFrags. If SymDict is different from None
    we use the parameters contained in it (see below) to project the commutators to a symmetric manifold.
    Input:
    ListFrags: List of Hamiltonian fragments either in fermionic or qubit form.
    nqubs: number of qubits that are used to build the total Hamiltonian
    *SymDict entries****
    SymOps: list of operators that commute with all Hamiltonian fragments listed in ListFrags, either in Fermionic or Qubit form
    QNumbs: array whose ith entry contains the eigenvalue associated to the ith SymOps element of the symmetric manifold where
    the commutators are projected.
    """
    GaussProj = None
    if SymDict:
        # Build an (approximate) projector in sparse matrix form...
        ListSymOps = SymDict["SymOps"]
        ListQNums = SymDict["QNumbs"]
        NOps = len(ListQNums)

        SpSymOp = openfermion.linalg.get_sparse_operator(ListSymOps[0], n_qubits=nqubs)
        QNum = ListQNums[0]

        GaussProj = sparse.linalg.expm(-10 * (SpSymOp - QNum * sparse.csc_matrix(np.eye(2**nqubs))) ** 2)

        for i in range(1, NOps):
            SpSymOp = openfermion.linalg.get_sparse_operator(ListSymOps[i], n_qubits=nqubs)
            QNum = ListQNums[i]
            GaussProj = GaussProj * sparse.linalg.expm(
                -10 * (SpSymOp - QNum * sparse.csc_matrix(np.eye(2**nqubs))) ** 2
            )

    NFrags = len(ListFrags)
    idxs = []

    for i in range(NFrags):
        for j in range(i + 1, NFrags):
            idxs.append([i, j])

    Npairs = int(NFrags * (NFrags - 1) / 2)

    # mp.set_start_method('spawn')
    # Define the function that computes alpha_2 for a pair of fragments
    if pool is None:
        pool = mp.Pool(mp.cpu_count())
        # with mp.Pool(mp.cpu_count()) as pool:
    args_list = [(pair, ListFrags, nqubs, GaussProj) for pair in idxs]

    # Use the multiprocessing pool to compute alpha_2 for all pairs of fragments
    results = pool.starmap(compute_alpha_2, args_list)

    # Compute the sum of alpha_2 over all pairs of fragments
    alpha_2 = sum(results)

    # Close the multiprocessing pool
    pool.close()
    pool.join()

    return 2 * alpha_2
