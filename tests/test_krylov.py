import scipy
import tequila as tq
from tequila import braket
from tequila.hamiltonian.qubit_hamiltonian import QubitHamiltonian
from tequila.tools.random_generators import make_random_circuit
import itertools as it
import numpy as np

def simple_krylov(n_krylov_states: int=2):
    """Function that applies the Krylov method to an Hamiltonian 
       defined from the Krylov states.

    Args:
        n_krylov_states (int, optional): _description_. Defaults to 2.
    """

    np.random.seed(111)
    #we create states randomly, in this way it is very unlikely they will be orthogonal
    krylov_circs = [make_random_circuit(2, enable_controls=True) for i in range(2)] 

    # creating the wavefunctions from the circuits
    krylov_states = [tq.simulate(circ) for circ in krylov_circs]
    
    krylov_states_couples = list(it.product(krylov_states, repeat=2)) # list of all possible couples of Krylov states

    # creating an hamiltonian from the obtained wavefunctions
    # in this way the ground state will be known
    H = QubitHamiltonian()
    for i, j in krylov_states_couples:
        H -= tq.paulis.KetBra(ket = i, bra = j)

    #print(H)

    #print(H.is_hermitian())

    ham_eval_matrix = make_H_eval_matrix(krylov_circs, H)
    #print(ham_eval_matrix)
    overlap_matrix = make_overlap_matrix(krylov_circs)

    kry_eigenvalues, kry_eigenvectors = scipy.linalg.eigh(ham_eval_matrix, overlap_matrix)
    #print('Overlap matrix:\n',overlap_matrix,'\n')
    
    eigenvalues, eigenvectors = np.linalg.eig(H.to_matrix())

    #print('Ground State Krylov',kry_eigenvalues[0])
    #print('Ground State:', eigenvalues[0])
    assert np.isclose(kry_eigenvalues[0],eigenvalues[0], atol=1e-4)

    return

def make_overlap_matrix(circs:list)->np.array:
    """Function that builds a matrix of the overlap between 
       the given states.

    Args:
        
        circs (list): List of QCircuit()

    Returns:
        np.array: _description_
    """
    n_states = len(circs)
    circs_couples = list(it.product(circs, repeat=2)) # list of all possible couples of circuits

    overlap_matrix = []
    for i,j in circs_couples:
        overlap_real, overlap_im = braket(ket=i, bra=j)
        overlap = tq.simulate(overlap_real)+ 1j*tq.simulate(overlap_im)
        overlap_matrix.append(overlap )
        
    overlap_matrix = np.reshape(np.array(overlap_matrix), (n_states,n_states))
    return overlap_matrix

def make_H_eval_matrix(circs: list, H: QubitHamiltonian)->np.array:
    """Function that builds a matrix of the transition elements between 
    the given states and the Hamiltonian operator. 


    Args:
        
        circs (list): List of QCircuit()
        H (QubitHamiltonian): Hamiltonian operator

    Returns:
        np.array: _description_
    """

    n_states = len(circs)
    circs_couples = list(it.product(circs, repeat=2)) # list of all possible couples of circuits

    ham_expval_matrix = []
    for i,j in circs_couples:
        # if id(i) == id(j):
        #     ham_expval_matrix.append(tq.simulate(braket(ket=i, bra=j, operator=H)))
        # else:
        tmp_real, tmp_im = braket(ket=i, bra=j, operator=H)
        ham_expval_matrix.append(tq.simulate(tmp_real)+ 1j*tq.simulate(tmp_im) )
    ham_expval_matrix = np.reshape(np.array(ham_expval_matrix), (n_states,n_states))
    return ham_expval_matrix
