import tequila as tq
from tequila.apps.krylov.krylov import krylov_method
from tequila.hamiltonian.qubit_hamiltonian import QubitHamiltonian
from tequila.tools.random_generators import make_random_circuit
import itertools as it
import numpy as np

def test_simple_krylov(n_krylov_states: int=2):
    """Function that applies the Krylov method to an Hamiltonian 
       defined from the Krylov states.

    Args:
        n_krylov_states (int, optional): _description_. Defaults to 2.
    """

    np.random.seed(111)
    #we create states randomly, in this way it is very unlikely they will be orthogonal
    krylov_circs = [make_random_circuit(2, enable_controls=True) for i in range(n_krylov_states)] 

    # creating the wavefunctions from the circuits
    krylov_states = [tq.simulate(circ) for circ in krylov_circs]
    krylov_states_couples = list(it.product(krylov_states, repeat=2)) # list of all possible couples of Krylov states

    # creating an hamiltonian from the obtained wavefunctions
    # in this way the ground state will be known
    H = QubitHamiltonian()
    for i, j in krylov_states_couples:
        H -= tq.paulis.KetBra(ket = i, bra = j)
    
    # applying Krylov method
    kry_energies, kry_coefficients_matrix = krylov_method(krylov_circs, H)

    kry_ground_energy = kry_energies[0]
    kry_ground_coefficients = kry_coefficients_matrix[0]
    
    #exact diagonalization
    eigenvalues, eigenvectors = np.linalg.eig(H.to_matrix())

    #print('Ground State Energy Krylov',kry_eigenvalues[0])
    #print('Ground State Energy:', eigenvalues[0])
    assert np.isclose(kry_ground_energy, eigenvalues[0], atol=1e-4)

    return
