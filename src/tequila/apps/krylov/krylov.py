import copy
import scipy
from tequila import braket, QTensor, simulate
from tequila.hamiltonian.qubit_hamiltonian import QubitHamiltonian


def krylov_method(krylov_circs:list, H:QubitHamiltonian, variables:dict=None, assume_real:bool=False, *args, **kwargs)->tuple:
    """Function that applies Krylov method to an Hamiltonian operator,
    given the list of Krylov quantum circuits. If the circuits are parametrized 
    also the variables need to be passed. The method returns the ground state energy 
    and the array of coefficients allowing to obtain an approximation of the ground state.
    Optional function arguments (*args, **kwargs) allows to change simulation options.

    Args:
        krylov_circs (list): List of Krylov circuits.
        H (QubitHamiltonian): Hamiltonian on which we want to apply Krylov method
        variables (dict, optional): Dicitionary containing possible variables to be stored in the Krylov circuits. 
        Defaults to None.
        assume_real (bool): If set to True the function does not compute the imaginary part.
        Default to False.

    Returns:
        tuple(np.ndarray, np.ndarray): array of energies, array of krylov coefficients corresponding to the energies
    """
    
    n_krylov_states = len(krylov_circs)
    HM = QTensor(shape=[n_krylov_states,n_krylov_states])
    SM = QTensor(shape=[n_krylov_states,n_krylov_states])
    
    if variables is not None:
        krylov_circs_x = [U.map_variables(variables) for U in krylov_circs] 
    else:
        krylov_circs_x = copy.deepcopy(krylov_circs)

    for i in range(n_krylov_states):
        for j in range(i,n_krylov_states):
            if assume_real:
                h_real = braket(bra=krylov_circs_x[i], ket=krylov_circs_x[j], operator=H)[0]
                h_im = 0
            else:
                h_real, h_im = braket(bra=krylov_circs_x[i], ket=krylov_circs_x[j], operator=H)
            HM[i,j] = h_real + 1j*h_im
            HM[j,i] = h_real - 1j*h_im
            s_real, s_im = braket(bra=krylov_circs_x[i], ket=krylov_circs_x[j])
            SM[i,j] = s_real + 1j*s_im
            SM[j,i] = s_real - 1j*s_im

    h = simulate(HM, *args, **kwargs) 
    s = simulate(SM, *args, **kwargs)

    v,vv = scipy.linalg.eigh(h,s)

    return v, vv
