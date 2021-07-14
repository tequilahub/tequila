from tequila.grouping.binary_rep import BinaryHamiltonian
import tequila as tq

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
    raise NotImplementedError
