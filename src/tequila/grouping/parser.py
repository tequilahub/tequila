from tequila.hamiltonian import QubitHamiltonian

def hamiltonian_to_binary(hamiltonian : QubitHamiltonian):
    '''
    Return a list of binary vector and a list of coefficients
    in the hamiltonian. 
    '''
    return hamiltonian_to_binary_vector(hamiltonian), hamiltonian_to_coeff(hamiltonian)

def hamiltonian_to_binary_vector(hamiltonian : QubitHamiltonian):
    '''
    Return a list of binary vector representing each paulistring
    in the hamiltonian
    '''
    n_qubit = hamiltonian.n_qubits
    return [p.binary(n_qubit).binary for p in hamiltonian.paulistrings]

def hamiltonian_to_coeff(hamiltonian : QubitHamiltonian):
    '''
    Return a list of coefficients in the hamiltonian. 
    '''
    n_qubit = hamiltonian.n_qubits
    return [p.binary(n_qubit).coeff for p in hamiltonian.paulistrings]

