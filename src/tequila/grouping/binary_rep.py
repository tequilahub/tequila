from tequila import TequilaException
from tequila.hamiltonian import QubitHamiltonian
import numpy as np
import numbers

class BinaryHamiltonian:
    def __init__(self, hamiltonian: QubitHamiltonian):
        self.n_qubit = hamiltonian.n_qubits

        self.binary_terms = [
            BinaryPauliString(
                p.binary(self.n_qubit).binary,
                p.binary(self.n_qubit).coeff) for p in hamiltonian.paulistrings
        ]

    def get_binary_matrix(self):
        matrix = [p.get_binary() for p in self.binary_terms]
        return np.array(matrix)
    
    def get_coeff(self):
        coeff = [p.get_coeff() for p in self.binary_terms]
        return np.array(coeff)

    def nullspace(self):
        pass

    def symplectic_gram_schmidt(self):
        pass 

    def to_qubit_hamiltonian(self):
        pass

class BinaryPauliString:
    def is_binary(self):
        if not isinstance(self.binary, np.ndarray):
            raise TequilaException(
                'Unknown representation of binary vector. Got ' +
                str(self.binary) + ' with type ' + type(self.binary))
        if not all([x == 1 or x == 0 for x in self.binary]):
            raise TequilaException(
                'Not all number in the binary vector is 0 or 1. Got ' +
                str(self.binary))

    def is_coeff(self):
        if not isinstance(self.coeff, numbers.Number):
            raise TequilaException('Unknown coefficients. Got ' +
                                   str(self.coeff))

    def commute(self, other):
        '''
        Determine whether the corresponding pauli-strings of 
        the two binary vectors commute. 
        '''
        inner_product = self.symplectic_inner_product(other)
        if inner_product == 0:
            return True
        elif inner_product == 1:
            return False
        else:
            raise TequilaException('Computed unexpected inner product. Got ' +
                                   str(inner_product))

    def symplectic_inner_product(self, other):
        '''
        Return the symplectic inner product between two BinaryPauli. 

        Return: 0 or 1. 
        '''
        if not self.same_n_qubit(other):
            raise TequilaException(
                'Two binary vectors given do not share same number of qubits. '
            )

        re = self.binary[:self.n_qubit] @ other.binary[self.n_qubit:] \
            + other.binary[:self.n_qubit] @ self.binary[self.n_qubit:]

        return re % 2

    def same_n_qubit(self, other):
        return self.n_qubit == other.n_qubit

    def same_pauli(self, other):
        return all(self.binary == other.binary)

    def to_pauli_strings(self):
        pass

    def get_coeff(self):
        return self.coeff

    def get_binary(self):
        return self.binary
    
    def __init__(self, binary_vector=np.array([0, 0]), coeff=1.0):
        '''
        Stores a list of binary vectors and a list of corresponding coefficients. 
        '''
        self.binary = np.array(binary_vector)
        self.coeff = coeff
        self.n_qubit = len(binary_vector) // 2
        self.is_binary()
        self.is_coeff()
