from tequila import TequilaException
from tequila.hamiltonian import QubitHamiltonian, PauliString
from tequila.grouping.binary_utils import get_lagrangian_subspace, binary_symplectic_inner_product, binary_solve, binary_phase, gen_single_qubit_term
import numpy as np
import numbers


class BinaryHamiltonian:
    def __init__(self, binary_terms):
        '''
        Initiate from a list of Pauli Strings
        '''
        self.binary_terms = binary_terms

        self.n_qubit = binary_terms[0].get_n_qubit()

        self.n_term = len(self.binary_terms)

    @classmethod
    def init_from_qubit_hamiltonian(cls, hamiltonian: QubitHamiltonian):
        n_qubit = hamiltonian.n_qubits
        binary_terms = [
            BinaryPauliString(
                p.binary(n_qubit).binary,
                p.binary(n_qubit).coeff) for p in hamiltonian.paulistrings
        ]
        return BinaryHamiltonian(binary_terms)

    def get_binary(self):
        matrix = [p.get_binary() for p in self.binary_terms]
        return matrix

    def get_coeff(self):
        coeff = [p.get_coeff() for p in self.binary_terms]
        return coeff

    def get_qubit_wise(self):
        '''
        Return the qubit-wise form of the current binary hamiltonian.
        And the components of corresponding unitary transformation U, 
        where U = prod_i (1/2) ** (1/2) * (lagrangian_basis[i] + new_basis[i])
        '''
        if not self.is_commuting():
            raise TequilaException(
                'Not all terms in the Hamiltonians are commuting.')

        lagrangian_basis = get_lagrangian_subspace(self.get_binary())
        new_basis = self.get_single_qubit_basis(lagrangian_basis)
        lagrangian_basis = [BinaryPauliString(p) for p in lagrangian_basis]
        new_basis = [BinaryPauliString(p) for p in new_basis]
        qubit_wise_hamiltonian = self.basis_transform(lagrangian_basis,
                                                      new_basis)

        # Return the basis in terms of Binary Hamiltonian
        return qubit_wise_hamiltonian, lagrangian_basis, new_basis

    def get_single_qubit_basis(self, lagrangian_basis):
        '''
        Find the single_qubit_basis such that single_qubit_basis[i] anti-commutes
        with lagrangian_basis[i], and commute for all other cases. 
        '''
        dim = len(lagrangian_basis)

        # Free Qubits
        free_qub = [qub for qub in range(dim)]
        pair = []

        for i in range(dim):
            cur_pair = self.find_single_qubit_pair(lagrangian_basis[i],
                                                   free_qub)
            for j in range(dim):
                if i != j:
                    if binary_symplectic_inner_product(
                            cur_pair, lagrangian_basis[j] == 1):
                        lagrangian_basis[j] = (lagrangian_basis[i] +
                                               lagrangian_basis[j]) % 2
            pair.append(cur_pair)
        return pair

    def find_single_qubit_pair(self, cur_basis, free_qub):
        '''
        Find the single qubit pair that anti-commute with cur_basis such that the single qubit is in free_qub 

        Return: Binary vectors representing the single qubit pair
        Modify: Pops the qubit used from free_qub
        '''
        dim = len(cur_basis) // 2
        for idx, qub in enumerate(free_qub):
            for term in range(3):
                pair = gen_single_qubit_term(dim, qub, term)
                # if anticommute
                if (binary_symplectic_inner_product(pair, cur_basis) == 1):
                    free_qub.pop(idx)
                    return pair

    def basis_transform(self, old, new):
        '''
        Transforms the given hamiltonian from the old basis to the new basis.
        
        Return: The transformed Binary hamiltonian
        '''
        return BinaryHamiltonian(
            [p.basis_transform(old, new) for p in self.binary_terms])

    def is_commuting(self):
        '''
        Return whether all terms in the Hamiltonian are commuting
        '''
        for i in range(self.n_term):
            for j in range(i + 1, self.n_term):
                if not self.binary_terms[i].commute(self.binary_terms[j]):
                    return False
        return True

    def to_qubit_hamiltonian(self):
        qub_ham = QubitHamiltonian.init_zero()
        for p in self.binary_terms:
            qub_ham += QubitHamiltonian.init_from_paulistring(
                p.to_pauli_strings())
        return qub_ham


class BinaryPauliString:
    def __init__(self, binary_vector=np.array([0, 0]), coeff=1.0):
        '''
        Stores a list of binary vectors and a list of corresponding coefficients. 
        '''
        self.binary = np.array(binary_vector)
        self.coeff = coeff
        self.n_qubit = len(binary_vector) // 2
        self.is_binary()
        self.is_coeff()

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
        inner_product = binary_symplectic_inner_product(
            self.binary, other.binary)

        if inner_product == 0:
            return True
        elif inner_product == 1:
            return False
        else:
            raise TequilaException('Computed unexpected inner product. Got ' +
                                   str(inner_product))

    def basis_transform(self, old, new):
        '''
        Transform the pauli string from old to new binary basis.

        Return: Pauli string in the new basis.
        '''
        old_basis_coeff = binary_solve([p.get_binary() for p in old],
                                       self.binary)
        original_pauli_vec = np.zeros(self.n_qubit * 2)
        new_pauli_vec = np.zeros(self.n_qubit * 2)
        phase = 1
        for i, i_coeff in enumerate(old_basis_coeff):
            if i_coeff == 1:
                phase *= binary_phase(original_pauli_vec, old[i].get_binary(),
                                      self.n_qubit)
                original_pauli_vec = (original_pauli_vec +
                                      old[i].get_binary()) % 2
                new_pauli_vec = (new_pauli_vec + new[i].get_binary()) % 2

        new_pauli_str = BinaryPauliString(new_pauli_vec)
        new_pauli_str.set_coeff(self.coeff / phase)
        return new_pauli_str

    def same_n_qubit(self, other):
        return self.n_qubit == other.n_qubit

    def same_pauli(self, other):
        return all(self.binary == other.binary)

    def to_pauli_strings(self):
        data = {}
        for i in range(self.n_qubit):
            if (self.binary[i] == 1 and self.binary[i + self.n_qubit] == 0):
                data[i] = 'X'
            elif (self.binary[i] == 1 and self.binary[i + self.n_qubit] == 1):
                data[i] = 'Y'
            elif (self.binary[i] == 0 and self.binary[i + self.n_qubit] == 1):
                data[i] = 'Z'
        return PauliString(data, self.coeff)

    def get_coeff(self):
        return self.coeff

    def set_coeff(self, new_coeff):
        self.coeff = new_coeff

    def get_binary(self):
        return self.binary

    def get_n_qubit(self):
        return self.n_qubit
