from tequila import TequilaException
from tequila.hamiltonian import QubitHamiltonian, PauliString
from tequila.grouping.binary_utils import get_lagrangian_subspace, binary_symplectic_inner_product, binary_solve, binary_phase, gen_single_qubit_term, largest_first, recursive_largest_first
import numpy as np
import tequila as tq
import numbers


class BinaryHamiltonian:
    def __init__(self, binary_terms):
        '''
        Initiate from a list of Binary Pauli Strings
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

    def single_qubit_form(self):
        '''
        Returns
        ----------
        hamiltonian : BinaryHamiltonian
            The original hamiltonian in qubit-wise commuting form
        lagrangian_basis : list of BinaryPauliStrings 
            Represents the basis of original Hamiltonian
        new_basis : list of BinaryPauliStrings 
            Represents the basis of new Hamiltonian
        '''
        lagrangian_basis = get_lagrangian_subspace(self.get_binary())
        new_basis = self.get_single_qubit_basis(lagrangian_basis)
        lagrangian_basis = [BinaryPauliString(p) for p in lagrangian_basis]
        new_basis = [BinaryPauliString(p) for p in new_basis]
        qubit_wise_hamiltonian = self.basis_transform(lagrangian_basis,
                                                      new_basis)
        return qubit_wise_hamiltonian, lagrangian_basis, new_basis

    def z_form(self):
        '''
        Parameters
        ----------
        self : BinaryHamiltonian 
            a qubit-wise commuting hamiltonian

        Modifies
        ----------
        self : BinaryHamiltoina
            The original hamiltonian but in qubit-wise commuting form with all z

        Returns
        ----------
        U : QCircuit
            the single qubit transformation that rotates each term to z
        '''
        U = tq.QCircuit()
        non_z = {}
        for p in self.binary_terms:
            for qub in range(self.n_qubit):
                z_data = {qub: 'z'}
                if p.has_x(qub):
                    p.set_z(qub)
                    non_z[qub] = 'x'
                elif p.has_y(qub):
                    p.set_z(qub)
                    non_z[qub] = 'y'
        
        for qub, term in non_z.items():
            xy_data = {qub: term}
            z_data = {qub: 'z'}
            U += tq.gates.ExpPauli(angle = -tq.numpy.pi/2, paulistring=tq.PauliString(z_data))
            U += tq.gates.ExpPauli(angle = -tq.numpy.pi/2, paulistring=tq.PauliString(xy_data))
            U += tq.gates.ExpPauli(angle = -tq.numpy.pi/2, paulistring=tq.PauliString(z_data))

        return U

    def get_qubit_wise(self, binary = False):
        '''
        Return the qubit-wise form of the current binary hamiltonian.
        And the unitary transformation U, 
        where U = prod_i (1/2) ** (1/2) * (lagrangian_basis[i] + new_basis[i])
        
        Parameters
        ----------
        binary : determines whether the returned qwc hamiltonian is binary or qubit hamiltonian


        Returns
        -------
        hamiltonian : BinaryHamiltonian or QubitHamiltonian
            The original hamiltonian in qubit-wise commuting form
        U : QCircuit
            The unitary circuit that transforms the original hamiltonian into qwc form
        '''
        if not self.is_commuting():
            raise TequilaException(
                'Not all terms in the Hamiltonians are commuting.')
        if self.is_qubit_wise_commuting():
            qubit_wise_hamiltonian = self
            qwc_u = tq.QCircuit()
        else: 
            qubit_wise_hamiltonian, lagrangian_basis, new_basis = self.single_qubit_form()

            # Constructing the unitary that rotates into qubit-wise parts
            qwc_u = tq.QCircuit()
            for i in range(len(lagrangian_basis)):
                sigma = lagrangian_basis[i].to_pauli_strings()
                tau = new_basis[i].to_pauli_strings()
                qwc_u += tq.gates.ExpPauli(angle=-tq.numpy.pi/2, paulistring=sigma)
                qwc_u += tq.gates.ExpPauli(angle=-tq.numpy.pi/2, paulistring=tau)
                qwc_u += tq.gates.ExpPauli(angle=-tq.numpy.pi/2, paulistring=sigma)
        
        single_qub_u = qubit_wise_hamiltonian.z_form()
        # Return the basis in terms of Binary Hamiltonian
        if binary:
            return qubit_wise_hamiltonian, qwc_u + single_qub_u
        else:
            return qubit_wise_hamiltonian.to_qubit_hamiltonian(), qwc_u + single_qub_u

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

    def is_qubit_wise_commuting(self):
        '''
        Return whether all terms in the Hamiltonian are qubit-wise commuting 
        '''
        # Keep a dictionary of qubit-wise term found
        qubit_term = {}
        for i in range(self.n_term):
            for qub in range(self.n_qubit):
                cur_binary_term = self.binary_terms[i].get_binary()
                cur_qub_term = (cur_binary_term[qub], cur_binary_term[qub+self.n_qubit])
                if cur_qub_term != (0, 0):
                    if qub not in qubit_term:
                        qubit_term[qub] = cur_qub_term
                    else:
                        if not qubit_term[qub] == cur_qub_term:
                            return False
        return True
                

    def to_qubit_hamiltonian(self):
        qub_ham = QubitHamiltonian()
        for p in self.binary_terms:
            qub_ham += QubitHamiltonian.from_paulistrings(
                p.to_pauli_strings())
        return qub_ham

    def anti_commutativity_matrix(self):
        """
        Return an adjacency matrix. If term[i] and term[j] anticommute,
        the entry [i][j] is 1, else the entry is 0
        """
        n = self.n_qubit
        matrix = np.array(self.get_binary())
        gram = np.block([[np.zeros((n,n)), np.eye(n)], [np.eye(n), np.zeros((n,n))]])
        return matrix @ gram @ matrix.T % 2

    def commuting_groups(self, method='rlf'):
        """
        Return the partitioning of the hamiltonian into commuting groups.
        List of BinaryHamiltonian's
        """
        terms = self.binary_terms
        n = self.n_term
        cg = self.anti_commutativity_matrix()

        if method == 'lf':
            colors = largest_first(terms, n, cg)
        elif method == 'rlf':
            colors = recursive_largest_first(terms, n, cg)
        else:
            raise TequilaException(f"There is no algorithm {method}")
        return [BinaryHamiltonian(value) for key, value in colors.items()]


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

    def has_x(self, i):
        return self.binary[i] == 1 and self.binary[i + self.n_qubit] == 0

    def has_y(self, i):
        return self.binary[i] == 1 and self.binary[i + self.n_qubit] == 1

    def has_z(self, i):
        return self.binary[i] == 0 and self.binary[i + self.n_qubit] == 1
    
    def set_x(self, i):
        '''
        Set the ith qubit to having x
        '''
        self.binary[i] = 1
        self.binary[i + self.n_qubit] = 0
    
    def set_y(self, i):
        '''
        Set the ith qubit to having y
        '''
        self.binary[i] = 1
        self.binary[i + self.n_qubit] = 1

    def set_z(self, i):
        '''
        Set the ith qubit to having z
        '''
        self.binary[i] = 0
        self.binary[i + self.n_qubit] = 1
    
    def to_pauli_strings(self):
        data = {}
        for i in range(self.n_qubit):
            if self.has_x(i):
                data[i] = 'X'
            elif self.has_y(i):
                data[i] = 'Y'
            elif self.has_z(i):
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
