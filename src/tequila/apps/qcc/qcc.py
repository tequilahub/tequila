# Qubit Coupled Cluster
# as described in Ryabinkin, Yen, Genin, Izmaylov: https://doi.org/10.1021/acs.jctc.8b00932
# and Ryabinkin, Lang, Genin, Izmaylov: https://doi.org/10.1021/acs.jctc.9b01084

import itertools
import numpy as np
from math import floor, log10
from openfermion import QubitOperator, count_qubits, commutator

import tequila as tq
from tequila.circuit import QCircuit
from tequila.hamiltonian.qubit_hamiltonian import BinaryPauli
from tequila.grouping.binary_utils import binary_phase
from tequila.grouping.binary_rep import BinaryHamiltonian, BinaryPauliString


def are_equal_bitwise_check(binary_vec1, binary_vec2):
    n = len(binary_vec1)
    for i in range(n):
        if binary_vec1[i] != binary_vec2[i]:
            return False
    return True

def z_expectation(binary_occupations, binary_z_placements):
    return 1 - 2 * (len(np.where(binary_occupations + binary_z_placements == 2)[0]) % 2)

def even_x_y_overlap_phase(z_vector, x_vector):
    return 1 - 2 * ((len(np.where(x_vector + z_vector == 2)[0]) // 2) % 2)

class IterativeQCC:
    def __init__(self, molecule):
        #get hamiltonian in TQ format and binary format

        self.hamiltonian = molecule.make_hamiltonian()
        self.n_qubits = 2 * molecule.n_orbitals
        self.ref_circ = molecule.prepare_reference()
        self.ref_occs = np.zeros(2 * molecule.n_orbitals)
        np.put(self.ref_occs, self.ref_circ.qubits, 1)
        self.binary_repr = [pauli.binary(n_qubits = self.n_qubits) for pauli in self.hamiltonian.paulistrings]
        self.iteration_energies = []

    def __get_x_z_factors(self):

        x_z_factors = []
        for binary_pauli in self.binary_repr:
            pauli_xs = binary_pauli.binary[:self.n_qubits]
            pauli_zs = binary_pauli.binary[self.n_qubits:]
            coeff = binary_pauli.coeff
            new_pauli_xs = True
            for element in x_z_factors:
                if are_equal_bitwise_check(element[0], pauli_xs):
                    element[1].append((pauli_zs, coeff))
                    new_pauli_xs = False
                    break
            if new_pauli_xs:
                x_z_factors.append((pauli_xs, [(pauli_zs, coeff)]))

        self.x_z_factors = x_z_factors


    def get_pauli_from_x_string(self, x_string, type="min_weight", format="tequila", sampling_index=0):

        """
        Given a binary list describing the x-components of a Pauli word in the binary vector representation,
        returns a Pauli word with an odd number of Pauli-Y operations.

        Arguments:
            x_string (np.array[bool]): A binary vector describing the x-components of a Pauli word in the
                binary vector representation.
            type (str): can currently only be set to "min_weight" (default). This is the 'canonical' choice made in
                QCC for sampling specific generators from a direct interaction set partition.
            format (str): can be set to "tequila" (default) or "openfermion". If set to "tequila", returns the
                Pauli word as a string describing a Pauli word in tequila format, e.g. "Y(0) X(1)".
                If set to "openfermion", returns a QubitOperator instance of the Pauli word.
            sampling_index (int): Given a sampling scheme via 'type' keyword, there are multiple generators
                within the direct interaction set partition to sample from. To avoid redundant generators
                entering the QCC ansatz, different generators may be sampled via this keyword. For exanglee,
                given `x_string = np.array([1,0])` and `type='minimal'`, the Pauli word with `sampling_index = 0`
                will return 'Y(0) X(1)', while `sampling_index = 1` will return 'X(0) Y(1)'. Default value is 0.

        Returns:
            str or QubitOperator: A string describing the Pauli word if input `format` is set to "tequila", otherwise
                                  with `format` set to "openfermion", a QubitOperator instance of the Pauli word is
                                  returned.

        Raises:
            ValueError: if `format` is not "tequila" or "openfermion".
        """

        type = type.lower()
        if type not in ["min_weight"]:  # other sampling schemes can be implemented
            raise ValueError(
                'Optional argument `type` must be "min_weight" (for minimal weight generators), instead got {}'.format(
                    type
                ),
            )

        pauli_string = ""
        x_indices = x_string.nonzero()[0]
        n_x_indices = len(x_indices)

        if not x_string.any():  # identifies zero vector
            return QubitOperator("")

        if type == "min_weight":

            n_x_indices = len(x_indices)
            sampling_index = sampling_index % 2 ** (n_x_indices - 1)
            num_y = 1
            y_placements = list(itertools.combinations(x_indices, num_y))
            n_combs = n_x_indices
            effective_idx = sampling_index

            while effective_idx >= n_combs:
                effective_idx -= n_combs
                num_y += 2
                y_placements = list(itertools.combinations(x_indices, num_y))
                n_combs = len(y_placements)
            y_placement = y_placements[effective_idx]
            for idx in x_indices:
                if idx in y_placement:
                    pauli_string += "Y({}) ".format(idx)
                else:
                    pauli_string += "X({}) ".format(idx)

        if format == "tequila":
            return tq.PauliString.from_string(pauli_string.replace(' ',''))
        elif format == "openfermion":
            return QubitOperator("".join(c for c in pauli_string if c not in "()"))
        else:
            raise ValueError(
                "Keyword `format` must be `tequila` or `openfermion`, instead got {}".format(format)
            )


    def select(self, n_gen = 1, grad_threshold=1e-6):

        self.__get_x_z_factors()
        print('\nX-strings in new code:')
        grad_partitions = []
        for x_term, ising_terms in self.x_z_factors:
            if not are_equal_bitwise_check(x_term, np.zeros(self.n_qubits)):
                grad = abs(sum([coeff * even_x_y_overlap_phase(z_term, x_term) * z_expectation(self.ref_occs, z_term) for z_term, coeff in ising_terms]))
                if grad >= grad_threshold:
                    grad_partitions.append((x_term, grad))

        grad_partitions.sort(key=lambda x: x[1], reverse=True)
        self.grad_partitions = grad_partitions
        for x, grad in grad_partitions:
            print('{} : {}'.format(x, grad))

        n_dis = len(grad_partitions)
        generators = []
        for idx in range(n_gen):
            part_to_sample = grad_partitions[idx % n_dis]
            sampling_idx = idx // n_dis
            generators.append(self.get_pauli_from_x_string(
                part_to_sample[0], type="min_weight", sampling_index=sampling_idx
            ))

        print('Selected QCC generators:')
        for generator in generators:
            print(generator)
        self.generators = {'g{}'.format(idx) : generators[idx] for idx in range(len(generators))}
        return generators

    def get_qcc_unitary(self):

        U_qcc = QCircuit()
        for idx in range(len(self.generators)):
            U_qcc += tq.gates.ExpPauli(angle="t{}".format(idx), paulistring=self.generators['g{}'.format(idx)])
        self.qcc_circ = U_qcc

        return U_qcc

    def optimize(self, method = 'COBYLA'):
        #optimization performed in TQ format

        qcc_energy = tq.ExpectationValue(H = self.hamiltonian, U = self.ref_circ + self.qcc_circ)

        #could add a codeblock to solve for optimal qcc energy by linear variational principle
        #if n_gens is less than ~6?
        result = tq.minimize(
            objective=qcc_energy, method="COBYLA", initial_values={k: 0.01 for k in qcc_energy.extract_variables()}
        )

        print('Optimized QCC parameters:\n {}'.format(result.variables))
        self.angles = result.variables
        self.energy = result.energy
        self.iteration_energies.append(self.energy)
        return result

    def update(self, compression_threshold = 1e-8):
        #dressing is performed in binary format

        generators_binary = {'g{}'.format(idx) : self.generators['g{}'.format(idx)].binary(n_qubits = self.n_qubits) for idx in range(len(self.generators))}
        for idx in range(len(self.generators) - 1, -1, -1):
            gen_binary = generators_binary['g{}'.format(idx)]
            gen_x = gen_binary.binary[:self.n_qubits]
            gen_z = gen_binary.binary[self.n_qubits:]
            angle = self.angles['t{}'.format(idx)]
            introduced_terms = []
            for h_idx in range(len(self.binary_repr)):
                h_term = self.binary_repr[h_idx]
                h_term_x = h_term.binary[:self.n_qubits]
                h_term_z = h_term.binary[self.n_qubits:]
                h_coeff =  h_term.coeff
                if (np.dot(h_term_x, gen_z) + np.dot(h_term_z, gen_x)) % 2 == 1: #non-commutative
                    self.binary_repr[h_idx] = BinaryPauli(h_coeff * np.cos(angle), h_term.binary)
                    phase = binary_phase(gen_binary.binary, h_term.binary, self.n_qubits)
                    product_binary = (gen_binary.binary + h_term.binary) % 2
                    new_coeff = 1j * h_coeff * phase * np.sin(angle)
                    introduced_terms.append(BinaryPauli(new_coeff, product_binary))
            self.binary_repr += introduced_terms

        #update self.hamiltonian from self.binary_repr
        binary_h = BinaryHamiltonian([BinaryPauliString(binary_pauli.binary, binary_pauli.coeff) for binary_pauli in self.binary_repr])
        self.hamiltonian = binary_h.to_qubit_hamiltonian()
        self.hamiltonian = self.hamiltonian.simplify(threshold=compression_threshold)
        #reinstantiating `self.binary_repr` seems redundant here, but it lets us consolidate identical binary strings
        #without explicitly performing a search over the binary terms earlier:
        self.binary_repr = [pauli.binary(n_qubits = self.n_qubits) for pauli in self.hamiltonian.paulistrings]
        print('Number of terms in iQCC Hamiltonian: {}'.format(len(self.hamiltonian)))
        return self.hamiltonian

    def grad_norm(self):
        #needs implement
        return 1

    def do_iteration(self, n_gen=1):

        self.select(n_gen=n_gen)
        self.get_qcc_unitary()
        self.optimize()
        self.update()
