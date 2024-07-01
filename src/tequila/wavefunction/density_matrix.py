# Module for handling density matrices

import numpy
from tequila.utils.bitstrings import BitNumbering, BitString, initialize_bitstring
from tequila import TequilaException, QubitHamiltonian
from openfermion import expectation, get_sparse_operator, variance
import scipy
from scipy.sparse import csc_matrix as csc
import copy

def reverse_density(density, n_qubits):
    density_qubits = numpy.reshape(density, tuple([2]*2*n_qubits))
    for i in range(n_qubits//2):
        density_qubits = numpy.swapaxes(numpy.swapaxes(density_qubits, i, n_qubits - i -1), n_qubits + i, 2*n_qubits -i -1)
    rev_density = numpy.reshape(density_qubits, (2**n_qubits, 2**n_qubits))
    return rev_density

class DensityMatrix:
    """
    Stores density matrix as a scipy.sparse object
    """
    numbering = BitNumbering.LSB

    def __init__(self, density = None, n_qubits = None, numbering = BitNumbering.MSB):
        self._n_qubits = n_qubits
        if density is None:
            density = numpy.zeros(shape=(2**n_qubits, 2**n_qubits), dtype=numpy.complex128)
            return self.from_array(density_matrix=density, n_qubits=n_qubits, numbering = numbering)
        else:
            self.density = density
            self._n_qubits = int(numpy.log2(density.shape[0]))
        
        if self.numbering != numbering:
            self.flip_qubit_order() #store by default in LSB, in conflict with QubitWavefunction!
        return
    
    @classmethod
    def from_array(cls, density_matrix, n_qubits = None, numbering = BitNumbering.MSB):
        sparse_density = csc(density_matrix)
        return cls(density = sparse_density, n_qubits = n_qubits, numbering = numbering) 
    
    def get_density(self):
        return self.density
    
    def get_density_as_array(self):
        return self.density.toarray()

    def trace(self):
        return self.density.trace()
    
    @property
    def n_qubits(self):
        if self._n_qubits is not None:
            return self._n_qubits
        else:
            return int(numpy.log2(self.density.shape[0]))
    
    @n_qubits.setter
    def n_qubits(self, value):
        self._n_qubits = value
    
    def flip_qubit_order(self):
        """
        Flip qubit order, inplace
        Converts to array and then reorders as the operations are faster on numpy.array objects
        """
        self.density = csc(reverse_density(self.get_density_as_array(), n_qubits = self._n_qubits))
        return

    def expectation(self, operator):
        """
        Returns expectation of (sparse) operator at density
        """
        sparse_density = self.density

        return expectation(operator, sparse_density)

    def QubitHamiltonian_expectation(self, hamiltonian: QubitHamiltonian):
        """
        Returns expectation of hamiltonian object
        """
        assert hamiltonian.n_qubits <= self.n_qubits, 'Qubit Hamiltonian is larger than density operator'

        sparse_density = self.density
        sparse_hamiltonian = get_sparse_operator(hamiltonian.to_openfermion(), self.n_qubits)
        return expectation(sparse_hamiltonian, sparse_density)
    
    def QubitHamiltonian_variance(self, hamiltonian: QubitHamiltonian):
        """
        Returns variance of hamiltonian object
        """
        assert hamiltonian.n_qubits <= self.n_qubits, 'Qubit Hamiltonian is larger than density operator'

        sparse_density = self.density
        sparse_hamiltonian = get_sparse_operator(hamiltonian.to_openfermion(), self.n_qubits)
        return variance(sparse_hamiltonian, sparse_density)
    
    def normalize(self):
        """
        Normalizes density, inplace
        """
        N = self.trace()
        self.density = self.density/N
        return
    
    def multiply_projector(self, projector, one_sided = False):
        """
        Multiply density by projector
        one_sided: if True, then saves P*rho/N else P*rho*P/N' - useful for timesaving by reducing matrix operations in some cases
        """
        self.density = projector @ self.density

        if not one_sided:
            self.density = self.density @ projector

        self.normalize()
        return self

    def apply_keymap(self, keymap, initial_state: BitString = None):
        if hasattr(keymap, 'numbering'):
            #numbered keymap like MSB, LSB
            if keymap.numbering != self.numbering:
                self.flip_qubit_order()
                return self
        else:
            #general keymap #todo
            return
        return
    
    def __add__(self, other):
        if self.n_qubits != other.n_qubits:
            raise TequilaException('Incompatible densities of {} and {} qubits added.'.format(self.n_qubits, other.n_qubits))
        
        return DensityMatrix(self.density + other.density, n_qubits=self.n_qubits, numbering=self.numbering)

    def __iadd__(self, other):
        if self.n_qubits != other.n_qubits:
            raise TequilaException('Incompatible densities of {} and {} qubits added.'.format(self.n_qubits, other.n_qubits))
        
        if self.numbering != other.numbering:
            other_copy = copy.deepcopy(other)
            other_copy.flip_qubit_order()
            self.density = self.density + other_copy.density
        else:
            self.density = self.density + other.density
        return self

    def __sub__(self, other):
        if self.n_qubits != other.n_qubits:
            raise TequilaException('Incompatible densities of {} and {} qubits subtracted.'.format(self.n_qubits, other.n_qubits))
        
        return DensityMatrix(self.density - other.density, n_qubits=self.n_qubits, numbering=self.numbering)
    
    def __isub__(self, other):
        if self.n_qubits != other.n_qubits:
            raise TequilaException('Incompatible densities of {} and {} qubits subtracted.'.format(self.n_qubits, other.n_qubits))
        
        if self.numbering != other.numbering:
            other_copy = copy.deepcopy(other)
            other_copy.flip_qubit_order()
            self.density = self.density - other_copy.density
        else:
            self.density = self.density - other.density
        return self

    def __rmult__(self, other):
        if self.n_qubits != other.n_qubits:
            raise TequilaException('Incompatible densities of {} and {} qubits multiplied.'.format(self.n_qubits, other.n_qubits))
        
        return DensityMatrix(density = self.density @ other.density, n_qubits = self.n_qubits, numbering=self.numbering)
    
    def __ne__(self, other):
        return not self.__eq__(other)

    def __eq__(self, other):
        return self.isclose(other)
    
    def isclose(self, other, tol=1e-6):
        """
        Returns True if two densities are close upto tol
        """
        dist = self.trace_distance(self, other, 0)
        if dist <= tol:
            return True
        else:
            return False
    
    def trace_distance(self, other, tol=0):
        """
        Returns Trace distance (1 norm of difference) of two DensityMatrix objects, assumes hermiticity
        d(\rho, \sigma) = Tr(\sqrt((\rho - \sigma)^ (\rho - \sigma))) = 0.5\sum_i |lambda_i| where \lambda_i is the eigen values obtained by svd
        """
        density_diff = self - other
        return 0.5*(sum(abs(density_diff.spectrum(tol))))

    def __repr__(self):
        return 'Tequila DensityMatrix over {} qubits stored as sparse matrix\n{}'.format(self.n_qubits, self.get_density_as_array().__repr__())
    
    def __str__(self):
        return self.__repr__()

    def is_positive_semi_definite(self):
        spectrum = self.spectrum()
        if any(spectrum < 0):
            return False
        return True
    
    def spectrum(self, tol=0):
        '''
        returns spec(self.density)
        '''
        return scipy.sparse.linalg.svds(self.density, k = 2**self.n_qubits, tol= tol)