import numpy as np
from tequila import QubitHamiltonian, ExpectationValue, BitString, PauliString, QCircuit


class PauliClique:
    """
    Small Helper Class for cliques of computing Pauli operators
    that are combined to a diagonal operator
    Op = c_i h_i
    where each h_i is a PauliString of Units and Pauli-Z
    class.U transforms into the eigenbasis where Op is diagonal
    """

    def __init__(self, coeff, H, U, n_qubits):
        assert H.is_all_z()
        self.n_qubits = n_qubits
        self.U = U
        self.paulistrings = H.paulistrings
        self.coeff = coeff

    def compute_eigenvalues(self, sort=True):
        """
        Returns
            The eigenvalues of the diagonal operator
        -------
        """
        n_qubits = self.n_qubits
        eig = np.asarray([0.0 for n in range(2 ** n_qubits)], dtype=float)
        for ps in self.paulistrings:
            x = np.asarray([1.0 for n in range(2 ** n_qubits)], dtype=int)
            paulis = [[1, 1]] * n_qubits
            for d in ps.keys():
                try:
                    paulis[d] = [1, -1]
                except:
                    raise Exception("weird {} with len={} with d={}".format(paulis, len(paulis), d))
            for i in range(2 ** n_qubits):
                binary_array = BitString.from_int(integer=i, nbits=n_qubits).array
                for j, k in enumerate(binary_array):
                    x[i] *= paulis[j][k]
            eig += ps.coeff * x

        if sort:
            eig = sorted(eig)

        return eig

    def normalize(self):
        """
        Returns
            Normalized PauliClique with eigenvalues between -1 and 1
        -------
        """

        eig = self.compute_eigenvalues(sort=True)
        lowest = eig[0]
        highest = eig[-1]
        highest_abs = max([abs(lowest), abs(highest)])
        normalized_ps = []
        for ps in self.paulistrings:
            normalized_ps.append(PauliString(coeff=ps.coeff / highest_abs, data=ps._data))

        return PauliClique(coeff=self.coeff * highest_abs, H=QubitHamiltonian.from_paulistrings(normalized_ps),
                           U=self.U, n_qubits=self.n_qubits)

    def naked(self):
        return PauliClique(coeff=1.0, H=self.H, U=self.U, n_qubits=self.n_qubits)

    def __len__(self):
        return len(self.paulistrings)

    @property
    def H(self):
        return QubitHamiltonian.from_paulistrings(self.paulistrings)


def make_paulicliques(H):
    E = ExpectationValue(H=H, U=QCircuit(), optimize_measurements=True)
    result = []
    for clique in E.get_expectationvalues():
        result.append(PauliClique(H=clique.H[0], U=clique.U, coeff=1.0, n_qubits=H.n_qubits))
    return result
