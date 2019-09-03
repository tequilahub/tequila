"""
(Quantum) Ising Hamiltonian (with) without transversal Field
"""
from openvqe import OpenVQEException
from openvqe.hamiltonian import QubitHamiltonian


class IsingHamiltonian(QubitHamiltonian):

    def __init__(self, n_qubits, g=None):
        """
        :param n_qubits: number of sites
        :param g: strenth of the transversal field, default is None
        """
        self._n = n_qubits
        self._g = g

    @property
    def g(self):
        if self._g is None:
            return 0
        else:
            return self._g

    @property
    def g(self, other):
        self._g = other
        return self
