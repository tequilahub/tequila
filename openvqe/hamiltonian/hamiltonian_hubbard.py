"""
Interface to get Hubbard-Model Hamiltonians for OpenVQE
"""

from .hamiltonian_base import HamiltonianBase


class HamiltonianHubbard(HamiltonianBase):

    def get_fermionic_hamiltonian(self):
        raise NotImplementedError("get_hamiltonian not yet implemented for "+type(self).__name__)
