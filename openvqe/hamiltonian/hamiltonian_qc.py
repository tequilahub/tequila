"""
Interface to get
Quantum Chemistry Hamiltonians for OpenVQE
Derived class of HamiltonianBase: Overwrites the get_hamiltonian function
"""
from openfermion import MolecularData, FermionOperator
from openfermion.transforms import jordan_wigner, get_fermion_operator, bravyi_kitaev
from openvqe.hamiltonian import QubitHamiltonian
from openvqe import BitString, typing


class HamiltonianQC(QubitHamiltonian):

    def __init__(self, molecule: MolecularData, transformation: typing.Union[str, typing.Callable] = None):
        self.molecule = molecule
        if transformation is None:
            self.transformation = jordan_wigner
        elif hasattr(transformation, "lower") and transformation.lower() in ["jordan-wigner", "jw", "j-w", "jordanwigner"]:
            self.transformation = jordan_wigner
        elif hasattr(transformation, "lower") and transformation.lower() in ["bravyi-kitaev", "bk", "b-k", "bravyikitaev"]:
            self.transformation = bravyi_kitaev
        else:
            assert(callable(transformation))
            self.transformation = transformation
        super().__init__(hamiltonian=self.make_hamiltonian())

    def reference_state(self) -> BitString:
        """
        :return: Hartree-Fock Reference as binary-number
        """
        l = [0]*self.n_qubits
        for i in range(self.n_electrons):
            l[i] = 1

        return BitString.from_array(array=l, nbits=self.n_qubits)

    @property
    def hamiltonian(self):
        """
        If the Hamiltonian is not there yet it will be created
        """
        if not hasattr(self, "_hamiltonian") or self._hamiltonian is None:
            self._hamiltonian = self.make_hamiltonian()

        return self._hamiltonian

    def make_hamiltonian(self):
        return self.transformation(self.make_fermionic_hamiltonian())

    @property
    def molecule(self):
        return self._molecule

    @molecule.setter
    def molecule(self, other):
        self._molecule = other
        return self

    @property
    def n_electrons(self):
        """
        Convenience function
        :return: The total number of electrons
        """
        return self.molecule.n_electrons

    @property
    def n_orbitals(self):
        """
        Convenience function
        :return: The total number of (spatial) orbitals (occupied and virtual)
        """
        return self.molecule.n_orbitals

    @property
    def n_qubits(self):
        """
        Convenience function
        :return: Number of qubits needed
        """
        return 2 * self.n_orbitals

    def make_fermionic_hamiltonian(self) -> FermionOperator:
        """
        :return: The fermionic Hamiltonian as InteractionOperator structure
        """
        return get_fermion_operator(self.molecule.get_molecular_hamiltonian())
