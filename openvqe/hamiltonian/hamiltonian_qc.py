"""
Interface to get
Quantum Chemistry Hamiltonians for OpenVQE
"""

from openvqe.parameters import ParametersQC
from openfermion.hamiltonians import MolecularData
from .hamiltonian_base import Hamiltonian
import openfermion
import openfermionpsi4


class HamiltonianQC(Hamiltonian):

    molecule: MolecularData = None

    def __init__(self, parameters: ParametersQC):
        assert(isinstance(parameters, ParametersQC))
        self.parameters=parameters
        self.molecule=self.get_molecule(parameters)


    def get_Hamiltonian(self) -> openfermion.QubitOperator:
        return openfermion.jordan_wigner(openfermion.get_fermion_operator(self.get_molecular_Hamiltonian()))

    def get_molecular_Hamiltonian(self) -> openfermion.InteractionOperator:
        # fast return if possible
        if self.molecule is None:
            self.molecule = self.get_molecule(self.parameters)
        return self.molecule.get_molecular_hamiltonian()

    @staticmethod
    def get_molecule(parameters: ParametersQC) -> openfermion.InteractionOperator:
        molecule = MolecularData(geometry=parameters.get_geometry(),
                                 basis=parameters.basis_set,
                                 multiplicity=parameters.multiplicity,
                                 charge=parameters.charge,
                                 description=parameters.description,
                                 filename=parameters.filename)

        # try to load
        do_compute = True
        if parameters.filename:
            try:
                import os
                if os.path.exists(parameters.filename):
                    molecule.load()
                    do_compute = False
            except OSError:
                print("loading molecule from file=", parameters.filename, " failed. Try to recompute")
                do_compute = True

        if do_compute:
            molecule = openfermionpsi4.run_psi4(molecule, **parameters.psi4.__dict__)
        molecule.save()

        return molecule
