from openvqe import OpenVQEException
from openvqe.openvqe_abc import OpenVQEParameters
from openvqe.ansatz import ManyBodyAmplitudes
from openvqe import dataclass
from openfermion import MolecularData, FermionOperator
from openfermion.transforms import jordan_wigner, bravyi_kitaev, get_fermion_operator
from openfermionpsi4._psi4_conversion_functions import parse_psi4_ccsd_amplitudes
from openfermionpsi4 import run_psi4
from openvqe.hamiltonian import QubitHamiltonian
from openvqe.quantumchemistry.qc_base import ParametersQC
from openvqe import BitString, typing

class OpenVQEEPsi4Exception(OpenVQEException):
    pass


@dataclass
class ParametersPsi4:
    run_scf: bool = True
    run_mp2: bool = False
    run_cisd: bool = False
    run_ccsd: bool = False
    run_fci: bool = False
    verbose: bool = False
    tolerate_error: bool = False
    delete_input: bool = False
    delete_output: bool = False
    memory: int = None



class PSI4:

    def __init__(self, parameters: ParametersPsi4 = None):
        self.parameters = parameters

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
        return self.transformation(self.get_fermionic_hamiltonian())

    @property
    def molecule(self):
        if not hasattr(self, "_molecule") or self._molecule is None:
            self._molecule = self.make_molecule(self.parameters)

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

    def get_fermionic_hamiltonian(self) -> FermionOperator:
        """
        :return: The fermionic Hamiltonian as InteractionOperator structure
        """
        return get_fermion_operator(self.molecule.get_molecular_hamiltonian())

    @staticmethod
    def make_molecule(parameters: ParametersQC, parameters_psi4: ParametersPsi4) -> MolecularData:
        """
        Creates a molecule in openfermion format by running psi4 and extracting the data
        Will check for previous outputfiles before running
        :param parameters: An instance of ParametersQC, which also holds an instance of ParametersPsi4 via parameters.psi4
        The molecule will be saved in parameters.filename, if this file exists before the call the molecule will be imported from the file
        :return: the molecule in openfermion.MolecularData format
        """
        print("geom=", parameters.get_geometry())
        print("description=", parameters.description)
        print(parameters)
        molecule = MolecularData(geometry=parameters.get_geometry(),
                                 basis=parameters.basis_set,
                                 multiplicity=parameters.multiplicity,
                                 charge=parameters.charge,
                                 description=parameters.description.strip('\n'),  # '\n' causes openfermion to chrash
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
            molecule = run_psi4(molecule, run_scf=parameters_psi4.run_scf,
                                run_mp2=parameters_psi4.run_mp2,
                                run_ccsd=parameters_psi4.run_ccsd,
                                run_cisd=parameters_psi4.run_cisd,
                                run_fci=parameters_psi4.run_fci,
                                verbose=parameters_psi4.verbose,
                                tolerate_error=parameters_psi4.tolerate_error,
                                delete_input=parameters_psi4.delete_input,
                                delete_output=parameters_psi4.delete_output,
                                memory=parameters_psi4.memory)

        molecule.save()
        print("file was ", molecule.filename)
        return molecule

    def parse_ccsd_amplitudes(self, filename=None) -> ManyBodyAmplitudes:
        if filename is None:
            filename = self.parameters.filename

        from os import path, access, R_OK
        file_exists = path.isfile("./" + filename) and access("./" + filename, R_OK)

        # make sure the molecule is there, i.e. psi4 has been run
        # and that the output was kept, as well as CCSD was actually computed
        if not file_exists or (
                self._molecule is None or self.parameters.psi4.run_ccsd == False or self.parameters.psi4.delete_output):
            self.parameters.filename = filename.strip(".out")
            self.parameters.psi4.run_ccsd = True
            self.parameters.psi4.delete_output = False
            self._molecule = self.make_molecule(self.parameters)

        # adapting to the openfermion parser
        if ".out" not in filename:
            filename += ".out"

        singles, doubles = parse_psi4_ccsd_amplitudes(number_orbitals=self.n_orbitals * 2,
                                                      n_alpha_electrons=self.n_electrons // 2,
                                                      n_beta_electrons=self.n_electrons // 2,
                                                      psi_filename=filename)

        return ManyBodyAmplitudes(one_body=singles, two_body=doubles)

    def parse_mp2_amplitudes(self):
        raise NotImplementedError("not implemented yet")
