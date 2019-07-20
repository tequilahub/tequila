"""
Interface to get
Quantum Chemistry Hamiltonians for OpenVQE
Derived class of HamiltonianBase: Overwrites the get_hamiltonian function
"""

from openvqe.abc import OpenVQEParameters, parametrized
from dataclasses import dataclass
from openfermion.hamiltonians import MolecularData
from .hamiltonian_base import HamiltonianBase, ParametersHamiltonian
import openfermion
import openfermionpsi4
import numpy as np


@dataclass
class ParametersPsi4(OpenVQEParameters):
    run_scf: bool = True
    run_mp2: bool = False
    run_cisd: bool = False
    run_ccsd: bool = False
    run_fci: bool = False
    verbose: bool = False
    tolerate_error: bool = False
    delete_input: bool = False
    delete_output: bool = False
    memory: int = 8000


@dataclass
class ParametersQC(ParametersHamiltonian):
    """
    Specialization of ParametersHamiltonian
    Parameters for the HamiltonianQC class
    """
    psi4: ParametersPsi4 = ParametersPsi4()
    basis_set: str = ''  # Quantum chemistry basis set
    geometry: str = ''  # geometry of the underlying molecule (units: Angstrom!), this can be a filename leading to an .xyz file or the geometry given as a string
    filename: str = ''
    description: str = ''
    multiplicity: int = 1
    charge: int = 0

    @staticmethod
    def format_element_name(string):
        """
        OpenFermion uses case sensitive hash tables for chemical elements
        I.e. you need to name Lithium: 'Li' and 'li' or 'LI' will not work
        this conenience function does the naming
        :return: first letter converted to upper rest to lower
        """
        assert (len(string) > 0)
        assert (isinstance(string, str))
        fstring = string[0].upper() + string[1:].lower()
        return fstring

    @staticmethod
    def convert_to_list(geometry):
        """
        Convert a molecular structure given as a string into a list suitable for openfermion
        :param geometry: a string specifing a mol. structure. E.g. geometry="h 0.0 0.0 0.0\n h 0.0 0.0 1.0"
        :return: A list with the correct format for openferion E.g return [ ['h',[0.0,0.0,0.0], [..]]
        """
        result = []
        for line in geometry.split('\n'):
            words = line.split()
            if len(words) != 4:  break
            try:
                tmp = (ParametersQC.format_element_name(words[0]),
                       (np.float64(words[1]), np.float64(words[2]), np.float64(words[3])))
                result.append(tmp)
            except ValueError:
                print("get_geometry list unknown line:\n ", line, "\n proceed with caution!")
        return result

    def get_geometry(self):
        """
        Returns the geometry
        If a xyz filename was given the file is read out
        otherwise it is assumed that the geometry was given as string
        which is then reformated as a list usable as input for openfermion
        :return: geometry as list
        e.g. [(h,(0.0,0.0,0.35)),(h,(0.0,0.0,-0.35))]
        Units: Angstrom!
        """
        if self.geometry.split('.')[-1] == 'xyz':
            geomstring, comment = self.read_xyz_from_file(self.geometry)
            self.description = comment
            return self.convert_to_list(geomstring)
        elif self.geometry is not None:
            return self.convert_to_list(self.geometry)
        else:
            raise Exception("Parameters.qc.geometry is None")

    @staticmethod
    def read_xyz_from_file(filename):
        """
        Read XYZ filetype for molecular structures
        https://en.wikipedia.org/wiki/XYZ_file_format
        Units: Angstrom!
        :param filename:
        :return:
        """
        with open(filename, 'r') as file:
            content = file.readlines()
            natoms = int(content[0])
            comment = str(content[1])
            coord = ''
            for i in range(natoms):
                coord += content[2 + i]
            return coord, comment


@parametrized(parameter_class=ParametersHamiltonian)
class HamiltonianQC(HamiltonianBase):

    def __post_init__(self):
        """
        Constructor will run psi4 and create the molecule which is stored as member variable
        :param parameters: instance of ParametersQC which holds all necessary parameters
        """
        self.molecule = self.make_molecule(self.parameters)

    def n_electrons(self):
        """
        Convenience function
        :return: The total number of electrons
        """
        return self.molecule.n_electrons

    def n_orbitals(self):
        """
        Convenience function
        :return: The total number of (spatial) orbitals (occupied and virtual)
        """
        return self.molecule.n_orbitals

    def n_qubits(self):
        """
        Convenience function
        :return: Number of qubits needed
        """
        return 2*self.n_orbitals()

    def get_hamiltonian(self) -> openfermion.InteractionOperator:
        """
        Note that the Qubit Hamiltonian can be created over the call method which is already implemented in the baseclass
        :return: The fermionic Hamiltonian as InteractionOperator structure
        """
        # fast return if possible
        if self.molecule is None:
            self.molecule = self.make_molecule(self.parameters)
        return self.molecule.get_molecular_hamiltonian()

    def make_molecule(self) -> openfermion.MolecularData:
        """
        convenience function for internal calls
        """
        return self.make_molecule(self.parameters)

    @staticmethod
    def make_molecule(parameters: ParametersQC) -> openfermion.MolecularData:
        """
        Creates a molecule in openfermion format by running psi4 and extracting the data
        Will check for previous outputfiles before running
        :param parameters: An instance of ParametersQC, which also holds an instance of ParametersPsi4 via parameters.psi4
        The molecule will be saved in parameters.filename, if this file exists before the call the molecule will be imported from the file
        :return: the molecule in openfermion.MolecularData format
        """
        print("geom=",parameters.get_geometry())
        print("description=",parameters.description)
        print(parameters)
        molecule = MolecularData(geometry=parameters.get_geometry(),
                                 basis=parameters.basis_set,
                                 multiplicity=parameters.multiplicity,
                                 charge=parameters.charge,
                                 description=parameters.description.strip('\n'), # '\n' causes openfermion to chrash
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
            molecule = openfermionpsi4.run_psi4(molecule,
                                                run_scf=parameters.psi4.run_scf,
                                                run_mp2=parameters.psi4.run_mp2,
                                                run_cisd=parameters.psi4.run_cisd,
                                                run_ccsd=parameters.psi4.run_ccsd,
                                                run_fci=parameters.psi4.run_fci,
                                                verbose=parameters.psi4.verbose,
                                                tolerate_error=parameters.psi4.tolerate_error,
                                                delete_input=parameters.psi4.delete_input,
                                                delete_output=parameters.psi4.delete_output,
                                                memory=parameters.psi4.memory)

        molecule.save()
        print("file was ", molecule.filename)
        return molecule

    def verify(self) -> bool:
        from openvqe import OVQETypeError
        """
        Overwritten verify function to check specificly for ParametersQC type
        :return:
        """
        # do some verification specificaly for this class

        # check if the molecule was initialized
        if not isinstance(self.molecule, openfermion.MolecularData):
            raise OVQETypeError(attr=type(self).__name__ + ".molecule", type=type(self.molecule),
                                expected=type(openfermion.MolecularData))

        # do the standard checks for the baseclass
        return self._verify()


