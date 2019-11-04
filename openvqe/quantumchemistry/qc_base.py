from dataclasses import dataclass
from openvqe import OpenVQEParameters, typing
from openvqe.hamiltonian import HamiltonianQC

from openfermion import MolecularData


@dataclass
class ParametersQC(OpenVQEParameters):
    """
    Specialization of ParametersHamiltonian
    Parameters for the HamiltonianQC class
    """
    basis_set: str = ''  # Quantum chemistry basis set
    geometry: str = ''  # geometry of the underlying molecule (units: Angstrom!), this can be a filename leading to an .xyz file or the geometry given as a string
    description: str = ''
    multiplicity: int = 1
    charge: int = 0
    closed_shell: bool = True
    filename: str = "molecule"

    @property
    def molecular_data_param(self) -> dict:
        """
        :return: Give back all parameters for the MolecularData format from openfermion as dictionary
        """
        return {'basis': self.basis_set, 'geometry': self.get_geometry(), 'description': self.description,
                'charge': self.charge, 'multiplicity': self.multiplicity, 'filename': self.filename
                }

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
                       (float(words[1]), float(words[2]), float(words[3])))
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
            comment = str(content[1]).strip('\n')
            coord = ''
            for i in range(natoms):
                coord += content[2 + i]
            return coord, comment


class QuantumChemistryBase:

    def __init__(self, parameters: ParametersQC):
        self.parameters = parameters
        self.molecule = self.make_molecule()

    @property
    def n_orbitals(self) -> int:
        return self.molecule.n_orbitals

    @property
    def n_electrons(self) -> int:
        return self.molecule.n_electrons

    @property
    def n_alpha_electrons(self) -> int:
        return self.molecule.get_n_alpha_electrons()

    @property
    def n_beta_electrons(self) -> int:
        return self.molecule.get_n_beta_electrons()

    def get_hamiltonian(self, transformation: typing.Union[str, typing.Callable] = None) -> HamiltonianQC:
        return HamiltonianQC(molecule=self.molecule, transformation=transformation)

    def make_molecule(self):
        raise Exception("BaseClass Method")

    def compute_mp2_amplitudes(self):
        raise Exception("BaseClass Method")

    def compute_ccsd_amplitudes(self):
        raise Exception("BaseClass Method")
