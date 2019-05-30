"""
Parameters for OpenVQE
"""

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
import numpy as np

def return_none(): return None

@dataclass
@dataclass_json
@dataclass
class Parameters:
    @dataclass_json
    @dataclass
    class Optimizer:
        type: str = 'LBFGS'
        maxiter: int = 2000

    @dataclass_json
    @dataclass
    class Hamiltonian:
        type: str = 'QC'

    @dataclass_json
    @dataclass
    class Preparation:
        ansatz: str = "UCC"
        decomposition: str = "TROTTER1"

    @dataclass_json
    @dataclass
    class QC:
        basis_set: str = ''
        geometry: str = ''

        @staticmethod
        def convert_to_list(geometry):
            result = []
            for line in geometry.split('\n'):
                words = line.split()
                if len(words) != 4:  break
                try:
                    tmp = [words[0].upper(), [np.float64(words[1]), np.float64(words[2]), np.float64(words[3])]]
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
            e.g. [[h,[0.0,0.0,0.35]],[h,[0.0,0.0,-0.35]]]
            """
            if(self.geometry.split('.')[-1]=='xyz'): return self.convert_to_list(self.read_xyz_from_file(self.geometry))
            elif self.geometry is not None: return self.convert_to_list(self.geometry)
            else: raise Exception("Parameters.qc.geometry is None")

        @staticmethod
        def read_xyz_from_file(filename):
            """
            Read XYZ filetype for molecular structures
            https://en.wikipedia.org/wiki/XYZ_file_format
            :param filename:
            :return:
            """
            with open(filename, 'r') as file:
                content = file.readlines()
                natoms = int(content[0])
                coord = ''
                for i in range(natoms):
                    coord += content[2 + i]
                return coord

    optimizer: Optimizer = field(default=Optimizer())
    hamiltonian: Hamiltonian = field(default=Hamiltonian())
    preparation: Preparation = field(default=Preparation())
    qc: QC = field(default=QC())
    comment: str = "Comment about this run"


    def print_to_file(self, filename, name=None, write_mode='a+'):
        """
        Converts the Parameters class to JSON and prints it to a file
        :param filename:
        :param name: the comment of this parameter instance (converted to lowercase)
        if given it is printed before the content
        :param write_mode: specify if existing files shall be overwritten or appended (default)
        """
        content = self.to_json()
        with open(filename, write_mode) as file:
            if name is not None: file.write("\n" + name.lower() + " ")
            file.write(content)
            file.write("\n")
        return self  # easier chaining

    @staticmethod
    def read_from_file(filename, name=None):
        """
        Reads the parameters from a file in JSON format
        The content of the dictionary should be given in one line
        See the print_to_file function to create files which can be read by this function
        :param filename:
        :param name: The comment of the class object which was stored
        If given, the file is first parsed for the comment and the rest of the line is taken as content
        Otherwise it is assumed that the file only contains the information to this dataclass
        :return: the read in parameter class (self, for chaining)
        """
        found = name is None
        if not found: name = name.lower()

        json_content = ""
        with open(filename, 'r') as file:
            for line in file:
                if found or name in line.split():
                    json_content = line.lstrip(name)

        return Parameters.from_json(json_content)

    def __repr__(self):
        return self.to_json()
