"""
Parameters for OpenVQE
"""

from dataclasses import dataclass, field
from enum import Enum
import json
from dataclasses_json import dataclass_json


@dataclass
@dataclass_json
@dataclass
class Parameters:
    @dataclass_json
    @dataclass
    class Optimizer:
        class supported(Enum):
            LBFGS = 'LBFGS'
            COBYLA = 'COBYLA'

        type: supported = supported.LBFGS
        maxiter: int = 2000

        def sanity(self):
            assert(self.type in self.supported)

    @dataclass_json
    @dataclass
    class Hamiltonian:
        class supported(Enum):
            QC = 'QC'
            CUSTOM = 'CUSTOM'

        type: supported = supported.QC
        name: str = 'test'

    @dataclass_json
    @dataclass
    class Preparation:
        class supported_ansatz(Enum):
            UCC = 'UCC'
            CUSTOM = 'CUSTOM'

        ansatz: supported_ansatz = supported_ansatz.UCC

        class supported_decompositions(Enum):
            FULL = 'FULL'
            UCC = 'TROTTER'
            QDRIFT = 'QDRIFT'

        decomposition: supported_decompositions = supported_decompositions.FULL

    @dataclass_json
    @dataclass
    class QC:
        basis_set: str = None
        geometry: str = None

        @staticmethod
        def read_xyz_from_file(self, filename):
            file = open(filename, 'r')
            content = file.readlines()
            natoms = int(content[0])
            self.comment = content[1]
            coord = ''
            for i in range(natoms):
                coord += content[2 + i]
            self.geometry = coord

    optimizer: Optimizer = field(default_factory=Optimizer)
    hamiltonian: Hamiltonian = field(default_factory=Hamiltonian)
    preparation: Preparation = field(default_factory=Preparation)
    qc: QC = field(default_factory=QC)
    name: str = "Comment about this run"

    def __post_init__(self):
        if self.hamiltonian.type is not self.hamiltonian.supported.QC or self.hamiltonian.type is None: self.qc = None

    def print_to_file(self, filename, name=None, write_mode='a+'):
        """
        Converts the Parameters class to JSON and prints it to a file
        :param filename:
        :param name: the name of this parameter instance (converted to lowercase)
        if given it is printed before the content
        :param write_mode: specify if existing files shall be overwritten or appended (default)
        """
        content = self.to_json()
        with open(filename, "a+") as file:
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
        :param name: The name of the class object which was stored
        If given, the file is first parsed for the name and the rest of the line is taken as content
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
