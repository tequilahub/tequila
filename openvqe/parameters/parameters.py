"""
Parameters for OpenVQE
"""

from dataclasses import dataclass
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

    @dataclass_json
    @dataclass
    class Hamiltonian:
        class supported(Enum):
            QC = 'QC'
            CUSTOM = 'CUSTOM'

        @dataclass_json
        @dataclass
        class QC:
            basis_set: str = None
            geometry: str = None

        type: supported = supported.QC
        qc_data: QC = QC()
        name: str = 'test'

        def __post_init__(self):
            if self.type is not self.supported.QC or self.type is None: self.QC = None

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

    optimizer: Optimizer = Optimizer()
    hamiltonian: Hamiltonian = Hamiltonian()
    preparation: Preparation = Preparation()
    name: str = "Comment about this run"

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
