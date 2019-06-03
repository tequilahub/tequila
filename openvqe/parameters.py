from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
import numpy as np

"""
Parameterclasses for OpenVQE modules
All clases are derived from ParametersBase
"""


@dataclass
class ParametersBase:

    @staticmethod
    def to_bool(var):
        if type(var) == int:
            return var == 1
        elif type(var) == bool:
            return var
        elif type(var) == str:
            return var.lower() == 'true'
        else:
            raise Exception("ParameterBase.to_bool(var): not implemented for type(var)==", type(var))

    @classmethod
    def name(cls):
        return cls.__name__

    def print_to_file(self, filename, write_mode='a+'):
        """
        Converts the Parameters class to JSON and prints it to a file
        :param filename:
        :param name: the comment of this parameter instance (converted to lowercase)
        if given it is printed before the content
        :param write_mode: specify if existing files shall be overwritten or appended (default)
        """

        string = '\n'
        string += self.name() + " = {\n"
        for key in self.__dict__:
            if isinstance(self.__dict__[key], ParametersBase):
                self.__dict__[key].print_to_file(filename=filename)
                string += str(key) + " : " + str(True) + "\n"
            else:
                string += str(key) + " : " + str(self.__dict__[key]) + "\n"
        string += "}\n"
        with open(filename, write_mode) as file:
            file.write(string)

        return self

    @classmethod
    def read_from_file(cls, filename):
        """
        Reads the parameters from a file
        See the print_to_file function to create files which can be read by this function
        The input syntax is the same as creating a dictionary in python
        where the name of the dictionary is the derived class
        The function creates a new instance and returns this
        :param filename: the name of the file
        :return: A new instance of the read in parameter class
        """

        new_instance = cls()

        keyvals = {}
        with open(filename, 'r') as file:
            found = False
            for line in file:
                if found:
                    if line.split()[0].strip() == "}":
                        break
                    keyval = line.split(":")
                    keyvals[keyval[0].strip()] = keyval[1].strip()
                elif cls.name() in line.split():
                    found = True
        for key in keyvals:
            if not key in new_instance.__dict__:
                raise Exception("Unknown key for class=" + cls.name() + " and  key=", key)
            elif keyvals[key] == 'None':
                new_instance.__dict__[key]=None
            elif isinstance(new_instance.__dict__[key], ParametersBase) and cls.to_bool(keyvals[key]):
                new_instance.__dict__[key] = new_instance.__dict__[key].read_from_file(filename)
            else:
                if isinstance(cls.__dict__[key], type(None)):
                    raise Exception("Default values of classes derived from ParameterBase should NOT be set to None. Use __post_init()__ for that")
                elif isinstance(cls.__dict__[key], bool):
                    new_instance.__dict__[key] = cls.to_bool(keyvals[key])
                else:
                    new_instance.__dict__[key] = type(cls.__dict__[key])(keyvals[key])

        return new_instance

@dataclass
class ParametersPsi4(ParametersBase):
    run_scf: bool = True
    run_mp2: bool = False
    run_cisd: bool = False
    run_ccsd: bool = False
    run_fci: bool = False
    verbose: bool = False
    tolerate_error: bool = False
    delete_input: bool = True
    delete_output: bool = False
    memory: int = 8000
    template_file: str = ''

    def __post_init__(self):
        if self.template_file=='': self.template_file=None

@dataclass
class ParametersQC(ParametersBase):
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
        assert(len(string)>0)
        assert(isinstance(string,str))
        fstring = string[0].upper()+string[1:].lower()
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
                tmp = (ParametersQC.format_element_name(words[0]), (np.float64(words[1]), np.float64(words[2]), np.float64(words[3])))
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

@dataclass
class ParametersHamiltonian:
    type: str = 'QC'

if __name__ == "__main__":
    pqc = ParametersQC()
    pqc.psi4 = ParametersPsi4()  # .uninitialized()
    # pqc.psi4.run_ccsd=True
    pqc.print_to_file(filename='test')

    print(type(pqc))
    # psi4 = ParametersQC.psi4
    # print(isinstance(pqc, ParametersBase))

    pqc2 = ParametersQC().read_from_file(filename='test')
    print("worked=", pqc == pqc2)

    print(type(pqc2.__dict__['psi4']))

    # test = pqc.__dict__['psi4']()

    print(pqc2)
    print(type(pqc2))
    print(type(type(pqc2).__class__))
