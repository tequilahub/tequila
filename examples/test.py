from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
import numpy as np


@dataclass
class ParametersBase:

    def name(self):
        return type(self).__name__

    def print_to_file(self, filename, write_mode='a+'):
        """
        Converts the Parameters class to JSON and prints it to a file
        :param filename:
        :param name: the comment of this parameter instance (converted to lowercase)
        if given it is printed before the content
        :param write_mode: specify if existing files shall be overwritten or appended (default)
        """
        string = '\n'
        string += self.name() + "= {\n"
        for key in self.__dict__:
            if isinstance(self.__dict__[key], ParametersBase):
                self.__dict__[key].print_to_file(filename=filename)
                string += str(key) + " : " + str(True) + "\n"
            else: string += str(key) + " : " + str(self.__dict__[key]) + "\n"
        string+="}\n"
        with open(filename, write_mode) as file:
            file.write(string)

        return self


    def read_from_file(self, filename):
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
        keyvals = {}
        with open(filename, 'r') as file:
            found=False
            for line in file:
                if found:
                    if line.split()[0].strip() == "}":
                        break
                    keyval = line.split(":")
                    keyvals[keyval[0]]=keyval[1]
                elif self.name() in line.split():
                    found=True

        for key in keyvals:
            if not key in self.__dict__:
                raise Exception("Unknown key for class="+self.name()+" and  key=", key)
            if isinstance(self.__dict__[key], ParametersBase and keyvals[key]):
                self.__dict__[key] = self.__dict__[key].__get__(None,A).read_from_file()

            self.__dict__[key] = keyvals[key]

        return self






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
    template_file: str = None


@dataclass
class ParametersQC(ParametersBase):
    psi4: ParametersPsi4 = ParametersPsi4()
    basis_set: str = ''  # Quantum chemistry basis set
    geometry: str = ''  # geometry of the underlying molecule, this can be a filename leading to an .xyz file or the geometry given as a string
    filename: str = ""
    description: str = ''
    multiplicity: int = 1
    charge: int = 0

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
                tmp = (words[0].upper(), (np.float64(words[1]), np.float64(words[2]), np.float64(words[3])))
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
        """
        if (self.geometry.split('.')[-1] == 'xyz'):
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


if __name__ == "__main__":

    pqc = ParametersQC()
    pqc.psi4 = ParametersPsi4()#.uninitialized()
    #pqc.psi4.run_ccsd=True
    pqc.print_to_file(filename='test')

    print(type(pqc))
    #psi4 = ParametersQC.psi4
    #print(isinstance(pqc, ParametersBase))

    pqc2 = ParametersQC().read_from_file(filename='test')
    print("worked=",pqc==pqc2)

    print(type(pqc2.__dict__['psi4']))

    #test = pqc.__dict__['psi4']()

    print(pqc2)
    print(type(pqc2))
    print(type(type(pqc2).__class__))
