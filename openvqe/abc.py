from openvqe.parameters import ParametersBase, OutputLevel
from openvqe import OVQETypeError, OVQEException
from dataclasses import dataclass, field
from enum import Enum

"""
Parameterclasses for OpenVQE modules
All clases are derived from ParametersBase
I/O functions are defined in ParametersBase and are inherited to all derived classes
It is currently not possible to set the default of parameters to None (will confuse I/O routines)
"""

class OutputLevel(Enum):
    SILENT = 0
    STANDARD = 1
    DEBUG = 2
    ALL = 3


@dataclass
class OpenVQEParameters:
    # Parameters which every module of OpenVQE needs

    outfile: str = ""
    # outputlevel is stored as int to not confuse the i/o functions
    _ol: int = field(default=OutputLevel.STANDARD.value)

    def output_level(self) -> OutputLevel:
        """
        Enum handler
        :return: output_level as enum for more convenience
        """
        return OutputLevel(self._ol)

    @staticmethod
    def to_bool(var):
        """
        Convert different types to bool
        currently supported: int, str
        int: 1 -> True, everything else to False
        str: gets converted to all_lowercase then 'true' -> True, everything else to False
        :param var: an instance of the currently supported types
        :return: converted bool
        """
        if type(var) == int:
            return var == 1
        elif type(var) == bool:
            return var
        elif type(var) == str:
            return var.lower() == 'true' or var.lower() == '1'
        else:
            raise Exception("ParameterBase.to_bool(var): not implemented for type(var)==", type(var))

    @classmethod
    def name(cls):
        return cls.__name__

    def print_to_file(self, filename, write_mode='a+'):
        """
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
                new_instance.__dict__[key] = None
            elif isinstance(new_instance.__dict__[key], ParametersBase) and cls.to_bool(keyvals[key]):
                new_instance.__dict__[key] = new_instance.__dict__[key].read_from_file(filename)
            else:

                if key not in cls.__dict__:
                    # try to look up base class
                    assert (key in new_instance.__dict__)
                    if isinstance(new_instance.__dict__[key], type(None)):
                        raise Exception(
                            "Currently unresolved issue: If a ParameterClass has a subclass the parameters can never be set to none"
                        )
                    elif isinstance(new_instance.__dict__[key], bool):
                        new_instance.__dict__[key] = new_instance.to_bool(keyvals[key])
                    else:
                        new_instance.__dict__[key] = type(new_instance.__dict__[key])(keyvals[key])
                elif isinstance(cls.__dict__[key], type(None)):
                    raise Exception(
                        "Default values of classes derived from ParameterBase should NOT be set to None. Use __post_init()__ for that")
                elif isinstance(cls.__dict__[key], bool):
                    new_instance.__dict__[key] = cls.to_bool(keyvals[key])
                else:
                    new_instance.__dict__[key] = type(cls.__dict__[key])(keyvals[key])

        return new_instance

def parametrized(parameter_class=OpenVQEParameters, _cls=None):
    """
    Use this as decorator if you want your class to be parametrized by OpenVQEParameter types
    The decorator will implement the correct init and verify functions which you can overwrite afterwards
    :param : Type of the ParameterClass (should be derived from OpenVQEParameters)
    :return: Decorated class
    """

    def decorator(cls):
        def init(self, parameters: parameter_class, *args, **kwargs):
            """
            Default constructor of the baseclass, initializes the parameters
            :param parameters: Parameters of type or derived from OpenVQEParameters
            """
            if parameters is None:
                self.parameters = parameter_class()
            else:
                self.parameters = parameters
            self._verify()
            if hasattr(self, "__post_init__"):
                self.__post_init__(*args, **kwargs)

        def _verify(self) -> bool:
            """
            Verify that the class was initialized correctly
            :return: true if sane, raises exception if not
            """
            # check if the parameters are of the correct type
            if not isinstance(self.parameters, parameter_class):
                raise OVQETypeError(attr=type(self).__name__ + ".parameters", type=type(self.parameters),
                                    expected=parameter_class)

            return True

        setattr(cls, "__init__", init)
        setattr(cls, "_verify", _verify)

        return cls

    if _cls is None:
        return decorator
    else:
        return decorator(cls=_cls)

@parametrized(parameter_class=ParametersBase)
class OpenVQEModule:
    """
    Abstract Base Class for all OpenVQE modules
    """

    def print(self, *args, level=OutputLevel.STANDARD, sep=' ', end='\n', file=None):
        """
        Controlled ouput to file or terminal
        """
        if self.parameters.output_level().value >= level.value:
            if file is not None and self.parameters.outputfile != "":
                file = self.parameters.outputfile
            print(*args, sep, end, file)

    def __post_init__(self):
        """
        post init will be called from the __init__ function defined in the decorator
        :return: Nothing for the ABC
        """
        pass

    def greet(self):
        self.print("Hello from the " + type(self).__name__ + " class")

    def verify(self) -> bool:
        """
        check if the instance is sane, should be overwritten by derived classes
        :return: true if sane, raises exception if not
        """
        return self._verify(paramtype=ParametersBase)

    @dataclass
    class ParametersBase:

        # Parameters which every module of OpenVQE needs

        outfile: str = ""
        # outputlevel is stored as int to not confuse the i/o functions
        _ol: int = field(default=OutputLevel.STANDARD.value)

        def output_level(self) -> OutputLevel:
            """
            Enum handler
            :return: output_level as enum for more convenience
            """
            return OutputLevel(self._ol)

        @staticmethod
        def to_bool(var):
            """
            Convert different types to bool
            currently supported: int, str
            int: 1 -> True, everything else to False
            str: gets converted to all_lowercase then 'true' -> True, everything else to False
            :param var: an instance of the currently supported types
            :return: converted bool
            """
            if type(var) == int:
                return var == 1
            elif type(var) == bool:
                return var
            elif type(var) == str:
                return var.lower() == 'true' or var.lower() == '1'
            else:
                raise Exception("ParameterBase.to_bool(var): not implemented for type(var)==", type(var))

        @classmethod
        def name(cls):
            return cls.__name__

        def print_to_file(self, filename, write_mode='a+'):
            """
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
                    new_instance.__dict__[key] = None
                elif isinstance(new_instance.__dict__[key], ParametersBase) and cls.to_bool(keyvals[key]):
                    new_instance.__dict__[key] = new_instance.__dict__[key].read_from_file(filename)
                else:

                    if key not in cls.__dict__:
                        # try to look up base class
                        assert (key in new_instance.__dict__)
                        if isinstance(new_instance.__dict__[key], type(None)):
                            raise Exception(
                                "Currently unresolved issue: If a ParameterClass has a subclass the parameters can never be set to none"
                            )
                        elif isinstance(new_instance.__dict__[key], bool):
                            new_instance.__dict__[key] = new_instance.to_bool(keyvals[key])
                        else:
                            new_instance.__dict__[key] = type(new_instance.__dict__[key])(keyvals[key])
                    elif isinstance(cls.__dict__[key], type(None)):
                        raise Exception(
                            "Default values of classes derived from ParameterBase should NOT be set to None. Use __post_init()__ for that")
                    elif isinstance(cls.__dict__[key], bool):
                        new_instance.__dict__[key] = cls.to_bool(keyvals[key])
                    else:
                        new_instance.__dict__[key] = type(cls.__dict__[key])(keyvals[key])

            return new_instance
