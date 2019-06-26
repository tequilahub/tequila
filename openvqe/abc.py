from openvqe.parameters import ParametersBase, OutputLevel
from openvqe import OvqeTypeError

class OvqeModule:
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

    def __init__(self, parameters: ParametersBase):
        """
        Default constructor of the baseclass, initializes the parameters
        :param parameters: Parameters of type or derived from ParametersHamiltonian
        """

        # assure that whatever is set as parameters was originally derived from ParameterBase
        assert (isinstance(parameters, ParametersBase))
        # create parameters for the current instance
        self.parameters = parameters
        # call the verify function
        self.verify()

    def greet(self):
        self.print("Hello from the " + type(self).__name__ + " class")

    def verify(self) -> bool:
        """
        check if the instance is sane, should be overwritten by derived classes
        :return: true if sane, raises exception if not
        """
        return self._verify(ParameterType=ParametersBase)

    def _verify(self, ParameterType: type) -> bool:
        """
        Actual verify function
        :return: true if sane, raises exception if not
        """

        # check if verify was called correctly
        if not isinstance(ParameterType, type):
            raise OvqeException(
                "Wrong input type for " + type(self).__name__ + "._verify"
            )
        # check if the parameters are of the correct type
        if not isinstance(self.parameters, ParameterType):
            # raise OpenVQEException(
            #     "parameters attribute of instance of class " + type(
            #         self).__name__ + " should be of type " + ParameterType.__name__ + " but is of type " + type(
            #         self.parameters).__name__)
            raise OvqeTypeError(attr=type(self).__name__ + ".parameters", type=type(self.parameters),
                                expected=ParameterType)

        return True
