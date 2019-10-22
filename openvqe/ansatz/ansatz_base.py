from openvqe.openvqe_exceptions import OpenVQEException

class AnsatzBase:
    """
    Base Class for the VQE Ansatz
    Derive all specializations from this Base Class
    """

    def __init__(self):
        pass

    def __call__(self, angles: list):
        """
        :param angles: The angles which parametrize the circuit
        :return: the circuit in the correct format for the simulator backend secified by self.parameters.backend
        """
        return self.construct_circuit(angles=angles)

    def construct_circuit(self, angles: list):
        """
        Construct the circuit specifified by the given ansatz
        this function should be overwritten by specializations of this baseclass
        :param angles:
        :return: the circuit in the correct format for the simulator backend secified by self.parameters.backend
        """
        raise OpenVQEException(type(self).__name__ + ": You tried to call the ABC directly")

    def greet(self):
        print("Hello from the " + type(self).__name__ + " class")
