"""
Base Class for OpenVQE Hamiltonians
"""

from openvqe.parameters import ParametersHamiltonian


class Hamiltonian:

    parameters: ParametersHamiltonian = ParametersHamiltonian()


    def greet(self):
        print("This is the "+type(self).__name__+" class")

    def get_Hamiltonian(self):
        raise Exception("get_Hamiltonian of BaseClass should not be called")

