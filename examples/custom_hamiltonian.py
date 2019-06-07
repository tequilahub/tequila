"""
Example File on how to define a customized Hamiltonian
"""

from openvqe import ParametersHamiltonian, HamiltonianBase
import openfermion
import numpy as np

class MyQubitHamiltonian(HamiltonianBase):

    def my_trafo(self, H):
        """
        Here we define the hamiltonian to be already in qubit form, so no transformation will be needed
        """
        return H

    def get_hamiltonian(self):

        H = openfermion.QubitOperator()
        H.terms[((0, 'Z'),)] = 1.0
        H.terms[((1, 'Z'),)] = 1.0j
        print("Terms=", H.terms)

        return H

class MyFermionicHamiltonian(HamiltonianBase):

    def get_hamiltonian(self):
        constant=1.0
        obt = np.zeros([2,2])
        obt[0,0]=1.0
        tbd = np.zeros([2,2,2,2])
        H = openfermion.InteractionOperator(constant=constant, one_body_tensor=obt, two_body_tensor=tbd)
        return H



if __name__=="__main__":

    print("\nDefine a customized Qubit Hamiltonian by creating a class derived from HamiltonianBase and overwriting the get_hamiltonian function:")
    print("The class created defines the useless Hamiltonian: H = 1.0*Z(1) + 1.0j*Z(2) ")
    param=ParametersHamiltonian(transformation="my_trafo")
    myh = MyQubitHamiltonian(param)
    print(myh())

    print("\nDefine a customized Fermionic Hamiltonian by creating a class derived from HamiltonianBase and overwriting the get_hamiltonian function:")
    print("The class created defines the useless single mode Hamiltonian: H = 1.0 + 1.0*a^\dagger a which transforms under JW to 1.5 - 0.5 Z(0)")
    param=ParametersHamiltonian(transformation="JW") # can also use 'BK' or define a own transformation similar to
    myh = MyFermionicHamiltonian(param)
    print(myh())

