"""
Example File on how to define a customized Hamiltonian
"""

from openvqe.hamiltonian import HamiltonianBase, QubitHamiltonian, PZ, ParametersQC, HamiltonianPsi4
import openfermion
import numpy as np

class MyFermionicHamiltonian(HamiltonianPsi4):

    def get_fermionic_hamiltonian(self):
        """
        Overwrite this in order to not get a Hamiltonian from a molecule
        :return:
        """
        constant=1.0
        obt = np.zeros([2,2])
        obt[0,0]=1.0
        tbd = np.zeros([2,2,2,2])
        H = openfermion.get_fermion_operator(openfermion.InteractionOperator(constant=constant, one_body_tensor=obt, two_body_tensor=tbd))
        return H

    def my_trafo(self, fermionic: openfermion.FermionOperator):
        """
        Demonstration on how to overwrite transformations
        ... don't wanna invent something new here, so this will just do good old jordan wigner
        """
        return openfermion.jordan_wigner(fermionic)

if __name__=="__main__":

    print("\nDefine a customized Qubit Hamiltonian:")
    print("The class created defines the useless Hamiltonian: H = 1.0*Z(1) + 1.0j*Z(2) ")
    myh = PZ(qubit=1)+1j*PZ(qubit=2)
    print(myh())

    print("\nDefine a customized Fermionic Hamiltonian by creating a class derived from HamiltonianBase and overwriting the get_hamiltonian function:")
    print("The class created defines the useless single mode Hamiltonian: H = 1.0 + 1.0*a^\dagger a which transforms under JW to 1.5 - 0.5 Z(0)")
    param=ParametersQC(transformation="JW") # can also use 'BK' or define a own transformation similar to
    myh = MyFermionicHamiltonian(param)
    print(myh())

    print("\nSame with my_trafo instead of jordan-wigner .... should give the same result")
    param=ParametersQC(transformation="my_trAfo") # A is uppercase on purpose to test
    myh = MyFermionicHamiltonian(param)
    print(myh())

