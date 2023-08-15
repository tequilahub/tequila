from tequila.trotter_err.main_trot import EstTrotErr
import numpy as np
import tequila as tq
from tequila.hamiltonian import QubitHamiltonian, PauliString
from tequila.grouping.binary_rep import BinaryPauliString, BinaryHamiltonian
from tequila.grouping.fermionic_methods import do_svd
from tequila.grouping.fermionic_functions import obt_to_ferm, convert_tbts_to_frags
from tequila.grouping.fermionic_functions import n_elec
import openfermion
import pytest

HAS_PYSCF = "pyscf" in tq.quantumchemistry.INSTALLED_QCHEMISTRY_BACKENDS

def build_toymol():
    '''
    Build the qubit Hamiltonian with Jordan Wigner encoding for H2 at a particular geometry.
    '''

    trafo = "JordanWigner"
    mol = tq.chemistry.Molecule(
                            geometry="H 0.0 0.0 0.0 \n H 0.0 0.0 1.0",
                            basis_set="sto3g",
                            transformation=trafo,
                            backend='pyscf'
                                 )
    H = mol.make_hamiltonian()

    Hferm = mol.make_molecular_hamiltonian()

    return H,openfermion.get_fermion_operator(Hferm)

def get_fclf(H):
    '''
    Get the Hamiltonian fragments under the qubit FC-LF partition method, as explained in JCTC 16, 2400 (2020))
    '''

    Hbin = BinaryHamiltonian.init_from_qubit_hamiltonian(H)
    options={"method":"lf", "condition": "fc"}

    commuting_parts, _ = Hbin.commuting_groups(options=options)
    print("Number of FC-LF groups: {}".format(len(commuting_parts)))

    ListFrags=[]

    for i in range(len(commuting_parts)):
        tqqubFrag=commuting_parts[i].to_qubit_hamiltonian()
        ListFrags.append(tqqubFrag.to_openfermion())

    return ListFrags

def get_fcsi(H):
    '''
    Get the Hamiltonian fragments under the qubit FC-SI partition method, as explained in JCTC 16, 2400 (2020))
    '''

    Hbin = BinaryHamiltonian.init_from_qubit_hamiltonian(H)
    options={"method":"si", "condition": "fc"}

    commuting_parts, _ = Hbin.commuting_groups(options=options)
    print("Number of FC-SI groups: {}".format(len(commuting_parts)))

    ListFrags=[]

    for i in range(len(commuting_parts)):
        tqqubFrag=commuting_parts[i].to_qubit_hamiltonian()
        ListFrags.append(tqqubFrag.to_openfermion())

    return ListFrags

#set opf routines to get LR fragments...
def cartan_to_fermionic_operator(cobt, ctbts, orb_rot):
    '''
    Turn Hamiltonian fragments in their diagonal form into OpenFermion's FermionOperator.
    '''
    obt = np.einsum("pa, qb, ab", orb_rot[0], orb_rot[0], cobt)
    tbts = np.einsum("ipa, iqb, irc, isd, iabcd -> ipqrs",
                     orb_rot[1:], orb_rot[1:], orb_rot[1:], orb_rot[1:], ctbts)
    ferm_ops = [obt_to_ferm(obt,True)] + convert_tbts_to_frags(tbts, True)
    return ferm_ops

def get_LR(Hferm,name="h2"):
    '''
    Get Low-Rank Hamiltonian fragments, according to the technique outlined in npj Quantum Information 7, 83 (2021)
    '''

    orb_rots, cartan_obt, cartan_tbts, meas_alloc = do_svd(Hferm, n_elec(name))

    ferm_ops = cartan_to_fermionic_operator(cartan_obt, cartan_tbts, orb_rots)

    return ferm_ops

#####Testing functions.....
@pytest.mark.skipif(condition=not HAS_PYSCF, reason="you don't have pyscf")
def test_FCLF():
    Hq,Hferm=build_toymol()
    FCLFFrags=get_fclf(Hq)
    nqubs=openfermion.count_qubits(Hferm)
    alpha=EstTrotErr(FCLFFrags,nqubs)

    assert np.isclose(alpha, 0.42117695296, atol=1.e-4)

@pytest.mark.skipif(condition=not HAS_PYSCF, reason="you don't have pyscf")
def test_FCSI():
    Hq,Hferm=build_toymol()
    FCSIFrags=get_fcsi(Hq)
    nqubs=openfermion.count_qubits(Hferm)
    alpha=EstTrotErr(FCSIFrags,nqubs)

    assert np.isclose(alpha, 0.42117695296, atol=1.e-4)

@pytest.mark.skipif(condition=not HAS_PYSCF, reason="you don't have pyscf")
def test_LR():
    Hq,Hferm=build_toymol()
    LRFrags=get_LR(Hferm)
    nqubs=openfermion.count_qubits(Hferm)
    alpha=EstTrotErr(LRFrags,nqubs)

    assert np.isclose(alpha, 0.42178077369, atol=1.e-4)
