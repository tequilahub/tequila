import operator
import tequila as tq
import numpy as np
from tequila.circuit.gates import PauliGate
from tequila.objective.braket import make_overlap, make_transition
from tequila.tools.random_generators import make_random_circuit, make_random_hamiltonian

def test_simple_overlap():
    '''
    Function that tests if make_overlap function is working correctly.
    It creates a simple circuit in order to check that both real and imaginary 
    part are calculated in the right way. 

    Returns
    -------
    None.

    '''
    # two circuits to test
    U0 = tq.gates.Rz(angle=1.0, target=1)#tq.gates.H(target=1) + tq.gates.CNOT (1 ,2)
    U1 = tq.gates.Rz(angle=2, target=1)#tq.gates.X(target=[1,2])#
    
    objective_real, objective_im = make_overlap(U0,U1)
    
    Ex = tq.simulate(objective_real)
    Ey = tq.simulate(objective_im)
    
    exp_val = Ex + 1.0j*Ey
    
    #print('Evaluated overlap between the two states: {}\n'.format(exp_val))
    
    # we want the overlap of the wavefunctions
    # # to test we can compute it manually
    wfn0 = tq.simulate(U0)
    wfn1 = tq.simulate(U1)
   
    test = wfn0.inner(wfn1)
    #print('Correct overlap between the two states: {}'.format(test))
    
    #print('The two result are approximately the same?',np.isclose(test, exp_val, atol=1.e-4))
    
    assert np.isclose(test, exp_val, atol=1.e-4)
    
    return

def test_random_overlap():
    '''
    Function that tests if make_overlap function is working correctly.
    It creates circuits with random number of qubits, random rotations and 
    random angles.

    Returns
    -------
    None.

    '''

    # make random circuits
    
    #np.random.seed(111)
    n_qubits = np.random.randint(1, high=5)
    U = {k:tq.make_random_circuit(n_qubits) for k in range(2)}

    objective_real, objective_im = make_overlap(U[0],U[1])
    
    Ex = tq.simulate(objective_real)
    Ey = tq.simulate(objective_im)
    
    exp_val = Ex + 1.0j*Ey
    
    # we want the overlap of the wavefunctions
    # # to test we can compute it manually
    wfn0 = tq.simulate(U[0])
    wfn1 = tq.simulate(U[1])
    
    test = wfn0.inner(wfn1)
    
    #print(test, '\n', exp_val)
    
    assert np.isclose(test, exp_val, atol=1.e-4)
    
    return 

def test_simple_transition():
    '''
    Function that tests if make_transition function is working correctly.
    It creates a simple circuit in order to check that both real and imaginary 
    part of the transition elementare calculated in the right way for a given 
    Hamiltonian. 

    Returns
    -------
    None.

    '''
    # two circuits to test
    U0 = tq.gates.H(target=1) + tq.gates.CNOT (1 ,2)#tq.gates.Rx(angle=1.0, target=1)#
    U1 = tq.gates.X(target=[1,2]) + tq.gates.Ry(angle=2, target=1)#
    
    # defining the hamiltonian
    H = tq.QubitHamiltonian("1.0*Y(0)X(1)+0.5*Y(1)Z(0)")
    #print('Hamiltonian',H,'\n')
    
    # calculating the transition element
    trans_real, trans_im = make_transition(U0=U0, U1=U1, H=H)
    
    tmp_real = tq.simulate(trans_real)
    tmp_im = tq.simulate(trans_im)
    
    trans_el = tmp_real + 1.0j*tmp_im
    
    #print('Evaluated transition element between the two states: {}'.format( trans_el))
    
    # # to test we can compute it manually
    #print()
    correct_trans_el = 0.0 + 0.0j
    
    wfn0 = tq.simulate(U0)
    
    for ps in H.paulistrings:
        
        c_k = ps.coeff
       
        U_k = PauliGate(ps)
        wfn1 = tq.simulate(U1+U_k)
        
        tmp = wfn0.inner(wfn1)
        #print('contribution',c_k*tmp)
        correct_trans_el += c_k*tmp
    
    
    #print('Correct transition element value: {}'.format(correct_trans_el))
    
    #print('The two result are approximately the same?',np.isclose(correct_trans_el, trans_el, atol=1.e-4))
    
    assert np.isclose(correct_trans_el, trans_el, atol=1.e-4)
    
    return

def test_random_transition():
    '''
    Function that tests if make_transition function is working correctly.
    It creates circuits with random number of qubits, random rotations and 
    random angles and a random Hamiltonian with random number of (random) 
    Pauli strings.

    Returns
    -------
    None.

    '''
    
    #np.random.seed(111)
    n_qubits = np.random.randint(1, high=5)
    #print(n_qubits)
    
    U = {k:tq.make_random_circuit(n_qubits) for k in range(2)}
    
    #print(U[0])
    #print(U[1])
    
    #make random hamiltonian
    paulis = ['X','Y','Z']
    n_ps = np.random.randint(1, high=2*n_qubits+1)
    
    H = make_random_hamiltonian(n_qubits, paulis=paulis, n_ps=n_ps)
    
    trans_real, trans_im = make_transition(U0=U[0], U1=U[1], H=H)
    
    tmp_real = tq.simulate(trans_real)
    tmp_im = tq.simulate(trans_im)
    
    trans_el = tmp_real + 1.0j*tmp_im
    
    correct_trans_el = 0.0 + 0.0j
    
    
    wfn0 = tq.simulate(U[0])
    #print()
    #print(wfn0)
    
    for ps in H.paulistrings:
        
        c_k = ps.coeff
       
        U_k = PauliGate(ps)
        wfn1 = tq.simulate(U[1]+U_k)
        
        tmp = wfn0.inner(wfn1)
        correct_trans_el += c_k*tmp
    
    wfn1 = tq.simulate(U[1])
    # print(wfn1)
    
    correct_trans_el_2nd = wfn0.inner(H(wfn1))
    #print()
    #print(correct_trans_el, '\n',trans_el, '\n', correct_trans_el_2nd)
    
    
    assert np.isclose(correct_trans_el, trans_el, atol=1.e-4)
    
    assert np.isclose(correct_trans_el, correct_trans_el_2nd, atol=1.e-4)
    
    return


def test_braket():
    """_summary_
    """
    # make random circuits
    
    #np.random.seed(111)
    n_qubits = np.random.randint(1, high=5)
    
    U = {k:tq.make_random_circuit(n_qubits) for k in range(2)}
    
    ######## Testing self overlap #########
    self_overlap_real, self_overlap_im = tq.braket(ket=U[0])

    assert np.isclose(self_overlap_real(), 1, atol=1.e-4)
    assert np.isclose(self_overlap_im(), 0, atol=1.e-4)

    ######## Testing expectation value #########
    # make random hamiltonian
    paulis = ['X','Y','Z']
    n_ps = np.random.randint(1, high=2*n_qubits+1)
    
    H = make_random_hamiltonian(n_qubits, paulis=paulis, n_ps=n_ps)

    exp_value_tmp = tq.ExpectationValue(H=H, U=U[0])
    br_exp_value_real, br_exp_value_im = tq.braket(ket=U[0], operator=H)
    br_exp_value_tmp = br_exp_value_real + 1.0j*br_exp_value_im
    
    exp_value= tq.simulate(exp_value_tmp)
    br_exp_value = tq.simulate(br_exp_value_tmp)
    
    #print(exp_value, br_exp_value)
    assert np.isclose(exp_value, br_exp_value, atol=1.e-4)

    ######## Testing overlap #########
    
    objective_real, objective_im = make_overlap(U[1], U[0])
    
    Ex = tq.simulate(objective_real)
    Ey= tq.simulate(objective_im)
    
    overlap = Ex + 1.0j*Ey

    br_objective_real, br_objective_im = tq.braket(ket=U[0], bra=U[1])

    br_Ex = tq.simulate(br_objective_real)
    br_Ey = tq.simulate(br_objective_im)
    
    br_overlap = br_Ex + 1.0j*br_Ey
    
    assert np.isclose(br_overlap, overlap, atol=1.e-4)

    ######## Testing transition element #########
    
    trans_real, trans_im = make_transition(U0=U[1], U1=U[0], H=H)
    
    tmp_real = tq.simulate(trans_real)
    tmp_im = tq.simulate(trans_im)
    
    trans_el = tmp_real + 1.0j*tmp_im


    br_trans_real, br_trans_im = tq.braket(ket=U[0], bra=U[1], operator=H)

    br_tmp_real = tq.simulate(br_trans_real)
    br_tmp_im = tq.simulate(br_trans_im)
    
    br_trans_el = br_tmp_real + 1.0j*br_tmp_im

    assert np.isclose(br_trans_el, trans_el, atol=1.e-4)

    return
