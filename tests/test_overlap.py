import tequila as tq
import numpy as np
from tequila.circuit.gates import PauliGate
from tequila.overlap.overlap import make_overlap, make_transition



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
    Ey= tq.simulate(objective_im)
    
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
    rotation_gates = ['rx', 'ry', 'rz']
    
    #np.random.seed(111)
    n_qubits = np.random.randint(1, high=5)
    
    U = {}
    for j in range(2):
        n_rotations = np.random.randint(n_qubits, high=n_qubits*3)
        gates_list = [np.random.choice(rotation_gates) for i in range(n_rotations)]
        
        angles = 2*np.pi * np.random.rand(n_rotations)
        
        circ = tq.QCircuit()
        for i, angle in enumerate(angles):
            qb = i%n_qubits+1
            
            if gates_list[i]=='rx':
                circ += tq.gates.Rx(angle=angle, target=qb)
            
            elif gates_list[i]=='ry':
                circ += tq.gates.Ry(angle=angle, target=qb)
                
            elif gates_list[i]=='rz':
                circ += tq.gates.Rz(angle=angle, target=qb)
        U[j] = circ
    
    objective_real, objective_im = make_overlap(U[0],U[1])
    
    Ex = tq.simulate(objective_real)
    Ey= tq.simulate(objective_im)
    
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
    U1 = tq.gates.X(target=[1,2])+tq.gates.Ry(angle=2, target=1)#
    
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
    
    rotation_gates = ['rx', 'ry', 'rz']
    
    #np.random.seed(111)
    n_qubits = np.random.randint(1, high=5)
    #print(n_qubits)
    
    U = {}
    for j in range(2):
        n_rotations = np.random.randint(n_qubits, high=n_qubits*3)
        gates_list = [np.random.choice(rotation_gates) for i in range(n_rotations)]
        
        angles = 2*np.pi * np.random.rand(n_rotations)
        
        circ = tq.QCircuit()
        for i, angle in enumerate(angles):
            qb = i%n_qubits+1
            
            if gates_list[i]=='rx':
                circ += tq.gates.Rx(angle=angle, target=qb)
            
            elif gates_list[i]=='ry':
                circ += tq.gates.Ry(angle=angle, target=qb)
                
            elif gates_list[i]=='rz':
                circ += tq.gates.Rz(angle=angle, target=qb)
        U[j] = circ
    
    #print(U[0])
    #print(U[1])
    
    #make random hamiltonian
    paulis = ['X','Y','Z']
    n_ps = np.random.randint(1, high=2*n_qubits+1)
    
    ham = ''
    for ps in range(n_ps):
        #"1.0*Y(0)X(1)+0.5*Y(1)Z(0)"
        coeff = '{}*'.format(round(np.random.sample(),2))
        pauli_str = ''
        for qb in range(n_qubits):
            pauli_str += '{}({})'.format(np.random.choice(paulis), qb)
        
        if ps < n_ps-1:
            pauli_str += '+'
                
        ham += coeff+pauli_str
    
    #print(ham)
    
    H = tq.QubitHamiltonian(ham)
    
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
