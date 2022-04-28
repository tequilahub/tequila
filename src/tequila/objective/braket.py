from tequila.circuit.circuit import QCircuit, find_unused_qubit
from tequila.circuit.gates import H, X, Y, PauliGate
from tequila.hamiltonian.qubit_hamiltonian import QubitHamiltonian
from tequila.objective.objective import ExpectationValue
from tequila.hamiltonian import paulis

import numpy as np

def braket(ket, bra = None, operator = None):
    '''
    
    '''

    return


def make_overlap(U0:QCircuit = None, U1:QCircuit = None)->ExpectationValue:
    '''
    Function that calculates the overlap between two quantum states.

    Parameters
    ----------
    U0 : QCircuit tequila object, corresponding to the first state.
         
    U1 : QCircuit tequila object, corresponding to the second state.

    Returns
    -------
    Real and imaginary Tequila objectives to be simulated or compiled.

    '''
    
    ctrl = find_unused_qubit(U0=U0, U1=U1)
    
    print('Control qubit:',ctrl)
    
    U_a = U0.add_controls([ctrl]) #this add the control by modifying the previous circuit
    U_b = U1.add_controls([ctrl]) #NonType object
    
    #bulding the circuit for the overlap evaluation
    circuit = H(target=ctrl)
    circuit += U_a
    circuit += X(target=ctrl)
    circuit += U_b
    
    x = paulis.X(ctrl)
    y = paulis.Y(ctrl)
    Ex = ExpectationValue(H=x, U=circuit)
    Ey = ExpectationValue(H=y, U=circuit)
    
    return Ex, Ey


def make_transition(U0:QCircuit = None, U1:QCircuit = None, H: QubitHamiltonian = None) -> ExpectationValue:
    '''
    Function that calculates the transition elements of an Hamiltonian operator
    between two different quantum states.

    Parameters
    ----------
    U0 : QCircuit tequila object, corresponding to the first state.
         
    U1 : QCircuit tequila object, corresponding to the second state.
    
    H : QubitHamiltonian tequila object
        
    Returns
    -------
    Real and imaginary Tequila objectives to be simulated or compiled.

    '''
    
    # want to measure: <U1|H|U0> -> \sum_k c_k <U1|U_k|U0>
    
    trans_real = 0
    trans_im = 0
    
    for ps in H.paulistrings:
        #print('string',ps)
        c_k = ps.coeff
        #print('coeff', c_k)
        U_k = PauliGate(ps)
        objective_real, objective_im = make_overlap(U0=U0, U1=U1+U_k)
        
        trans_real += c_k*objective_real
        trans_im += c_k*objective_im

        #print('contribution', trans_real+trans_im)
        
    return trans_real, trans_im
