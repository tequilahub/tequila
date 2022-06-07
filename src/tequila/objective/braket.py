from tequila.circuit.circuit import QCircuit, find_unused_qubit
from tequila.circuit.gates import H, X, Y, PauliGate
from tequila.hamiltonian.qubit_hamiltonian import QubitHamiltonian
from tequila.objective.objective import ExpectationValue, Objective
from tequila.hamiltonian import paulis

import numpy as np

def braket(ket: QCircuit, bra: QCircuit = None, operator: QubitHamiltonian = None) -> ExpectationValue:
    """Function that allows to calculate different quantities 
       depending on the passed parameters:
       1) If only ket is passed, returns the overlap with itself (1).
       2) If ket and bra are passed, returns the overlap between the two states.
       3) If ket and operator are passed, returns the expectation value of the operator for the given state.
       4) If ket, bra and operator are passed, returns the transition element of the operator.

    Args:
        ket (QCircuit): QCircuit corresponding to a state. 
        bra (QCircuit, optional): QCircuit corresponding to a second state.
                                  Defaults to None.
        operator (QubitHamiltonian, optional): Operator of which we want to 
                                               calculate the transition element. 
                                               Defaults to None.

    Returns:
        ExpectationValue: 1, overlap, expectation value or transition element 
                          depending on the inputs.
    """

    if bra is None:
        bra = ket

    if id(ket) == id(bra):
        if operator is None:
            return Objective()+1.0 , Objective() 
        return ExpectationValue(H=operator, U=ket) , Objective()
    else:
        if operator is None:
            return make_overlap(U0 = bra, U1 = ket)
        
        return make_transition(U0 = bra, U1 = ket, H = operator)

def make_overlap(U0:QCircuit = None, U1:QCircuit = None) -> ExpectationValue:
    '''
    Function that calculates the overlap between two quantum states.

    Parameters
    ----------
    U0 : QCircuit tequila object, corresponding to the first state (will be the bra).
         
    U1 : QCircuit tequila object, corresponding to the second state (will be the ket).

    Returns
    -------
    Real and imaginary Tequila objectives to be simulated or compiled.

    '''
    
    ctrl = find_unused_qubit(U0=U0, U1=U1)
    
    #print('Control qubit:',ctrl)
    
    U_a = U0.add_controls([ctrl]) #this add the control by modifying the previous circuit
    U_b = U1.add_controls([ctrl]) #NonType object
    
    #bulding the circuit for the overlap evaluation
    circuit = H(target=ctrl)
    circuit += X(target=ctrl)
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
    U0 : QCircuit tequila object, corresponding to the first state (will be the bra).
         
    U1 : QCircuit tequila object, corresponding to the second state (will be the ket).
    
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
