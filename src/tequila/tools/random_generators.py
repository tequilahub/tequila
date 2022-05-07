import numpy as np
from tequila.circuit import gates
from tequila.circuit.circuit import QCircuit
from tequila.hamiltonian.qubit_hamiltonian import QubitHamiltonian

def make_random_circuit(n_qubits: int, rotation_gates: list=['rx', 'ry', 'rz'], n_rotations: int=None,
                        enable_controls: bool=None) -> QCircuit:
    """Function that creates a circuit with random rotations or random control rotations.

    Args:
        n_qubits (int): Dimension of the quantum register of the circuit
        rotation_gates (list): List of possible rotations in str form. Default to [rx, ry, rz].
        n_rotations (int): Number of rotations gates in the circuit. Default to None.
        enable_controls (bool): Boolean that switch on controls. Default to None.

    Returns:
        QCircuit: Random quantum circuit consiting of the given rotations gates 
        and their controlled versions
    """
    if n_rotations is None:
        n_rotations = np.random.randint(n_qubits, high=n_qubits*3)

    gates_list = [np.random.choice(rotation_gates) for i in range(n_rotations)]
    
    angles = 2*np.pi * np.random.rand(n_rotations)
    
    circ = QCircuit()
    for i, angle in enumerate(angles):
        target = i%n_qubits
        if enable_controls:
            controls = [i for i in circ.qubits if i != target]
            control = np.random.choice(controls + [None])
        else: 
            control = None

        if gates_list[i]=='rx':
            circ += gates.Rx(angle=angle, target=target, control=control)
        
        elif gates_list[i]=='ry':
            circ += gates.Ry(angle=angle, target=target, control=control)
            
        elif gates_list[i]=='rz':
            circ += gates.Rz(angle=angle, target=target, control=control)
        
    return circ

def make_random_hamiltonian(n_qubits: int , paulis: list=['X','Y','Z'], n_ps: int = None) -> QubitHamiltonian:
    """Function that creates a random Hamiltonian, given the list
       of Pauli ops. to use and the number of Pauli strings.

    Args:
        n_qubits (int): Dimension of the quantum register of the circuit
        paulis (list): List of possible Pauli operators in str form. Default to ['X','Y','Z'].
        n_ps (int): Number of Pauli strings composing the Hamiltonian. Default to None.

    Returns:
        tq.QubitHamiltonian: Random Hamiltonian
    """
    if n_ps is None:
        n_ps = np.random.randint(1, high=2*n_qubits+1)
    
    ham = ''
    for ps in range(n_ps):
        coeff = '{}*'.format(round(np.random.sample(),2))
        pauli_str = ''
        for qb in range(n_qubits):
            pauli_str += '{}({})'.format(np.random.choice(paulis), qb)
        
        if ps < n_ps-1:
            pauli_str += '+'
                
        ham += coeff+pauli_str
    
    #print(ham)
    
    H = QubitHamiltonian(ham)
    return H
