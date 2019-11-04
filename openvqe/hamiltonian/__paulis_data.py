"""
Explicit matrix forms for the Pauli operators
"""
import numpy as np

pauli_matrices = {
    'I':np.array([[  1,  0 ], [  0,  1 ]], dtype=np.complex),
    'Z':np.array([[  1,  0 ], [  0, -1 ]], dtype=np.complex),
    'X':np.array([[  0,  1 ], [  1,  0 ]], dtype=np.complex),
    'Y':np.array([[  0, 1j ], [-1j,  0 ]], dtype=np.complex)}

