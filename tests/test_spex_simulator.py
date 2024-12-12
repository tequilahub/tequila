import tequila as tq
import numpy as np
from tequila.hamiltonian import PauliString

U = tq.gates.ExpPauli(paulistring=PauliString({0: "X", 1: "Z"}), angle=np.pi / 2)

H = tq.QubitHamiltonian("Z(0)")

E = tq.ExpectationValue(U=U, H=H)

print("\nspex:", tq.simulate(E, backend='spex'))
print("qulacs:", tq.simulate(E, backend='qulacs'))
