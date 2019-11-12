"""
Example File on how to initialize Gates
which are exponentials of single paulistrings

ExponentialPauliGates follow the same conventions as rotational gates:

e.g.

Rz(target=0, angle) = Exp(-i angle/2 * Z(0))

ExpPauli(Z(0), angle) = Exp(-i angle/2 * Z(0))


"""

from openvqe.circuit import gates
from openvqe.hamiltonian import PauliString
from openvqe.simulator.simulator_cirq import SimulatorCirq


# default initialization
ps = PauliString(data={0:"x", 1:"y", 2:"z"}, coeff=2.0)
U = gates.ExpPauli(paulistring=ps, angle=2.0)
print(U)

# initializat from openfermion key
ps = PauliString.from_openfermion(key=[(0, "X"), (1, "Y"), (2, "Z")])
U = gates.ExpPauli(paulistring=ps, angle=2.0)
print(U)

# string based initialization
ps = PauliString.from_string("X(0)Y(1)Z(2)")
U = gates.ExpPauli(paulistring=ps, angle=2.0)
print(U)

# Direct initialization (should also work with the other two variants)
U = gates.ExpPauli(paulistring="X(0)Y(1)Z(2)", angle=2.0)
print(U)

result = SimulatorCirq().simulate_wavefunction(U)

print("This is the old abstract data object")
print(result.abstract_circuit)
print("This is how the circuit looks in the cirq backend")
print(result.circuit)


print("\nSee the conventions: The next two circuits should be the same")
U = gates.ExpPauli(paulistring="Z(0)", angle=2.0)
result = SimulatorCirq().simulate_wavefunction(U)
print(result.circuit)
U2 = gates.Rz(target=0, angle=2.0)
result = SimulatorCirq().simulate_wavefunction(U)
print(result.circuit)