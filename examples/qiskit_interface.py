"""
Currently only the construction of qiskit circuits is implemented
"""

from openvqe.simulator.simulator_qiskit import SimulatorQiskit
from openvqe.circuit import gates

if __name__ == "__main__":

    simulator = SimulatorQiskit()
    testc = [gates.X(target=0), gates.X(target=1), gates.X(target=1, control=0),
             gates.Rx(target=1, control=0, angle=2.0), gates.Ry(target=1, control=0, angle=2.0),
             gates.Rz(target=1, control=0, angle=2.0)]

    for c in testc:
        c_qiskit = simulator.create_circuit(abstract_circuit=c)
        print(c_qiskit.draw())