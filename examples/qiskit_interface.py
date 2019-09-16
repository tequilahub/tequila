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

    # Bell states example
    print("\n\nBell State\n")
    c = gates.H(target=0)*gates.CNOT(target=1, control=0)*gates.Measurement(target=[0,1])
    result = simulator.run(abstract_circuit=c, samples=1000)

    wfn = simulator.simulate_wavefunction(abstract_circuit=c).wavefunction
    print("counts:\n", result.measurements)
    print("wfn:\n", wfn)

    # notation example
    print("\n\nNotation\n")
    c = gates.X(target=0)
    c.n_qubits = 2
    wfn = simulator.simulate_wavefunction(abstract_circuit=c).wavefunction
    print("wfn:\n", wfn)
