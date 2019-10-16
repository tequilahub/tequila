from openvqe.circuit import gates
from numpy import pi
from openvqe.tools.backends import initialize_simulator

"""
From the current backends only Pyquil and Cirq support full wavefunction
Simulation. We also have an internal symbolic simulator which was created
for debugging purposes. It can be described with many words but 'fast'
or 'efficient' wouldn't be one of them.

If you use Pyquil, make sure you have installed the QVM
 -> open terminal, type 'qvm -S'
"""


if __name__ == "__main__":

    """
    Here comes the actual example
    """

    # create a simple abstract circuit
    ac = gates.X(0)+gates.Ry(target=1, control=0, angle=pi / 2)

    # choose a backend which supports full wavefunction simulation
    # Qiskit can not do it for example --> error will be thrown when you try it later
    simulator = initialize_simulator(simulator_type='symbolic') # choose between 'cirq', 'pyquil', 'symbolic'

    simulator_results = simulator.simulate_wavefunction(abstract_circuit=ac, initial_state=0)
    wfn = simulator_results.wavefunction

    print("backend translated circuit:\n", simulator_results.circuit, "\n")

    print("resulting wavefunction:\n", wfn, "\n")

    # if you want to work with the backend circuit structure directly
    backend_circuit = simulator.create_circuit(abstract_circuit=ac)
    print("Type of backend circuit: ", type(backend_circuit), " \n")






