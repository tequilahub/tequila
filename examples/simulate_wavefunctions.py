from openvqe.circuit import gates
from numpy import pi

"""
From the current backends only Pyquil and Cirq support full wavefunction
Simulation. We also have an internal symbolic simulator which was created
for debugging purposes. It can be described with many words but 'fast'
or 'efficient' wouldn't be one of them.

If you use Pyquil, make sure the QVM runs in the background
 -> open terminal, type 'qvm -S'
And make sure the rigetti qvm is installed 
"""

def initialize_simulator(simulator_type:str):
    """
    Convenience for this example
    This function is only here so that the example runs even when not all backends are installed
    And it will allow for eay switching of backens to play around
    :param simulator_type: 'cirq', 'pyquil', 'symbolic' (default), 'qiskit'
    :return: the initialized simulator backend
    """
    # moving import statements to here, so the example also runs when not all are installed
    from openvqe.simulator.simulator_cirq import SimulatorCirq
    from openvqe.simulator.simulator_pyquil import SimulatorPyquil
    from openvqe.simulator.simulator_symbolic import SimulatorSymbolic
    from openvqe.simulator.simulator_qiskit import SimulatorQiskit

    if simulator_type.lower() == "cirq":
        return SimulatorCirq()
    elif simulator_type.lower() == 'pyquil':
        return SimulatorPyquil()
    elif simulator_type.lower() == 'qiskit':
        return SimulatorQiskit()
    else:
        return SimulatorSymbolic()

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







