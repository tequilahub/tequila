from openvqe.circuit import gates
from numpy import pi

"""
Same example as 'simulate_wavefunctions' but here we simulate counts

If you use Pyquil, make sure the QVM runs in the background
 -> open terminal, type 'qvm -S'
And make sure the rigetti qvm is actually installed 
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

    # we need to add the measurements, here we measure both qubits
    ac += gates.Measurement(target=[0,1], name="a") #naming is possible but is not supported by every backend

    # choose a backend
    # 'symbolic' can only simulate wavefunctions, counts are not supported here. Could be implemented quickly
    # 'pyquil' has currently no support ... on the list
    simulator = initialize_simulator(simulator_type='qiskit') # choose between 'cirq', 'pyquil', 'qiskit'

    simulator_results = simulator.run(abstract_circuit=ac, samples=100)
    counts = simulator_results.counts

    print("circuit:\n", simulator_results.circuit, "\n")
    print("counts:\n", counts)
