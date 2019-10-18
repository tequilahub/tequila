from openvqe.circuit import gates
from openvqe.simulator.simulator_qiskit import SimulatorQiskit
from openvqe.simulator.simulator_cirq import SimulatorCirq
from openvqe.simulator.simulator_pyquil import SimulatorPyquil
from openvqe.tools import plotters
from openvqe.simulator.heralding import HeraldingProjector
from numpy import pi

"""
Create a linear combination of two unitaries:
We prepare a circuit which prepares the state:

    |00> --> 0.5*|0>(U_0+U_1)|0> + 0.5*|1>(U_0-U_1)|0> 

using a postselection strategy with the HeraldingProjector
we count only measurements that correspond to the (U_0 + U_1)|0> part
or respectively the other.

See also the heralding.py example

Note that the Heralding objects can not yet be used in combination with Objectives
"""


if __name__ == "__main__":

    # Create the circuit
    U  = gates.H(target=0)
    U += gates.Ry(target=1, angle=pi/2, control=0)
    U += gates.X(target=0)
    U += gates.Ry(target=1, angle=-pi/2, control=0)
    U += gates.X(target=0)
    U += gates.H(target=0)

    # initialize the heralding object
    # register: the full qubit register
    # subregister: the domain of qubits which are anylyzed
    # projector_space: all bitstrings that are valid on the subregister domain
    heralding = HeraldingProjector(register=[0,1], subregister=[1], projector_space=["1"])

    simulator = SimulatorQiskit(heralding=heralding)

    # we need to add measurement instructions for that
    U += gates.Measurement(target=[0,1])

    result = simulator.run(abstract_circuit=U, samples=10000)

    counts = result.counts

    print("counts:\n", counts)

    plotters.plot_counts(counts=counts)
