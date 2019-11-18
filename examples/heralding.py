"""
Simple example how to use Heralding

First we prepare the state

Ry(angle)|0> = exp(-i*angle/2)|0>

and display the distribution it gives after several measurements

Then we prepare the state

CCRy(angle)_2 |++0> where CCRy is controlled by qubit 0 and qubit 1 which are in the |+> states

This gives of course a different distribution

At third we prepare the same state again but use 'heralding' to count only measurements where
qubit 0 and qubit 1 are in state |1>
This will then give the same distribution as in the first state

Note that the Heralding objects can not yet be used in combination with Objectives
"""

from openvqe.simulators.heralding import HeraldingProjector
from openvqe.circuit import gates
from openvqe.simulators.simulator_qiskit import SimulatorQiskit
from openvqe.simulators.simulator_cirq import SimulatorCirq
from openvqe import numpy
from openvqe.tools import plotters
from openvqe import BitString

if __name__ == "__main__":
    targets = [1]
    ancillas = [0, 2]
    angle = numpy.pi / 2
    samples = 10000

    U = gates.Ry(target=targets[0], angle=angle) + gates.Measurement(target=targets)

    counts = SimulatorCirq().run(abstract_circuit=U, samples=samples).counts

    plotters.plot_counts(counts=counts, title="Plain One-Qubit Example")

    U = gates.H(target=ancillas[0]) + gates.H(target=ancillas[1])
    U += gates.Ry(target=targets[0], control=ancillas, angle=angle)
    U += gates.Measurement(target=targets+ancillas)

    counts = SimulatorCirq().run(abstract_circuit=U, samples=samples).counts
    plotters.plot_counts(counts=counts, title="Full Two-Qubit Example")

    heralder = HeraldingProjector(register=targets + ancillas, subregister=ancillas, projector_space=["11"])
    counts = SimulatorCirq(heralding=heralder).run(abstract_circuit=U, samples=samples).counts
    plotters.plot_counts(counts=counts, title="Heralded Two-Qubit Example")
