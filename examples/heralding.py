"""
Simple example how to use Heralding
"""

from openvqe.simulator.heralding import HeraldingProjector
from openvqe.circuit import gates
from openvqe.simulator.simulator_qiskit import SimulatorQiskit
from openvqe.simulator.simulator_cirq import SimulatorCirq
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
