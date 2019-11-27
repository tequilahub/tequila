from tequila.simulators.heralding import HeraldingProjector
from tequila.circuit import gates
from tequila.simulators.simulator_qiskit import SimulatorQiskit
from tequila.simulators.simulator_cirq import SimulatorCirq

import typing
import numpy
import pytest

@pytest.mark.parametrize("angle", [numpy.random.uniform(0, 2 * numpy.pi) for i in range(4)])
@pytest.mark.parametrize("backend", [SimulatorQiskit, SimulatorCirq])
def test_projector(angle, backend: typing.Union[SimulatorCirq, SimulatorQiskit]):
    targets = [1]
    ancillas = [0, 2]
    samples = 1000
    U = gates.Ry(target=0, angle=angle) + gates.Ry(target=0, angle=-angle) + gates.Measurement(target=0)
    Uh = gates.H(target=ancillas[0])+gates.H(target=ancillas[1])
    Uh += gates.Ry(target=targets[0], angle=angle, control=ancillas)
    Uh += gates.Ry(target=targets[0], angle=-angle, control=ancillas)
    Uh += gates.Measurement(target=targets+ancillas)
    heralding = HeraldingProjector(register=targets + ancillas, subregister=ancillas, projector_space=["11"])
    counts0 = backend().run(abstract_circuit=U, samples=samples).counts
    counts1 = backend(heralding=heralding).run(abstract_circuit=Uh, samples=4*samples).counts
    for k ,v in counts1.items():
        assert(k in counts0.keys())
        # tolerance here is more a heuristic thing, might fail for some instances
        assert(numpy.isclose(v, counts0[k], atol=100))