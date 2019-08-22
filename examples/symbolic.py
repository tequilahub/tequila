"""
Hi Maha, here an illustration how to use stuff
Should be quite similar to before
Two ways to get OpenVQE:

Either:
Add the path to the OpenVQE directory to your pythonpath
in a terminal:
export PYTHONPATH=${PYTHONPATH}:/path/to/wherever/you/saved/it/OpenVQE/
in your script:
import sys
sys.path.append("/path/to/wherever/you/saved/it/OpenVQE/")

Or:
install OpenVQE with
"pip install ." in the OpenVQE directory (the directory which has setup.py in it)
this will install a minimal package without too many dependencies

"""

from openvqe.simulator import SimulatorSymbolic
from openvqe.circuit import QCircuit, gates
import sympy


if __name__ == "__main__":

    # gates are in openvqe.circuit.gates
    Ry = gates.Ry(target=0, control=None, angle=sympy.Symbol("a"))
    cnot = gates.CNOT(control=0, target=1)

    circuit = Ry + cnot
    # equivalent: cnot=gates.X(control=0, target=1)

    simulator = SimulatorSymbolic()
    result = simulator.simulate_wavefunction(abstract_circuit=circuit)

    print(result)

