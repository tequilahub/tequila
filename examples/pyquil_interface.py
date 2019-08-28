from openvqe.simulator.simulator_pyquil import SimulatorPyquil
from openvqe.circuit.circuit import QCircuit
from openvqe.circuit.gates import  Ry, X
from openvqe.simulator.simulator import SimulatorReturnType
from numpy import pi

"""
Play around with pyquil simulator interface
"""


class CustomReturnType(SimulatorReturnType):
    """
    Example how to use custom return types
    """

    def __post_init__(self, result):
        self.result = result
        pass

    def __repr__(self):
        return str(self.result)
        pass


if __name__ == "__main__":
    ac = QCircuit()
    ac *= X(0)
    ac *= Ry(target=1, control=0, angle=pi / 2)

    simulator = SimulatorPyquil()

    full_state = simulator.simulate_wavefunction(abstract_circuit=ac)

    print("only_state result:\n", type(full_state), "\n", full_state)

    custom_state = simulator.simulate_wavefunction(abstract_circuit=ac, returntype=CustomReturnType)

    print("result with custom return type:\n", type(custom_state), "\n", custom_state)
    print("\nPyquil program:\n", custom_state.circuit)
