from openvqe.circuit.simulator import SimulatorReturnType
from openvqe.circuit.simulator_cirq import SimulatorCirq
from openvqe.circuit.circuit import QCircuit, Ry, X, CNOT
from numpy import pi
import cirq

"""
Play around with cirq simulator interface
"""


class CustomReturnType(SimulatorReturnType):
    """
    Example how to use custom return types
    """

    def __init__(self, result, circuit, abstract_circuit, *args, **kwargs):
        self.state_array = result.final_simulator_state.state_vector
        self.qubit_map = result.final_simulator_state.qubit_map
        self.dirac_string = result.dirac_notation()
        self.circuit = circuit
        self.abstract_circuit = abstract_circuit

    def __repr__(self):
        result = "This is CustomReturnType with the following state\n"
        result += self.dirac_string
        result += "\nproducted by the circuit:\n"
        result += str(self.circuit)
        return result


if __name__ == "__main__":
    ac = QCircuit()
    ac += X(0)
    ac += Ry(target=1, control=0, angle=pi / 2)

    simulator = SimulatorCirq()

    full_state = simulator.simulate_wavefunction(abstract_circuit=ac, returntype="full_state", initial_state=0)

    print("only_state result:\n", type(full_state), "\n", full_state)

    object_state = simulator.simulate_wavefunction(abstract_circuit=ac, returntype="object", initial_state=0)

    print("object result:\n", type(object_state), "\n", object_state)

    custom_state = simulator.simulate_wavefunction(abstract_circuit=ac, returntype=CustomReturnType, initial_state=0)

    print("object result:\n", type(custom_state), "\n", custom_state)

    print("\nTranslated circuit:\n", custom_state.circuit)

    ac = QCircuit()
    ac += X(0)
    ac += X(1)
    density_matrix_result = simulator.simulate_density_matrix(abstract_circuit=ac)

    print("density_matrix_result:\n", density_matrix_result)
    print("density_matrix:\n", density_matrix_result.result.final_density_matrix)

    try:
        ac = QCircuit()
        ac += X(0)
        #ac += CNOT(control=0, target=1) # with this: no error but wrong density matrix
        ac += X(target=1, control=0)
        density_matrix_result = simulator.simulate_density_matrix(abstract_circuit=ac)

        print("density_matrix_result:\n", density_matrix_result)
        print("density_matrix:\n", density_matrix_result.result.final_density_matrix)
        print("circuit:\n", density_matrix_result.circuit)

    except AttributeError as e:
        print("cirq currently can not do controlled operations in density matrix simulations")
        print("caught error:", e)