from openvqe.simulator.simulator_cirq import SimulatorCirq
from openvqe.circuit.circuit import QCircuit
from openvqe.circuit.gates import Ry, X
from numpy import pi

"""
Play around with cirq simulator interface
"""

def make_gradient(circuit: QCircuit):
    grad_list = []
    for gate in circuit.gates:
        if gate.is_parametrized() and not gate.is_frozen():
            if gate.is_controlled():
                recompiled_gate = compile_controlled_rotation_gate(gate=gate)
                grad_list += make_gradient(recompiled_gate)
            else:
                shifted_gate = gate
                shifted_gate.angle=gate.angle/2
                grad_list.append(shifted_gate)
        else:
            grad_list.append(gate)

    return grad_list

if __name__ == "__main__":
    ac = QCircuit()
    ac += X(0)
    ac += Ry(target=1, control=0, angle=pi / 2)

    simulator = SimulatorCirq()

    custom_state = simulator.simulate_wavefunction(abstract_circuit=ac, initial_state=0)

    print("object result:\n", type(custom_state), "\n", custom_state)

    print("\nTranslated circuit:\n", custom_state.circuit)

    ac = QCircuit()
    ac += X(0)
    ac += X(1)
    density_matrix_result = simulator.simulate_density_matrix(abstract_circuit=ac)

    print("density_matrix_result:\n", density_matrix_result)
    print("density_matrix:\n", density_matrix_result.result.final_density_matrix)

    print("\n\nRecompile controled-rotations")
    from openvqe.circuit.compiler import compile_controlled_rotation_gate
    ac = X(0) + Ry(target=1, control=0, angle=pi / 2)
    # print(ac)
    # rac=ac.recompile_gates(instruction=compile_controlled_rotation_gate)
    # print("after recompilation:\n", rac)

    # should also work
    rac2 = compile_controlled_rotation_gate(gate=ac)
    print("after recompilation2:\n", rac2)

    angles = rac2.extract_parameters()
    print("angles=", angles)

    print("Try to get gradient for:")
    print(ac)
    grad_list = make_gradient(circuit=ac)
    print("grad_list=",grad_list)
    exit()
