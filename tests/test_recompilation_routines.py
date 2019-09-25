from openvqe.circuit import gates
from openvqe.circuit.compiler import compile_controlled_rotation_gate
from numpy.random import uniform, randint
from numpy import pi
from openvqe.simulator.simulator_cirq import SimulatorCirq

def test_controlled_rotations():
    angles = uniform(0, 2 * pi, 5)
    gs = [gates.Rx, gates.Ry, gates.Rz]
    for angle in angles:
        for gate in gs:
            qubit = randint(0, 1)
            control = randint(2, 3)
            U = gate(target=qubit, control=control, angle=angle)*gates.X(target=control)
            RCU = compile_controlled_rotation_gate(gate=U)
            wfn1 = SimulatorCirq().simulate_wavefunction(abstract_circuit=U, initial_state=0).wavefunction
            wfn2 = SimulatorCirq().simulate_wavefunction(abstract_circuit=RCU, initial_state=0).wavefunction
            assert (wfn1 == wfn2)


