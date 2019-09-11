from openvqe.circuit import QCircuit, gates
from openvqe.circuit.compiler import compile_controlled_rotation_gate, compile_multitarget
from numpy.random import uniform, randint, choice
from numpy import pi, isclose
from sympy import Float, Abs
from openvqe.simulator.simulator_symbolic import SimulatorSymbolic


def test_controlled_rotations():
    angles = uniform(0, 2 * pi, 5)
    gs = [gates.Rx, gates.Ry, gates.Rz]
    for angle in angles:
        for gate in gs:
            qubit = randint(0, 1)
            control = randint(2, 3)
            U = gate(target=qubit, control=control, angle=angle)
            RCU = compile_controlled_rotation_gate(gate=U)
            wfn1 = SimulatorSymbolic().simulate_wavefunction(abstract_circuit=U, initial_state=0)
            wfn2 = SimulatorSymbolic().simulate_wavefunction(abstract_circuit=RCU, initial_state=0)
            assert (Abs(wfn1.inner(wfn2).evalf() - 1.0) < Float(1.e-5))


