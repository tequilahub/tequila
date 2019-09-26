from openvqe.circuit import gates
from openvqe.circuit.compiler import compile_controlled_rotation_gate, change_basis
from numpy.random import uniform, randint
from numpy import pi, isclose
from openvqe.simulator.simulator_cirq import SimulatorCirq
from openvqe.hamiltonian import PX, PY, PZ
from openvqe.objective import Objective


def test_controlled_rotations():
    angles = uniform(0, 2 * pi, 5)
    gs = [gates.Rx, gates.Ry, gates.Rz]
    for angle in angles:
        for gate in gs:
            qubit = randint(0, 1)
            control = randint(2, 3)
            U = gates.X(target=control) * gate(target=qubit, control=control, angle=angle)
            RCU = compile_controlled_rotation_gate(gate=U)
            wfn1 = SimulatorCirq().simulate_wavefunction(abstract_circuit=U, initial_state=0).wavefunction
            wfn2 = SimulatorCirq().simulate_wavefunction(abstract_circuit=RCU, initial_state=0).wavefunction
            assert (wfn1 == wfn2)


def test_basis_change():
    for angle in list(uniform(0, 2 * pi, 5)):
        EX = SimulatorCirq().expectation_value(
            objective=Objective(unitaries=[gates.Rx(target=0, angle=angle)], observable=PX(0)))
        EY = SimulatorCirq().expectation_value(
            objective=Objective(unitaries=[gates.Rx(target=0, angle=angle)], observable=PY(0)))
        EZ = SimulatorCirq().expectation_value(
            objective=Objective(unitaries=[gates.Rx(target=0, angle=angle)], observable=PZ(0)))

        EXX = SimulatorCirq().expectation_value(
            objective=Objective(unitaries=[gates.Rx(target=0, angle=angle) * change_basis(target=0, axis=0)],
                                observable=PZ(0)))
        EYY = SimulatorCirq().expectation_value(
            objective=Objective(unitaries=[gates.Rx(target=0, angle=angle) * change_basis(target=0, axis=1)],
                                observable=PZ(0)))
        EZZ = SimulatorCirq().expectation_value(
            objective=Objective(unitaries=[gates.Rx(target=0, angle=angle) * change_basis(target=0, axis=2)],
                                observable=PZ(0)))

        assert (isclose(EX, EXX))
        assert (isclose(EY, EYY))
        assert (isclose(EZ, EZZ))

    for i,gate in enumerate([gates.Rx, gates.Ry, gates.Rz]):
        angle = uniform(0, 2*pi)
        U1 = gate(target=0, angle=angle)
        U2 = change_basis(target=0, axis=i)*gates.Rz(target=0, angle=angle)*change_basis(target=0, axis=i, daggered=True)
        wfn1 = SimulatorCirq().simulate_wavefunction(abstract_circuit=U1)
        wfn2 = SimulatorCirq().simulate_wavefunction(abstract_circuit=U2)
        assert(wfn1.wavefunction == wfn2.wavefunction)
        wfn1 = SimulatorCirq().simulate_wavefunction(abstract_circuit=U1, initial_state=1)
        wfn2 = SimulatorCirq().simulate_wavefunction(abstract_circuit=U2, initial_state=1)
        assert(wfn1.wavefunction == wfn2.wavefunction)

