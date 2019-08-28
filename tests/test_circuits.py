from openvqe.circuit.gates import X, Y, Z, Rx, Ry, Rz, H, CNOT, SWAP
from openvqe.simulator.simulator_symbolic import SimulatorSymbolic, QState, sympy
import numpy


def test_basic_gates():
    I = sympy.I
    cos = sympy.cos
    sin = sympy.sin
    exp = sympy.exp
    BS = QState.initialize_from_integer
    angle = sympy.pi
    gates = [X(0), Y(0), Z(0), Rx(target=0, angle=angle), Ry(target=0, angle=angle), Rz(target=0, angle=angle), H(0)]
    results = [
        BS(1),
        -I * BS(1),
        BS(0),
        cos(angle / 2) * BS(0) + -I * sin(angle / 2) * BS(1),
        cos(angle / 2) * BS(0) + - sin(angle / 2) * BS(1),
        exp(-I * angle / 2) * BS(0),
        1 / sympy.sqrt(2) * (BS(0) + BS(1))
    ]
    for i, g in enumerate(gates):
        wfn = SimulatorSymbolic().simulate_wavefunction(abstract_circuit=g, initial_state=0)
        assert (wfn == results[i])


def test_consistency():
    angle = sympy.pi / 2
    cpairs = [
        (CNOT(target=0, control=1), X(target=0, control=1)),
        (Ry(target=0, angle=sympy.pi), Rz(target=0, angle=-2 * sympy.pi) * X(target=0)),
        (Rz(target=0, angle=sympy.pi), Rz(target=0, angle=sympy.pi) * Z(target=0)),
        (Rz(target=0, angle=angle), Rz(target=0, angle=angle / 2) * Rz(target=0, angle=angle / 2)),
        (Rx(target=0, angle=angle), Rx(target=0, angle=angle / 2) * Rx(target=0, angle=angle / 2)),
        (Ry(target=0, angle=angle), Ry(target=0, angle=angle / 2) * Ry(target=0, angle=angle / 2))
    ]

    for c in cpairs:
        wfn1 = SimulatorSymbolic().simulate_wavefunction(abstract_circuit=c[0], initial_state=0)
        wfn2 = SimulatorSymbolic().simulate_wavefunction(abstract_circuit=c[1], initial_state=0)
        assert (wfn1 == wfn2)


def test_arithmetic():
    for c in [None, 4, [4,5]]:
        qubit = numpy.random.randint(0, 3)
        power = numpy.random.uniform(0, 5)
        X2 = X(target=qubit, control=c, power=power)
        X1 = X(target=qubit, control=c, power=1.0)
        assert (X2 == X1 ** power)
        X2 = Y(target=qubit, control=c, power=power)
        X1 = Y(target=qubit, control=c, power=1.0)
        assert (X2 == X1 ** power)
        X2 = Z(target=qubit, control=c, power=power)
        X1 = Z(target=qubit, control=c, power=1.0)
        assert (X2 == X1 ** power)
        X2 = H(target=qubit, control=c, power=power)
        X1 = H(target=qubit, control=c, power=1.0)
        assert (X2 == X1 ** power)
        X2 = Rx(target=qubit, control=c, angle=power)
        X1 = Rx(target=qubit, control=c, angle=1.0)
        assert (X2 == X1 ** power)
        X2 = Ry(target=qubit, control=c, angle=power)
        X1 = Ry(target=qubit, control=c, angle=1.0)
        assert (X2 == X1 ** power)
        X2 = Rz(target=qubit, control=c, angle=power)
        X1 = Rz(target=qubit, control=c, angle=1.0)
        assert (X2 == X1 ** power)


if __name__ == "__main__":
    test_basic_gates()
