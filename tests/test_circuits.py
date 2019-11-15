from openvqe.circuit.gates import X, Y, Z, Rx, Ry, Rz, H, CNOT, SWAP, QCircuit, RotationGate
from openvqe.simulator.simulator_symbolic import SimulatorSymbolic, sympy
from openvqe.qubit_wavefunction import QubitWaveFunction
from openvqe.circuit._gates_impl import RotationGateImpl
from openvqe.circuit.variable import Variable
import numpy


def test_conventions():
    qubit = numpy.random.randint(0, 3)
    angle = Variable()

    Rx1 = Rx(target=qubit, angle=angle)
    Rx2 = QCircuit.wrap_gate(RotationGateImpl(axis="X", target=qubit, angle=angle))
    Rx3 = QCircuit.wrap_gate(RotationGateImpl(axis="x", target=qubit, angle=angle))
    Rx4 = RotationGate(axis=0, target=qubit, angle=angle)
    Rx5 = RotationGate(axis="X", target=qubit, angle=angle)
    Rx6 = RotationGate(axis="x", target=qubit, angle=angle)
    Rx7 = RotationGate(axis=0, target=qubit, angle=angle)
    Rx7.axis = "X"

    ll = [Rx1, Rx2, Rx3, Rx4, Rx5, Rx6, Rx7]
    for l1 in ll:
        for l2 in ll:
            assert (l1 == l2)

    qubit = 2
    for c in [None, 0, 3]:
        for angle in ["angle", 0, 1.234]:
            for axes in [[0, "x", "X"], [1, "y", "Y"], [2, "z", "Z"]]:
                ll = [RotationGate(axis=i, target=qubit, control=c, angle=angle) for i in axes]
                for l1 in ll:
                    for l2 in ll:
                        assert (l1 == l2)
                        l1.axis = axes[numpy.random.randint(0, 2)]
                        assert (l1 == l2)

def strip_sympy_zeros(wfn:QubitWaveFunction):
    result = QubitWaveFunction()
    for k,v in wfn.items():
        if v !=0:
            result[k]=v
    return  result


def test_basic_gates():
    I = sympy.I
    cos = sympy.cos
    sin = sympy.sin
    exp = sympy.exp
    BS = QubitWaveFunction.from_int
    angle = sympy.pi
    gates = [X(0), Y(0), Z(0), Rx(target=0, angle=angle), Ry(target=0, angle=angle), Rz(target=0, angle=angle), H(0)]
    results = [
        BS(1),
        I * BS(1),
        BS(0),
        cos(-angle / 2) * BS(0) + I * sin(-angle / 2) * BS(1),
        cos(-angle / 2) * BS(0) + I * sin(-angle / 2) * I * BS(1),
        exp(-I * angle / 2) * BS(0),
        1 / sympy.sqrt(2) * (BS(0) + BS(1))
    ]
    for i, g in enumerate(gates):
        wfn = SimulatorSymbolic().simulate_wavefunction(abstract_circuit=g, initial_state=0).backend_result
        assert (wfn == strip_sympy_zeros(results[i]))


def test_consistency():
    angle = sympy.pi / 2
    cpairs = [
        (CNOT(target=0, control=1), X(target=0, control=1)),
        (Ry(target=0, angle=sympy.pi), Rz(target=0, angle=4 * sympy.pi) + X(target=0)),
        (Rz(target=0, angle=sympy.pi), Rz(target=0, angle=sympy.pi) + Z(target=0)),
        (Rz(target=0, angle=angle), Rz(target=0, angle=angle / 2) + Rz(target=0, angle=angle / 2)),
        (Rx(target=0, angle=angle), Rx(target=0, angle=angle / 2) + Rx(target=0, angle=angle / 2)),
        (Ry(target=0, angle=angle), Ry(target=0, angle=angle / 2) + Ry(target=0, angle=angle / 2))
    ]

    for c in cpairs:
        print("circuit=", c[0], "\n", c[1])
        wfn1 = SimulatorSymbolic().simulate_wavefunction(abstract_circuit=c[0], initial_state=0).wavefunction
        wfn2 = SimulatorSymbolic().simulate_wavefunction(abstract_circuit=c[1], initial_state=0).wavefunction
        assert (wfn1 == wfn2)


def test_arithmetic():
    for c in [None, 4, [4, 5]]:
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

        # not supported yet
        # X1 = X(target=qubit, control=c, power=2.0)
        # X2 = X(target=qubit, control=c, power=1.0)
        # assert (X1 == X2*X2)


if __name__ == "__main__":
    test_basic_gates()
