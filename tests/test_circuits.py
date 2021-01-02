from tequila.circuit.gates import X, Y, Z, Rx, Ry, Rz, H, CNOT, QCircuit, RotationGate, Phase, ExpPauli, Trotterized, \
                                  U, u1, u2, u3
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila.circuit._gates_impl import RotationGateImpl
from tequila.objective.objective import Variable
from tequila.simulators.simulator_api import simulate
from tequila import assign_variable, paulis
import numpy, sympy
import pytest


def test_qubit_map():

    for gate in [Rx, Ry, Rz]:
        U1 = gate(angle="a", target=0, control=1)
        U2 = gate(angle="a", target=1, control=0)
        mapped = U1.map_qubits(qubit_map={0:1, 1:0})
        assert len(mapped.gates) == len(U2.gates)
        for i in range(len(mapped.gates)):
            for k, v in mapped.gates[i].__dict__.items():
                assert U2.gates[i].__dict__[k] == v

    for gate in [H, X, Y, Z]:
        U1 = gate(target=0, control=1)
        U2 = gate(target=1, control=0)
        mapped = U1.map_qubits(qubit_map={0:1, 1:0})
        assert len(mapped.gates) == len(U2.gates)
        for i in range(len(mapped.gates)):
            for k, v in mapped.gates[i].__dict__.items():
                assert U2.gates[i].__dict__[k] == v

    for gate in [ExpPauli]:
        U1 = gate(angle="a", paulistring="X(0)Y(2)Z(3)", control=1)
        U2 = gate(angle="a", paulistring="X(5)Y(6)Z(7)", control=0)
        mapped = U1.map_qubits(qubit_map={0:5, 1:0, 2:6, 3:7})
        assert len(mapped.gates) == len(U2.gates)
        for i in range(len(mapped.gates)):
            for k, v in mapped.gates[i].__dict__.items():
                assert U2.gates[i].__dict__[k] == v

    for gate in [Trotterized]:
        U1 = gate(generators=[paulis.X(0) + paulis.Y(0)*paulis.Z(1), paulis.Z(3)*paulis.Z(4)], angles=["a", "b"], control=2, steps=1)
        U2 = gate(generators=[paulis.X(1) + paulis.Y(1)*paulis.Z(2), paulis.Z(4)*paulis.Z(5)], angles=["a", "b"], control=0, steps=1)
        mapped = U1.map_qubits(qubit_map={0:1, 1:2, 3:4, 4:5, 2:0})
        assert len(mapped.gates) == len(U2.gates)
        for i in range(len(mapped.gates)):
            for k, v in mapped.gates[i].__dict__.items():
                assert U2.gates[i].__dict__[k] == v

def test_conventions():
    qubit = numpy.random.randint(0, 3)
    angle = Variable("angle")

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


def strip_sympy_zeros(wfn: QubitWaveFunction):
    result = QubitWaveFunction()
    for k, v in wfn.items():
        if v != 0:
            result[k] = v
    return result


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
        wfn = simulate(g, backend="symbolic", variables={angle: sympy.pi})
        assert wfn.isclose(strip_sympy_zeros(results[i]))


def test_consistency():
    angle = numpy.pi / 2
    cpairs = [
        (CNOT(target=0, control=1), X(target=0, control=1)),
        (Ry(target=0, angle=numpy.pi), Rz(target=0, angle=4 * numpy.pi) + X(target=0)),
        (Rz(target=0, angle=numpy.pi), Rz(target=0, angle=numpy.pi) + Z(target=0)),
        (Rz(target=0, angle=angle), Rz(target=0, angle=angle / 2) + Rz(target=0, angle=angle / 2)),
        (Rx(target=0, angle=angle), Rx(target=0, angle=angle / 2) + Rx(target=0, angle=angle / 2)),
        (Ry(target=0, angle=angle), Ry(target=0, angle=angle / 2) + Ry(target=0, angle=angle / 2))
    ]

    for c in cpairs:
        print("circuit=", c[0], "\n", c[1])
        wfn1 = simulate(c[0], backend="symbolic")
        wfn2 = simulate(c[1], backend="symbolic")
        assert (numpy.isclose(wfn1.inner(wfn2), 1.0))


def test_moments():
    c = QCircuit()
    c += CNOT(target=0, control=(1, 2, 3))
    c += H(target=[0, 1])
    c += Rx(angle=numpy.pi, target=[0, 3])
    c += Z(target=1)
    c += Phase(phi=numpy.pi, target=4)
    moms = c.moments
    assert len(moms) == 3
    assert (moms[0].gates[1].parameter == assign_variable(numpy.pi))
    assert (moms[0].gates[1].target == (4,))


def test_canonical_moments():
    c = QCircuit()
    c += CNOT(target=0, control=(1, 2, 3))
    c += Rx(angle=Variable('a'), target=[0, 3])
    c += H(target=[0, 1])
    c += Rx(angle=Variable('a'), target=[2, 3])
    c += Rx(angle=Variable('a'), target=[0, 3])
    c += Z(target=1)
    c += Phase(phi=numpy.pi, target=4)
    moms = c.canonical_moments
    assert len(moms) == 6
    assert (moms[0].gates[1].parameter == assign_variable(numpy.pi))
    assert (moms[0].gates[1].target == (4,))
    assert hasattr(moms[3].gates[0], 'axis')
    assert len(moms[0].qubits) == 5


def test_circuit_from_moments():
    c = QCircuit()
    c += CNOT(target=0, control=(1, 2, 3))
    c += Phase(phi=numpy.pi, target=4)
    c += Rx(angle=Variable('a'), target=[0, 3])
    c += H(target=[0, 1])
    c += Rx(angle=Variable('a'), target=[2, 3])
    ## table[1] should equal 1 at this point, len(moments should be 3)
    c += Z(target=1)
    c += Rx(angle=Variable('a'), target=[0, 3])
    moms = c.moments
    c2 = QCircuit.from_moments(moms)
    assert c == c2


@pytest.mark.parametrize("ctrl", [None, 1])
def test_unitary_gate_u1(ctrl):
    """
    Since Z = u1(\\pi)
    """
    c_u1 = u1(lambd=numpy.pi, target=0, control=ctrl)
    c_z = Z(target=0, control=ctrl)

    wfn1 = simulate(c_u1, backend="symbolic")
    wfn2 = simulate(c_z, backend="symbolic")

    assert (numpy.isclose(abs(wfn1.inner(wfn2)), 1.0))


@pytest.mark.parametrize("ctrl", [None, 1])
def test_unitary_gate_u2(ctrl):
    """
    Since H = u2(0, \\pi)
    """
    c_u2 = u2(phi=0, lambd=numpy.pi, target=0, control=ctrl)
    c_h = H(target=0, control=ctrl)

    wfn1 = simulate(c_u2, backend="symbolic")
    wfn2 = simulate(c_h, backend="symbolic")

    assert (numpy.isclose(abs(wfn1.inner(wfn2)), 1.0))


@pytest.mark.parametrize("ctrl", [None, 1])
def test_unitary_gate_u3(ctrl):
    """
    Since X = u3(\\pi, 0, \\pi)
    """
    c_u3 = u3(theta=numpy.pi, phi=0, lambd=numpy.pi, target=0, control=ctrl)
    c_x = X(target=0, control=ctrl)

    wfn1 = simulate(c_u3, backend="symbolic")
    wfn2 = simulate(c_x, backend="symbolic")

    assert (numpy.isclose(abs(wfn1.inner(wfn2)), 1.0))


@pytest.mark.parametrize("ctrl", [None, 1])
def test_unitary_gate_u(ctrl):
    """
    Since Rz(\\lambda) = U(0, 0, \\lambda)
    """
    c_u = U(theta=0, phi=0, lambd=numpy.pi/3, target=0, control=ctrl)
    c_rz = Rz(angle=numpy.pi/3, target=0, control=ctrl)

    wfn1 = simulate(c_u, backend="symbolic")
    wfn2 = simulate(c_rz, backend="symbolic")

    assert (numpy.isclose(abs(wfn1.inner(wfn2)), 1.0))

