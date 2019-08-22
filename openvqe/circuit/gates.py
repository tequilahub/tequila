"""
Predefined Gates
"""
from openvqe.circuit import QCircuit
from openvqe.circuit.circuit import QGate


# Convenience
def H(target: int, control: int = None):
    return QCircuit(gates=[QGate(name="H", target=target, control=control)])


# Convenience
def S(target: int, control: list = None):
    return QCircuit(gates=[QGate(name="S", target=target, control=control)])


# Convenience
def X(target: int, control: list = None):
    return QCircuit(gates=[QGate(name="X", target=target, control=control)])


# Convenience
def Y(target: int, control: list = None):
    return QCircuit(gates=[QGate(name="Y", target=target, control=control)])


# Convenience
def Z(target: int, control: list = None):
    return QCircuit(gates=[QGate(name="Z", target=target, control=control)])


# Convenience
def I(target: int):
    return QCircuit(gates=[QGate(name="I", target=target)])


# Convenience
def CNOT(control: int, target: int):
    return QCircuit(gates=[QGate(name="CNOT", target=target, control=control)])


# Convenience
def aCNOT(control: int, target: int):
    return QCircuit(gates=[
        QGate(name="X", target=control),
        QGate(name="CNOT", target=target, control=control),
        QGate(name="X", target=control)
    ])


# Convenience
def Rx(target: int, angle: float, control: int = None):
    return QCircuit(gates=[
        QGate(name="Rx", target=target, angle=angle, control=control)
    ])


# Convenience
def Ry(target: int, angle: float, control: int = None):
    return QCircuit(gates=[
        QGate(name="Ry", target=target, angle=angle, control=control)
    ])


# Convenience
def Rz(target: int, angle: float, control: int = None):
    return QCircuit(gates=[
        QGate(name="Rz", target=target, angle=angle, control=control)
    ])


# Convenience
def SWAP(target: list, control: list):
    return QCircuit(gates=[
        QGate(name="SWAP", target=target, control=control)
    ])


# Convenience
def TOFFOLI(target: list, control: list = None):
    return QCircuit(gates=[
        QGate(name="TOFFOLI", target=target, control=control)
    ])

if __name__ == "__main__":
    circuit = Ry(control=0, target=3, angle=numpy.pi / 2)
    circuit += CNOT(control=1, target=0)
    circuit += Ry(control=0, target=1, angle=numpy.pi / 2)
    circuit += aCNOT(control=2, target=0)
    circuit += CNOT(control=3, target=2)
    circuit += Ry(control=0, target=3, angle=numpy.pi / 2)
    circuit += X(0) + X(2)

    print(circuit)

    cr = circuit.make_dagger()