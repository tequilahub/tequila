from openvqe.circuit import QCircuit
from openvqe.circuit.gates import Ry, X, Rx, Rz
from openvqe.observable import Observable, make_gradient
from numpy import pi

if __name__ == "__main__":
    ac = QCircuit()
    ac += X(0)
    ac += Ry(target=1, control=None, angle=pi / 2)

    observable = Observable()
    observable.unitary = [ac]
    gradient = make_gradient(observable=observable)

    print("gradient=", gradient)

    ac = QCircuit()
    ac += X(0)
    ac += Ry(target=1, control=0, angle=pi / 2)

    observable = Observable()
    observable.unitary=[ac]
    gradient = make_gradient(observable=observable)

    print("gradient=", gradient)

    ac = QCircuit()
    ac += X(0)
    ac += Rx(target=2, angle=0.5)
    ac += Ry(target=1, control=0, angle=pi / 2)
    ac += Rz(target=1, control=[0,2], angle=pi / 2)

    observable = Observable()
    observable.unitary=[ac]
    gradient = make_gradient(observable=observable)

    print("gradient=", gradient)