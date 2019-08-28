
from openvqe.circuit import *
from openvqe.objective import Objective
from openvqe.circuit.gradient import grad
from openvqe.circuit.gates import X, Ry, Rx, Rz
from openvqe.circuit._gates_impl import RotationGateImpl


from numpy import pi

if __name__ == "__main__":
    gate = Ry(target=1, control=3, angle=pi / 3, phase=1.0, frozen=False)

    gradient = grad(gate)
    print("gradient at of Ry at 'gate' level", gradient)

    #######
    # the following should not be done .... but would work anyway
    #######
    gate = RotationGateImpl(axis=1, angle=pi / 3, target=1, control=3)
    gradientx = grad(gate)
    print("gradient at of Ry at gate level", gradientx, " test:", gradient[0] == gradientx[0])

    ac = QCircuit()
    ac *= X(target=0, control=None)
    ac *= Ry(target=1, control=None, angle=pi / 2)

    obj = Objective(unitaries=ac, observable=None)
    gradient = grad(obj.unitaries[0])

    print("gradient of X, Ry at objective level", gradient)

    ac = QCircuit()
    ac *= X(target=0, power=2.3, phase=-1.0)
    ac *= Ry(target=1, control=0, angle=pi / 2)

    print('gradient of Xpower, controlled Ry at circuit level:', grad(ac))
    obj = Objective(unitaries=ac, observable=None)
    gradient = grad(obj.unitaries[0])

    print("gradient of Xpower controlled Ry at objective level", gradient)

    ac = QCircuit()
    ac *= X(0)
    ac *= Rx(target=2, angle=0.5, frozen=True)
    ac *= Ry(target=1, control=0, angle=pi / 2)
    ac *= Rz(target=1, control=[0, 2], angle=pi / 2)

    obj = Objective(unitaries=ac, observable=None)
    gradient = grad(obj.unitaries[0])

    print("gradient at objective level", gradient)

    from openvqe.circuit.gradient import grad

    print("new impl", grad(ac))
