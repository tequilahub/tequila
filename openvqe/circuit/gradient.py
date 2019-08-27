from openvqe.circuit import QCircuit
from openvqe.circuit.compiler import compile_controlled_rotation_gate
from openvqe.circuit._gates_impl import ParametrizedGateImpl, RotationGateImpl, PowerGateImpl
from openvqe.objective import Objective
from openvqe import OpenVQEException
import copy
from numpy import pi

def grad(obj):
    if isinstance(obj, QCircuit):
        return grad_unitary(unitary=obj)
    elif isinstance(obj, Objective):
        return grad_objective(objective=obj)
    elif isinstance(obj, ParametrizedGateImpl):
        return grad_unitary(QCircuit.wrap_gate(gate=obj))
    else:
        raise OpenVQEException("Gradient not implemented for other types than QCircuit or Objective")

def grad_unitary(unitary:QCircuit):
    gradient=[]
    angles = unitary.extract_parameters()
    for i in range(len(angles)):
        index = angles[i][0]
        gradient.append(make_gradient_component(unitary=unitary, index=index))
    return gradient

def grad_objective(objective: Objective):
    assert(len(objective.unitaries==1))
    return grad_unitary(unitary=objective.unitaries[0])


def make_gradient_component(unitary: QCircuit, index:int):
    """
    :param unitary: the unitary
    :param index: position of gate in circuit (should be changed)
    :return: dU/dpi
    """

    pre = QCircuit()
    post = QCircuit()
    for i, g in enumerate(unitary.gates):
        if i < index:
            pre += QCircuit.wrap_gate(g)
        elif i > index:
            post += QCircuit.wrap_gate(g)

    g = unitary.gates[index]
    dg = []
    if isinstance(g, RotationGateImpl):
        if g.is_controlled():

            angles = [
                [(-g.angle + pi / 2) / 2, g.angle / 2],
                [(-g.angle - pi / 2) / 2, g.angle / 2],
                [-g.angle / 2, (g.angle + pi / 2) / 2],
                [-g.angle / 2, (g.angle - pi / 2) / 2]
            ]

            for i, angle_set in enumerate(angles):
                parity = 1.0 - 2.0 * (i // 2)
                U = compile_controlled_rotation_gate(g, angles=angle_set)
                U.weight = 0.5*parity
                dg.append(U)
        else:
            neo_a = copy.deepcopy(g)
            neo_a.angle = g.angle + pi / 2
            U1 = QCircuit.wrap_gate(neo_a)
            U1.weight = 0.5
            neo_b = copy.deepcopy(g)
            neo_b.angle = g.angle - pi / 2
            U2 = QCircuit.wrap_gate(neo_b)
            U2.weight = -0.5
            dg = [U1, U2]
    elif isinstance(g, PowerGateImpl):
        if g.is_controlled():
            raise NotImplementedError("Gradient for controlled PowerGate not here yet")
        else:
            new_gate = copy.deepcopy(g)
            new_gate.power = g.power - 1.0
            U = QCircuit.wrap_gate(new_gate)
            U.weight = g.power
            dg = [U]
    else:
        raise OpenVQEException("Automatic differentiation only for Rotational and PowerGates")

    # assemble
    unitaries = []
    for U in dg:
        unitaries.append(pre + U + post)

    return Objective(unitaries=unitaries)
