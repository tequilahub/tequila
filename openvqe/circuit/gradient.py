from openvqe.circuit import QCircuit
from openvqe.circuit.compiler import compile_controlled_rotation_gate
from openvqe.circuit._gates_impl import ParametrizedGateImpl, RotationGateImpl, PowerGateImpl
from openvqe.objective import Objective
from openvqe import OpenVQEException
import copy
from numpy import pi
import numpy as np

def grad(obj):
    if isinstance(obj, QCircuit):
        return grad_unitary(unitary=obj)
    elif isinstance(obj, Objective):
        return grad_objective(objective=obj)
    elif isinstance(obj, ParametrizedGateImpl):
        return grad_unitary(QCircuit.wrap_gate(gate=obj))
    else:
        raise OpenVQEException("Gradient not implemented for other types than QCircuit or Objective")


def grad_unitary(unitary: QCircuit):
    gradient = []
    angles = unitary.extract_parameters()
    for i in range(len(angles)):
        index = angles[i][0]
        gradient.append(make_gradient_component(unitary=unitary, index=index))
    return gradient


def grad_objective(objective: Objective):
    if len(objective.unitaries) > 1:
        raise OpenVQEException("Gradient of Objectives with more than one unitary not supported yet")
    result = grad_unitary(unitary=objective.unitaries[0])
    for i, r in enumerate(result):
        result[i].observable = objective.observable
    return result


def make_gradient_component(unitary: QCircuit, index: int):
    """
    :param unitary: the unitary
    :param index: position of gate in circuit (should be changed)
    :return: dU/dpi
    """

    pre = QCircuit()
    post = QCircuit()
    for i, g in enumerate(unitary.gates):
        if i < index:
            pre *= QCircuit.wrap_gate(g)
        elif i > index:
            post *= QCircuit.wrap_gate(g)

    g = unitary.gates[index]
    dg = []
    if isinstance(g, RotationGateImpl):
        if g.is_controlled():
            angles_and_weights = [
                ([-(g.angle) / 2 + pi / 2, g.angle / 2],.50),
                ([-(g.angle ) / 2 - pi / 2, g.angle / 2],-.50),
                ([-g.angle / 2, (g.angle) / 2  + pi / 2],-.50),
                ([-g.angle / 2, (g.angle ) / 2 - pi / 2],.50)
            ]

            for i, ang_set in enumerate(angles_and_weights):

                U = compile_controlled_rotation_gate(g, angles=ang_set[0])
                U.weight=0.5*ang_set[1]
                dg.append(U)
        else:
            neo_a = copy.deepcopy(g)
            neo_a.angle = g.angle + pi/2
            U1 = QCircuit.wrap_gate(neo_a)
            U1.weight = 0.5
            neo_b = copy.deepcopy(g)
            neo_b.angle = g.angle - pi/2
            U2 = QCircuit.wrap_gate(neo_b)
            U2.weight = -0.5
            dg = [U1, U2]
    elif isinstance(g, PowerGateImpl):
        if g.is_controlled():
            raise NotImplementedError("Gradient for controlled PowerGate not here yet")
        else:
            new=copy.deepcopy(g)
            new.power-=1.0
            U=QCircuit.wrap_gate(new)
            U.weight=g.power
            dg=[U]

            n_pow = pi*(g.power%2)
            target=g.target
            phase=g.phase
            ### does that need to be divided by two?
            ### trying to convert gates to rotations for quadrature
            if g.name in ['H','Hadamard']:
                raise NotImplementedError('Hi, sorry, I do not know how to do Hadamard gradients yet.')
                '''
                (Ry(pi)Rz(pi)Ry(pi))^t is the decomp. not sure where to go from here
                '''
            else:
                if g.name in ['X','x']:
                    axis=0
                elif g.name in ['Y','y']:
                    axis=1
                elif g.name in ['Z','z']:
                    axis=2
                else:
                    raise NotImplementedError('sorry, I have no idea what this gate is and cannot build the gradient.')
                U1 = QCircuit.wrap_gate(RotationGateImpl(axis=axis,target=target,angle=(n_pow+pi/2),phase=phase*np.exp(1j*n_pow/2)))
                U2 = QCircuit.wrap_gate(RotationGateImpl(axis=axis,target=target,angle=(n_pow-pi/2),phase=phase*np.exp(1j*n_pow/2)))
                U1.weight=0.5
                U2.weight=-0.5
                dg=[U1,U2]
                
    else:
        raise OpenVQEException("Automatic differentiation is implemented only for Rotational and Power Gates")

    # assemble
    unitaries = []
    for U in dg:
        unitaries.append(pre * U * post)

    return Objective(unitaries=unitaries)
