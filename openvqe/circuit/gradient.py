from openvqe.circuit import QCircuit
from openvqe.circuit.compiler import compile_controlled_rotation
from openvqe.circuit._gates_impl import ParametrizedGateImpl, RotationGateImpl, PowerGateImpl
from openvqe.objective import Objective
from openvqe import OpenVQEException
from openvqe import copy
from openvqe import numpy

def weight_chain(gate):
    '''
    This is actually a forward mode derivative, since we have a linear graph!
    '''
    value=gate.parameter._value
    t_weight=1.0
    for t in gate.parameter.transform:
        t_weight*=t.grad(value)
        value=t(value)

    return t_weight

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
    gradient = dict()
    angles = unitary.extract_parameters()
    for k, v in angles.items():
        indices = unitary.get_indices_for_parameter(name=k)
        gradient[k] = Objective(unitaries=[])
        for index in indices:
            gradient[k] += make_gradient_component(unitary=unitary, index=index)
    return gradient


def grad_objective(objective: Objective):
    if len(objective.unitaries) > 1:
        raise OpenVQEException("Gradient of Objectives with more than one unitary not supported yet")
    result = grad_unitary(unitary=objective.unitaries[0])
    for k, v in result.items():
        result[k].observable = objective.observable
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
                ([-(g.angle) / 2 + numpy.pi / 2, g.angle / 2], .50),
                ([-(g.angle) / 2 - numpy.pi / 2, g.angle / 2], -.50),
                ([-g.angle / 2, (g.angle) / 2 + numpy.pi / 2], -.50),
                ([-g.angle / 2, (g.angle) / 2 - numpy.pi / 2], .50)
            ]

            for i, ang_set in enumerate(angles_and_weights):
                U = compile_controlled_rotation(g, angles=ang_set[0])
                U.weight = 0.5 * ang_set[1]
                dg.append(U)
        else:
            neo_a = copy.deepcopy(g)
            neo_a.angle = g.angle + numpy.pi / 2
            U1 = QCircuit.wrap_gate(neo_a)
            U1.weight = 0.5*weight_chain(g)
            neo_b = copy.deepcopy(g)
            neo_b.angle = g.angle - numpy.pi / 2
            U2 = QCircuit.wrap_gate(neo_b)
            U2.weight = -0.5*weight_chain(g)
            dg = [U1, U2]
    else:
        raise OpenVQEException("Differentiation is implemented only for Rotational Gates")

    # assemble
    unitaries = []
    for U in dg:
        unitaries.append(pre * U * post)

    return Objective(unitaries=unitaries)
