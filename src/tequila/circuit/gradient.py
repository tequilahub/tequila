from tequila.circuit.compiler import compile_controlled_rotation
from tequila.circuit._gates_impl import RotationGateImpl,PhaseGateImpl
from tequila.circuit.compiler import compile_trotterized_gate, compile_exponential_pauli_gate, compile_multitarget,compile_power_gate,compile_controlled_phase,compile_h_power
from tequila.objective.objective import Objective, ExpectationValueImpl, Variable
from tequila import TequilaException

import numpy as np
import copy
import typing

import jax


def grad(objective: Objective, variable: Variable = None, no_compile=False):
    '''
    wrapper function for getting the gradients of Objectives,ExpectationValues, Unitaries (including single gates), and Transforms.
    :param obj (QCircuit,ParametrizedGateImpl,Objective,ExpectationValue,Transform,Variable): structure to be differentiated
    :param variables (list of Variable): parameter with respect to which obj should be differentiated.
        default None: total gradient.
    return: dictionary of Objectives, if called on gate, circuit, exp.value, or objective; if Variable or Transform, returns number.
    '''

    if not no_compile:
        compiled = compile_multitarget(gate=objective)
        compiled = compile_trotterized_gate(gate=compiled)
        compiled = compile_exponential_pauli_gate(gate=compiled)
        compiled = compile_h_power(gate=compiled)
        compiled = compile_power_gate(gate=compiled)
        compiled = compile_controlled_phase(gate=compiled)
        compiled = compile_controlled_rotation(gate=compiled)
    else:
        compiled = objective

    if variable is None:
        # None means that all components are created
        variables = compiled.extract_variables()
        result = {}

        if len(variables) == 0:
            raise TequilaException("Error in gradient: Objective has no variables")

        for k in variables:
            assert (k is not None)
            result[k] = grad(compiled, k)
        return result
    elif not isinstance(variable, Variable) and hasattr(variable, "__hash__"):
        variable = Variable(name=variable)

    if variable not in compiled.extract_variables():
        raise TequilaException("Error in taking gradient. Objective does not depend on variable {} ".format(variable))

    if isinstance(objective, ExpectationValueImpl):
        return __grad_expectationvalue(E=objective, variable=variable)
    elif objective.is_expectationvalue():
        return __grad_expectationvalue(E=compiled.args[-1], variable=variable)
    elif isinstance(compiled, Objective):
        return __grad_objective(objective=compiled, variable=variable)
    else:
        raise TequilaException("Gradient not implemented for other types than ExpectationValue and Objective.")


def __grad_objective(objective: Objective, variable: Variable):
    args = objective.args
    transformation = objective.transformation
    dO = None
    for i, arg in enumerate(args):
        df = jax.jit(jax.grad(transformation, argnums=i))
        outer = Objective(args=args, transformation=df)

        inner = __grad_inner(arg=arg, variable=variable)

        if inner == 0.0:
            # don't pile up zero expectationvalues
            continue

        if dO is None:
            dO = outer * inner
        else:
            dO = dO + outer * inner
    return dO


def __grad_inner(arg, variable):
    '''
    a modified loop over __grad_objective, which gets derivatives
     all the way down to variables, return 1 or 0 when a variable is (isnt) identical to var.
    :param arg: a transform or variable object, to be differentiated
    :param variable: the Variable with respect to which par should be differentiated.
    :ivar var: the string representation of variable
    '''

    assert (isinstance(variable, Variable))
    if isinstance(arg, Variable):
        if arg == variable:
            return 1.0
        else:
            return 0.0
    elif isinstance(arg, ExpectationValueImpl):
        return __grad_expectationvalue(arg, variable=variable)
    else:
        return __grad_objective(objective=arg, variable=variable)


def __grad_expectationvalue(E: ExpectationValueImpl, variable: Variable):
    '''
    implements the analytic partial derivative of a unitary as it would appear in an expectation value. See the paper.
    :param unitary: the unitary whose gradient should be obtained
    :param variables (list, dict, str): the variables with respect to which differentiation should be performed.
    :return: vector (as dict) of dU/dpi as Objective (without hamiltonian)
    '''
    hamiltonian = E.H
    unitary = E.U
    dO = None
    for i, g in enumerate(unitary.gates):
        if g.is_parametrized():
            if variable in g.extract_variables():
                if g.is_gaussian():
                    if g.is_controlled():
                        raise TequilaException("controlled gate in gradient: Compiler was not called")
                    else:
                        dOinc = __grad_gaussian(unitary, g, i, variable, hamiltonian)
                        if dO is None:
                            dO = dOinc
                        else:
                            dO = dO + dOinc
                else:
                    print(g,type(g))
                    raise TequilaException('currently, only the gradients of Gaussian gates can be calculated. If you have set compile to False, try setting it to to True.')
    return dO


def __grad_gaussian(unitary, g, i, variable, hamiltonian):
    '''
    function for getting the gradients of gaussian gates. NOTE: you had better compile first.
    :param unitary: QCircuit: the QCircuit object containing the gate to be differentiated
    :param g: a parametrized: the gate being differentiated
    :param i: Int: the position in unitary at which g appears
    :param variable: Variable or String: the variable with respect to which gate g is being differentiated
    :param hamiltonian: the hamiltonian with respect to which unitary is to be measured, in the case that unitary
        is contained within an ExpectationValue
    :return: an Objective, whose calculation yields the gradient of g w.r.t variable
    '''

    if not g.is_gaussian():
        raise TequilaException('Only parametrized gaussian gates are differentiable; recieved gate of type ' + str(type(g)))
    shift_a = g._parameter + np.pi /(4*g.shift)
    shift_b = g._parameter - np.pi / (4 * g.shift)
    if hasattr(g,'angle'):
        neo_a = RotationGateImpl(axis=g.axis,target=g.target, control=g.control, angle=shift_a)
        neo_b = RotationGateImpl(axis=g.axis, target=g.target, control=g.control, angle=shift_b)
    elif hasattr(g,'phase'):
        neo_a = PhaseGateImpl(phase=shift_a,target=g.target,control=g.control)
        neo_b = PhaseGateImpl(phase=shift_b, target=g.target, control=g.control)

    U1 = unitary.replace_gate(position=i, gates=[neo_a])
    w1 = g.shift * __grad_inner(g.parameter, variable)



    U2 = unitary.replace_gate(position=i, gates=[neo_b])
    w2 = -g.shift * __grad_inner(g.parameter, variable)

    Oplus = ExpectationValueImpl(U=U1, H=hamiltonian)
    Ominus = ExpectationValueImpl(U=U2, H=hamiltonian)
    dOinc = w1 * Objective(args=[Oplus]) + w2 * Objective(args=[Ominus])
    return dOinc


