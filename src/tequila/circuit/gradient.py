from tequila.circuit import QCircuit
from tequila.circuit.compiler import compile_controlled_rotation
from tequila.circuit._gates_impl import ParametrizedGateImpl, RotationGateImpl
from tequila.circuit.compiler import compile_trotterized_gate
from tequila.objective import Objective, ExpectationValue
from tequila import TequilaException
from tequila.circuit.variable import Variable, Transform, has_variable

import numpy as np
from jax import numpy as jnp
import copy
import operator

import jax


def __grad_transform(par, variable):
    '''
    function for getting and evaluating partial derivatives of transforn and variable objects with respect to the aforementioned.
    :param par: a transform or variable object, to be differentiated
    :param variable: the Variable with respect to which par should be differentiated.
    :ivar var: the string representation of variable
    :ivar expan: the list of terms in the expansion of the derivative of a transform
    '''
    if type(variable) is Variable:
        var = variable.name
    if type(variable) is str:
        var = variable

    if type(par) is Variable:
        if par.name == var:
            return 1.0
        else:
            return 0.0

    elif (type(par) is Transform or (hasattr(par, 'args') and hasattr(par, 'f'))):
        t = par
        la = len(t.args)
        expan = np.zeros(la)

        if t.has_var(var):
            for i in range(la):
                if has_variable(t.args[i], var):
                    floats = [complex(arg).real for arg in t.args]
                    #expan[i] = jax.jit(jax.grad(t.transformation, argnums=i))(*floats) * __weight_chain(t.args[i], var)
                    expan[i] = jax.jit(jax.grad(t.transformation, argnums=i))(*floats)
                else:
                    expan[i] = 0.0

        return np.sum(expan)

    else:
        s = 'Object of type {} passed to grad_transform; only strings and Variables are allowed.'.format(
            str(type(par)))
        raise TequilaException(s)



def grad(obj, variable: str = None, no_compile=False):
    '''
    wrapper function for getting the gradients of Objectives,ExpectationValues, Unitaries (including single gates), and Transforms.
    :param obj (QCircuit,ParametrizedGateImpl,Objective,ExpectationValue,Transform,Variable): structure to be differentiated
    :param variables (list of Variable): parameter with respect to which obj should be differentiated.
        default None: total gradient.
    return: dictionary of Objectives, if called on gate, circuit, exp.value, or objective; if Variable or Transform, returns number.
    '''
    if isinstance(obj,Variable) or isinstance(obj,Transform):
        return __grad_transform(obj,variable)
    if not no_compile:
        compiled = compile_trotterized_gate(gate=obj)
        compiled = compile_controlled_rotation(gate=compiled)
    else:
        compiled = obj
    if type(compiled) is QCircuit:
        compiled=ExpectationValue(U=compiled)
    if variable is None:
        variable = compiled.extract_variables()
    if not isinstance(variable, str):
        return {v : grad(obj=compiled, variable=v, no_compile=True) for v in variable}

    if isinstance(compiled, ExpectationValue):
        return __grad_expectationvalue(E=compiled, variable=variable)
    elif hasattr(compiled, "is_expectationvalue") and obj.is_expectationvalue():
        return __grad_expectationvalue(E=compiled.expectationvalues[-1], variable=variable)
    elif isinstance(compiled, Objective):
        return __grad_objective(objective=compiled, variable=variable)
    else:
        raise TequilaException("Gradient not implemented for other types than ExpectationValue and Objective.")


def __grad_objective(objective: Objective, variable: str = None):
    expectationvalues = objective.expectationvalues
    transformation = objective.transformation
    dO = None
    for i, E in enumerate(expectationvalues):
        if variable not in E.extract_variables():
            continue
        df = jax.jit(jax.grad(transformation, argnums=i))
        outer = Objective(expectationvalues=expectationvalues, transformation=df)
        inner = grad(E, variable=variable)
        if dO is None:
            dO = outer * inner
        else:
            dO = dO + outer * inner
    return dO


def __grad_expectationvalue(E: ExpectationValue, variable: str):
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
        if g.is_parametrized() and not g.is_frozen():
            if has_variable(g.parameter,variable):
                if hasattr(g, 'angle'):
                    if g.is_controlled():
                        dOinc =__grad_controlled_rotation(unitary,g,i,variable,hamiltonian)
                        if dO is None:
                            dO = dOinc
                        else:
                            dO = dO + dOinc
                    else:
                        dOinc  = __grad_rotation(unitary,g,i,variable,hamiltonian)
                        if dO is None:
                            dO = dOinc
                        else:
                            dO = dO + dOinc

                elif hasattr(g, 'power'):
                    if g.is_controlled():
                        raise NotImplementedError("Gradient for controlled PowerGate not here yet")
                    else:
                        if g.name in ['H', 'Hadamard']:
                            raise TequilaException('sorry, cannot figure out hadamard gradients yet')
                        else:
                            dOinc=__grad_power(unitary,g,i,variable,hamiltonian)
                            if dO is None:
                                dO = dOinc
                            else:
                                dO = dO + dOinc

                else:
                    print(type(g))
                    raise TequilaException("Automatic differentiation is implemented only for Rotational Gates")
    return dO

def __grad_controlled_rotation(unitary,g,i,variable,hamiltonian):
    '''
    function for getting the gradient of a controlled rotation gate. Should only be called on such objects.
    :param unitary: QCircuit: the QCircuit object containing the gate to be differentiated
    :param g: ParametrizedGateImpl: the gate being differentiated
    :param i: Int: the position in unitary at which g appears
    :param variable: Variable or String: the variable with respect to which gate g is being differentiated
    :param hamiltonian: the hamiltonian with respect to which unitary is to be measured, in the case that unitary
        is contained within an ExpectationValue
    :return: an Objective, whose calculation yields the gradient of g w.r.t variable
    '''
    angles_and_weights = [
        ([(g.angle / 2) + np.pi / 2, -g.angle / 2], .50),
        ([(g.angle) / 2 - np.pi / 2, -g.angle / 2], -.50),
        ([g.angle / 2, -(g.angle / 2) + np.pi / 2], -.50),
        ([g.angle / 2, -(g.angle / 2) - np.pi / 2], .50)
    ]
    dO=None
    for ang_set in angles_and_weights:
        U = unitary.replace_gate(position=i, gates=[gate for gate in compile_controlled_rotation(g, angles=ang_set[0])])
        w = 0.5 * ang_set[1] * __grad_transform(g.parameter, variable)
        ev=ExpectationValue(U=U,H=hamiltonian)
        dOinc=w *ev
        if dO is None:
            dO = dOinc
        else:
            dO = dO + dOinc

    return dO

def __grad_rotation(unitary,g,i,variable,hamiltonian):
    '''
    function for getting the gradients of UNCONTROLLED rotation gates.
    :param unitary: QCircuit: the QCircuit object containing the gate to be differentiated
    :param g: ParametrizedGateImpl: the gate being differentiated
    :param i: Int: the position in unitary at which g appears
    :param variable: Variable or String: the variable with respect to which gate g is being differentiated
    :param hamiltonian: the hamiltonian with respect to which unitary is to be measured, in the case that unitary
        is contained within an ExpectationValue
    :return: an Objective, whose calculation yields the gradient of g w.r.t variable
    '''

    neo_a = copy.deepcopy(g)
    neo_a.frozen = True

    neo_a.angle = g.angle + np.pi / 2
    U1 = unitary.replace_gate(position=i, gates=[neo_a])
    w1 = 0.5 * __grad_transform(g.parameter, variable)

    neo_b = copy.deepcopy(g)
    neo_b.frozen = True
    neo_b.angle = g.angle - np.pi / 2
    U2 = unitary.replace_gate(position=i, gates=[neo_b])
    w2 = -0.5 * __grad_transform(g.parameter, variable)

    Oplus = ExpectationValue(U=U1, H=hamiltonian)
    Ominus = ExpectationValue(U=U2, H=hamiltonian)
    dOinc = w1 * Objective(expectationvalues=[Oplus]) + w2 * Objective(expectationvalues=[Ominus])
    return dOinc

def __grad_power(unitary,g,i,variable,hamiltonian):
    '''
    function for getting the gradient of Power gates. note: doesn't yet work on Hadamard
    :param unitary: QCircuit: the QCircuit object containing the gate to be differentiated
    :param g: ParametrizedGateImpl: the gate being differentiated
    :param i: Int: the position in unitary at which g appears
    :param variable: Variable or String: the variable with respect to which gate g is being differentiated
    :param hamiltonian: the hamiltonian with respect to which unitary is to be measured, in the case that unitary
        is contained within an ExpectationValue
    :return: an Objective, whose calculation yields the gradient of g w.r.t variable
    '''
    target = g.target
    if g.name in ['H', 'Hadamard']:
        raise TequilaException('sorry, cannot figure out hadamard gradients yet')
    else:
        n_pow = g.parameter * np.pi
        if g.name in ['X', 'x']:
            axis = 0
        elif g.name in ['Y', 'y']:
            axis = 1
        elif g.name in ['Z', 'z']:
            axis = 2
        else:
            raise NotImplementedError(
                'sorry, I have no idea what this gate is and cannot build the gradient.')
        U1 = unitary.replace_gate(position=i, gates=[
            RotationGateImpl(axis=axis, target=target, angle=(n_pow + np.pi / 2), frozen=True)])
        U2 = unitary.replace_gate(position=i, gates=[
            RotationGateImpl(axis=axis, target=target, angle=(n_pow - np.pi / 2), frozen=True)])

        w1 = 0.5 * __grad_transform(g.parameter, variable)
        w2 = -0.5 * __grad_transform(g.parameter, variable)
        Oplus = ExpectationValue(U=U1, H=hamiltonian)
        Ominus = ExpectationValue(U=U2, H=hamiltonian)
        dOinc = w1 * Objective(expectationvalues=[Oplus]) + w2 * Objective(expectationvalues=[Ominus])
        return dOinc