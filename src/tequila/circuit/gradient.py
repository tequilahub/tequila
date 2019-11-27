from tequila.circuit import QCircuit
from tequila.circuit.compiler import compile_controlled_rotation
from tequila.circuit._gates_impl import ParametrizedGateImpl, RotationGateImpl
from tequila.circuit.compiler import compile_trotterized_gate
from tequila.objective import Objective
from tequila import TequilaException

import numpy as np
import copy
from tequila.circuit.variable import Variable,Transform,has_variable
import operator

def __weight_chain(par,variable):
    '''
    recursive function for getting and evaluating partial derivatives of transforn and variable objects with respect to the aforementioned.
    :param par: a transform or variable object, to be differentiated
    :param variable: the Variable with respect to which par should be differentiated.

    :ivar var: the string representation of variable
    :ivar expan: the list of terms in the expansion of the derivative of a transform
    '''
    if type(variable) is Variable:
        var=variable.name
    if type(variable) is str:
        var=variable

    if type(par) is Variable:
        if par.name == var:
            return 1.0
        else:
            return 0.0

    elif (type(par) is Transform or (hasattr(par,'args') and hasattr(par,'f'))):
        t=par
        la=len(t.args)
        expan=np.zeros(la)

        if t.has_var(var):
            for i in range(la):
                if has_variable(t.args[i],var):
                    floats=[complex(arg).real for arg in t.args]
                    expan[i]=tgrad(t.f,argnum=i)(*floats)*__weight_chain(t.args[i],var)
                else:
                    expan[i]=0.0


        return np.sum(expan)
        
    else:
        s='Object of type {} passed to weight_chain; only strings, Variables and Transforms are allowed.'.format(str(type(par)))
        raise TequilaException(s)


def tgrad(f,argnum):
    '''
    function to be replaced entirely by the use of jax.grad(); completely identical thereto but restricted to our toy functions.
    :param f: a function (must be one of those defined in circuit.variable) to be differentiated.
    :param argnum (int): which of the arguments of the function with respect toward which it should be differentiated.
    :returns lambda function representing the partial of f with respect to the argument numberd by argnum
    '''
    assert callable(f)

    if argnum == 0:
        if f.op == operator.add:
            return lambda x,y: 1.0

        elif f.op  == operator.mul:
            return lambda x,y: float(y)

        elif f.op == operator.sub:
            return lambda x,y: 1.0

        elif f.op == operator.truediv:
            return lambda x,y: 1/float(y)

        elif f.op  == operator.pow:
            return lambda x,y: float(y)*float(x)**(float(y)-1)

        else:
            raise TequilaException('Sorry, only pre-built tequila functions supported for tgrad at the moment.')


    elif argnum ==1:

        if f.op == operator.add:
            return lambda x,y: 1.0

        elif f.op == operator.mul:
            return lambda x,y: float(x)

        elif f.op == operator.sub:
            return lambda x,y: -1.0

        elif f.op == operator.truediv:
            return lambda x,y: -float(x)/(float(y)**2)

        elif f.op == operator.pow:
            return lambda x,y: (float(x)**float(y))*np.log(float(x))

        else:
            raise TequilaException('Sorry, only pre-built tequila functions supported for tgrad at the moment.')





def grad(obj, variables=None):
    '''
    wrapper function for getting the gradients of Objectives or Unitaries (including single gates).
    :param obj (QCircuit,ParametrizedGateImpl,Objective): structure to be differentiated
    :param variables (list of Variable): parameters with respect to which obj should be differentiated.
        default None: total gradient.
    return: dictionary of Objectives
    '''
    compiled = compile_trotterized_gate(gate=obj)
    compiled = compile_controlled_rotation(gate=compiled)


    if isinstance(obj, QCircuit):
        return __grad_unitary(unitary=compiled, variables=variables)
    elif isinstance(obj, Objective):
        return __grad_objective(objective=compiled, variables=variables)
    elif isinstance(obj, ParametrizedGateImpl):
        return __grad_unitary(QCircuit.wrap_gate(gate=compiled), variables=variables)
    else:
        raise TequilaException("Gradient not implemented for other types than QCircuit,Objective.")




def __grad_unitary(unitary: QCircuit,variables=None):
    '''
    wrapper function for getting the gradients of Unitaries (including single gates).
    :param unitary (QCircuit): circuit to be differentiated
    :param variables (list of Variable): parameters with respect to which obj should be differentiated.
        default None: total gradient of the unitary.

    returns: dictionary, entries of form {parameter name: Objective}
    '''
    unitary.validate()
    if variables is None:
        out= __make_selected_components(unitary, unitary.extract_variables())
    elif type(variables) is dict:
        out= __make_selected_components(unitary,variables)
    elif type(variables) is list:
        vs={}
        for var in variables:
            if hasattr(var,'name'):
                vs[var.name]=[]
            else:
                vs[str(k)]=[]
        out= __make_selected_components(unitary,vs)

    else:
        out= None
    return out


def __grad_objective(objective: Objective, variables=None):
    '''
    wrapper function for getting the gradients of Objectives; primarily a wrapper over __grad_unitary.
    :param obj (Objective): structure to be differentiated
    :param variables (list of Variable): parameters with respect to which obj should be differentiated.
        default None: total gradient.

    return dictionary of Objectives.
    '''

    if len(objective.unitaries)>1:
        raise TequilaException("Gradient of Objectives with more than one unitary not supported yet")
    result = __grad_unitary(unitary=objective.unitaries[0], variables=variables)
    for k in result.keys():
        result[k].observable=objective.observable
    return result


def __make_selected_components(unitary: QCircuit,variables):
    '''
    implements the analytic partial derivative of a unitary as it would appear in an expectation value. See the paper.
    :param unitary: the unitary whose gradient should be obtained
    :param variables (list, dict, str): the variables with respect to which differentiation should be performed.
    :return: vector (as dict) of dU/dpi as Objective (without hamiltonian)
    '''
    
    pi=np.pi
    if type(variables) is dict:
        out={k:[] for k in variables.keys()}
    elif type(variables) is list:
        out={str(k):[] for k in variables}
    for i,g in enumerate(unitary.gates):
        if g.is_parametrized() and not g.is_frozen():
            names=g.parameter.variables.keys()
            for k in names:
                if hasattr(g,'angle'):
                    if g.is_controlled():
                        angles_and_weights = [
                                ([(g.angle / 2) + pi / 2, -g.angle / 2],.50),
                                ([(g.angle ) / 2 - pi / 2, -g.angle / 2],-.50),
                                ([g.angle / 2, -(g.angle / 2)  + pi / 2],-.50),
                                ([g.angle / 2, -(g.angle / 2) - pi / 2],.50)
                            ]
                        for ang_set in angles_and_weights:

                            U = unitary.replace_gate(position=i,gates=[gate for gate in compile_controlled_rotation(g, angles=ang_set[0])])
                            U.weight=0.5*ang_set[1]*__weight_chain(g.parameter,k)
                            out[k].append(U)
                    else:
                        neo_a = copy.deepcopy(g)
                        neo_a.frozen=True

                        neo_a.angle = g.angle + pi/2
                        U1 = unitary.replace_gate(position=i,gates=[neo_a])
                        U1.weight = 0.5*__weight_chain(g.parameter,k)

                        neo_b = copy.deepcopy(g)
                        neo_b.frozen=True
                        neo_b.angle = g.angle - pi/2
                        U2=unitary.replace_gate(position=i,gates=[neo_b])
                        U2.weight = -0.5*__weight_chain(g.parameter,k)
                        out[k].append(U1)
                        out[k].append(U2)
                elif hasattr(g,'power'):
                        
                        if g.is_controlled():
                            raise NotImplementedError("Gradient for controlled PowerGate not here yet")
                        else:
                            n_pow = g.parameter*pi/4
                            target=g.target
                            ### does that need to be divided by two?
                            ### trying to convert gates to rotations for quadrature
                            if g.name in ['H','Hadamard']:
                                raise TequilaException('sorry, cannot figure out hadamard gradients yet')
                                '''
                                U1 = unitary.replace_gate(position=i,gates=[RotationGateImpl(axis=1,target=target,angle=(n_pow+pi/2),frozen=True)])
                                U2 = unitary.replace_gate(position=i,gates=[RotationGateImpl(axis=1,target=target,angle=(n_pow-pi/2),frozen=True)])
                                U1.weight=0.5*weight_chain(g.parameter,var)
                                U2.weight=-0.5*weight_chain(g.parameter,var)
                                dg.extend([U1,U2])
     `                          '''
                            else:
                                n_pow = g.parameter*pi
                                if g.name in ['X','x']:
                                    axis=0
                                elif g.name in ['Y','y']:
                                    axis=1
                                elif g.name in ['Z','z']:
                                    axis=2
                                else:
                                    raise NotImplementedError('sorry, I have no idea what this gate is and cannot build the gradient.')
                                U1 = unitary.replace_gate(position=i,gates=[RotationGateImpl(axis=axis,target=target,angle=(n_pow+pi/2),frozen=True)])
                                U2 = unitary.replace_gate(position=i,gates=[RotationGateImpl(axis=axis,target=target,angle=(n_pow-pi/2),frozen=True)])

                                U1.weight=0.5*__weight_chain(g.parameter,k)
                                U2.weight=-0.5*__weight_chain(g.parameter,k)
                                out[k].extend([U1,U2])
                        
                else:
                    print(type(g))
                    raise TequilaException("Automatic differentiation is implemented only for Rotational Gates")
    new_dict={}
    for k in out.keys():
        new_dict[k]=Objective(unitaries=out[k])
    return new_dict
