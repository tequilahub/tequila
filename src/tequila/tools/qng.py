from tequila import TequilaException
from tequila.hamiltonian import paulis
from tequila.objective.objective import Objective, ExpectationValueImpl, Variable, ExpectationValue
from tequila.circuit.circuit import QCircuit
from tequila.simulators.simulator_api import compile_objective
from tequila.circuit.gradient import __grad_inner
from tequila.autograd_imports import jax
from tequila.circuit.compiler import compile_controlled_rotation,compile_h_power,compile_power_gate, \
    compile_trotterized_gate,compile_controlled_phase, compile_multitarget

import numpy
import copy

class QngMatrix:

    @property
    def dim(self):
        d=0
        for block in self.blocks:
            d+=len(block)
        return (d,d)

    def __init__(self,blocks):
        self.blocks= blocks

    def __call__(self, variables,samples=None):
        output= numpy.zeros(self.dim)
        d_v = 0
        ### blocks are square, so if we are using a block diagonal approx, we need to
        ### displace our running index of position, as we enumerate through a block
        ### d_v does this. If you only provide one block (the whole QGT), this won't matter
        for block in self.blocks:

            d_v_temp = 0
            for i, row in enumerate(block):
                for j, term in enumerate(row):
                    if i <= j:
                        ### if its an objective, call it. Else, it is a float.
                        try:
                            output[i + d_v][j + d_v] = term(variables=variables,samples=samples)
                        except:
                            output[i + d_v][j + d_v] = term
                    else:
                        output[i + d_v][j + d_v] = output[j + d_v][i + d_v]
                d_v_temp += 1
            d_v += d_v_temp

        back = numpy.linalg.pinv(output)
        return back

class CallableVector:
    @property
    def dim(self):
        return (len(self._vector))

    def __init__(self,vector):
        self._vector=vector


    def __call__(self, variables,samples=None):
        output = numpy.empty(self.dim)
        for i,entry in enumerate(self._vector):
            if hasattr(entry, '__call__'):
                output[i] = entry(variables,samples=samples)
            else:
                output[i] = entry
        return output

def get_generator(gate):
    if gate.name.lower() == 'rx':
        gen=paulis.X(gate.target[0])
    elif gate.name.lower() == 'ry':
        gen=paulis.Y(gate.target[0])
    elif gate.name.lower()  == 'rz':
        gen=paulis.Z(gate.target[0])
    elif gate.name.lower() == 'phase':
        gen=paulis.Qm(gate.target[0])
    else:
        print(gate.name.lower())
        raise TequilaException('cant get the generator of a non Gaussian gate, you fool!')
    return gen


def stokes_block(expectation,initial_values=None,samples=None,device=None,
                             backend=None,noise=None):


    U=expectation.U
    ### orders the circuit into alternating layer ansatz, where a moment is all simultaneous gates
    moments=U.canonical_moments
    ### rebuild the sub circuits used in the expectation values that populate the QGT
    sub=[QCircuit.from_moments(moments[:i]) for i in range(1,len(moments),2)]
    ### this is the list of just the moments which are parametrized.
    parametric_moms=[moments[i] for i in range(1,len(moments)+1,2)]
    generators =[]
    ### generators is a list of lists, ultimately, where each sublist is all the generators in order
    ### for a given parametric layer (if said layer is occupied, which it might not be! a layer can be nothing, I.E, the identity.)
    for pm in parametric_moms:
        set=[]
        if len(pm.gates) !=0:
            for gate in pm.gates:
                ### get_generator takes a gaussian gate, and returns the pauli that is its generator. See that function for detail.
                gen=get_generator(gate)
                set.append(gen)
        if len(set) !=0:
            generators.append(set)
        else:
            ### blank sets get passed over
            generators.append(None)
    blocks=[]
    for i,set in enumerate(generators):
        if set is None:
            pass
        else:
            ### a block is a list of lists, and indexing it should correspond to indexing a matrix in A[row][column] fashion.
            ### alternate functions could have the whole QGT be a single block, but you need to have as a return a List of (List of Lists)!!!!
            block=[[0 for _ in range(len(set))] for _ in range(len(set))]
            for k,gen1 in enumerate(set):
                for q,gen2 in enumerate(set):
                    ### make sure you compile the objectives! otherwise this bad boy will not run
                    if k == q:
                        arg= (ExpectationValue(U=sub[i], H=gen1 * gen1) - ExpectationValue(U=sub[i],H=gen1)**2)/4
                    else:
                        arg = (ExpectationValue(U=sub[i], H=gen1 * gen2) - ExpectationValue(U=sub[i], H=gen1)*ExpectationValue(U=sub[i],H=gen2) ) / 4
                    block[k][q] = compile_objective(arg, variables=initial_values, samples=samples, backend=backend,device=device,
                                                    noise=noise)
            blocks.append(block)
    return blocks


def qng_circuit_grad(E: ExpectationValueImpl):
    '''
    implements the analytic partial derivatives of a unitary as it would appear in an expectation value, taking
    all parameters as final (only useful for qng, invalid method otherwise!)
    :param E: the Excpectation whose gradient should be obtained
    :return: vector (as dict) of dU/dpi as Objectives.
    '''
    hamiltonian = E.H
    unitary = E.U

    # fast return if possible
    out=[]
    for i, g in enumerate(unitary.gates):
        if g.is_parametrized():
            if g.is_controlled():
                raise TequilaException("controlled gate in qng circuit gradient: Compiler was not called")
            if hasattr(g, "shift"):
                if hasattr(g._parameter,'extract_variables'):
                    shifter = qng_grad_gaussian(unitary, g, i, hamiltonian)
                    out.append(shifter)
            else:
                print(g, type(g))
                raise TequilaException('No shift found for gate {}'.format(g))
    if out is None:
        raise TequilaException("caught a dead circuit in qng gradient")
    return out


def qng_grad_gaussian(unitary, g, i, hamiltonian):
    '''
    qng function for getting the gradients of gaussian gates.
    THIS variant of the function does not seek out underlying gate parameters; it treats each variable 'as is'.
    This treatment is necessary for the QNG but is incorrect elsewhere.
    :param unitary: QCircuit: the QCircuit object containing the gate to be differentiated
    :param g: a parametrized: the gate being differentiated
    :param i: Int: the position in unitary at which g appears
    :param variable: Variable or String: the variable with respect to which gate g is being differentiated
    :param hamiltonian: the hamiltonian with respect to which unitary is to be measured, in the case that unitary
        is contained within an ExpectationValue
    :return: a list of objectives; the gradient of the Exp. with respect to each of its (internal) parameters
    '''

    ### unlike grad_gaussian, this doesn't dig below, into a gate's underlying parametrization.
    ### In other words, if a gate is Rx(y), y=f(x), this gives you back d Rx / dy.

    if not hasattr(g, "shift"):
        raise TequilaException("No shift found for gate {}".format(g))

    # neo_a and neo_b are the shifted versions of gate g needed to evaluate its gradient
    shift_a = g._parameter + numpy.pi / (4 * g.shift)
    shift_b = g._parameter - numpy.pi / (4 * g.shift)
    neo_a = copy.deepcopy(g)
    neo_a._parameter = shift_a
    neo_b = copy.deepcopy(g)
    neo_b._parameter = shift_b

    U1 = unitary.replace_gates(positions=[i], circuits=[neo_a])
    w1 = g.shift

    U2 = unitary.replace_gates(positions=[i], circuits=[neo_b])
    w2 = -g.shift

    Oplus = ExpectationValueImpl(U=U1, H=hamiltonian)
    Ominus = ExpectationValueImpl(U=U2, H=hamiltonian)
    dOinc = w1 * Objective(args=[Oplus]) + w2 * Objective(args=[Ominus])
    return dOinc



def subvector_procedure(eval,initial_values=None,samples=None,device=None,
                        backend=None,noise=None):
    vect=qng_circuit_grad(eval)
    out=[]
    for entry in vect:
        out.append(compile_objective(entry, variables=initial_values, samples=samples,device=device,
                                     backend=backend,
                                     noise=noise))
    return CallableVector(out)

def get_self_pars(U):
    out=[]
    for g in U.gates:
        if g.is_parametrized():
            if hasattr(g._parameter,'extract_variables'):
                out.append(g._parameter)
    return out

def qng_dict(argument,matrix,subvector,mapping,positional):
    return {'arg':argument,'matrix':matrix,'vector':subvector,'mapping':mapping,'positional':positional}

def get_qng_combos(objective,func=stokes_block,initial_values=None,samples=None,backend=None,device=None,noise=None):
    combos=[]
    vars = objective.extract_variables()
    compiled = compile_multitarget(gate=objective)
    compiled = compile_trotterized_gate(gate=compiled)
    compiled = compile_h_power(gate=compiled)
    compiled = compile_power_gate(gate=compiled)
    compiled = compile_controlled_phase(gate=compiled)
    compiled = compile_controlled_rotation(gate=compiled)
    for i,arg in enumerate(compiled.args):
        if not isinstance(arg,ExpectationValueImpl):
            ### this is a variable, no QNG involved
            mat=QngMatrix([[[1]]])
            vec=CallableVector([__grad_inner(arg, arg)])
            mapping={0:{v:__grad_inner(arg,v) for v in vars}}
        else:
            ### if the arg is an expectationvalue, we need to build some qngs and mappings!
            blocks=func(arg,initial_values=initial_values,samples=samples,device=device,
                                            backend=backend,noise=noise)
            mat=QngMatrix(blocks)

            vec=subvector_procedure(arg,initial_values=initial_values,samples=samples,device=device,
                                    backend=backend,noise=noise)

            mapping={}
            self_pars=get_self_pars(arg.U)
            for j,p in enumerate(self_pars):
                indict={}
                for v in p.extract_variables():
                    gi=__grad_inner(p,v)
                    if isinstance(gi,Objective):
                        g=compile_objective(gi, variables=initial_values, samples=samples,device=device,
                                            backend=backend, noise=noise)
                    else:
                        g=gi
                    indict[v]=g
                mapping[j]=indict

        posarg = jax.grad(compiled.transformation, argnums=i)
        p = Objective(compiled.args, transformation=posarg)

        pos = compile_objective(p, variables=initial_values, samples=samples,device=device,
                                backend=backend, noise=noise)
        combos.append(qng_dict(arg, mat, vec, mapping, pos))
    return combos

def evaluate_qng(combos,variables,samples=None):
    gd={v:0 for v in variables.keys()}
    for c in combos:
        qgt=c['matrix']
        vec=c['vector']
        m=c['mapping']
        pos=c['positional']
        marco=qgt(variables,samples=samples)
        polo=vec(variables,samples=samples)
        ev=numpy.dot(marco,polo)
        for i,val in enumerate(ev):
            maps=m[i]
            for k in maps.keys():
                gd[k] += (val*maps[k]*pos)(variables=variables,samples=samples)

    out=[v for v in gd.values()]
    return out

class QNGVector():

    def __init__(self,combos):
        self.combos=combos

    def __call__(self, variables,samples=None):
        return numpy.asarray(evaluate_qng(self.combos,variables=variables,samples=samples))