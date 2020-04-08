"""
Primitive Compiler from Qubit-Operators to evolution operators
Replace with fancier external packages at some point
"""
from tequila import TequilaException
from tequila.circuit.circuit import QCircuit
from tequila.circuit.gates import Rx,Ry, H, X, Rz, ExpPauli,CNOT,Phase,T,Z,Y
from tequila.circuit._gates_impl import RotationGateImpl, PhaseGateImpl, QGateImpl, MeasurementImpl, ExponentialPauliGateImpl, TrotterizedGateImpl,PowerGateImpl
from tequila.utils import to_float
from tequila import Variable
from tequila import Objective
from tequila.objective.objective import ExpectationValueImpl
from tequila.autograd_imports import numpy as jnp
from tequila.autograd_imports import numpy
from numpy import pi as pi

import copy, typing


class TequilaCompilerException(TequilaException):
    pass


class Compiler:

    def __init__(self,
                 multitarget=True,
                 multicontrol=False,
                 trotterized=True,
                 gaussian=True,
                 exponential_pauli=True,
                 controlled_exponential_pauli=True,
                 hadamard_power=True,
                 controlled_power=True,
                 power=True,
                 toffoli=False,
                 controlled_phase=True,
                 phase=True,
                 phase_to_z=False,
                 controlled_rotation=True,
                 swap=True,
                 cc_max=False
                 ):
        self.multitarget = multitarget
        self.multicontrol = multicontrol
        self.gaussian = gaussian
        self.trotterized = trotterized
        self.exponential_pauli = exponential_pauli
        self.controlled_exponential_pauli = controlled_exponential_pauli
        self.hadamard_power= hadamard_power
        self.controlled_power = controlled_power
        self.power=power
        self.toffoli=toffoli
        self.controlled_phase=controlled_phase
        self.phase=phase
        self.phase_to_z=phase_to_z
        self.controlled_rotation = controlled_rotation
        self.swap = swap
        self.cc_max=cc_max
    def __call__(self, objective: typing.Union[Objective, QCircuit, ExpectationValueImpl], *args, **kwargs):
        if isinstance(objective, Objective) or hasattr(objective, "args"):
            return self.compile_objective(objective=objective)
        elif isinstance(objective, QCircuit) or hasattr(objective, "gates"):
            return self.compile_circuit(abstract_circuit=objective)
        elif isinstance(objective, ExpectationValueImpl) or hasattr(objective, "U"):
            return self.compile_objective_argument(arg=objective)

    def compile_objective(self, objective):
        compiled_args = []
        for arg in objective.args:
            compiled_args.append(self.compile_objective_argument(arg))
        return type(objective)(args=compiled_args, transformation=objective._transformation)

    def compile_objective_argument(self, arg):
        if isinstance(arg, ExpectationValueImpl) or (hasattr(arg, "U") and hasattr(arg, "H")):
            return ExpectationValueImpl(H=arg.H, U=self.compile_circuit(abstract_circuit=arg.U))
        elif isinstance(arg, Variable) or hasattr(arg, "name"):
            return arg
        else:
            raise TequilaCompilerException(
                "Unknown argument type for objectives: {arg} or type {type}".format(arg=arg, type=type(arg)))

    def compile_circuit(self, abstract_circuit: QCircuit) -> QCircuit:
        n_qubits = abstract_circuit.n_qubits
        compiled = QCircuit()
        for gate in abstract_circuit.gates:

            cg = gate
            #print('into compile comes ', cg)
            controlled = gate.is_controlled()

            # order matters
            # first the real multi-target gates
            if controlled or self.trotterized:
                cg = compile_trotterized_gate(gate=cg)
            if controlled or self.gaussian:
                cg = compile_gaussian_gate(gate=cg)
            if controlled or self.exponential_pauli:
                cg = compile_exponential_pauli_gate(gate=cg)
            if self.swap:
                cg = compile_swap(gate=cg)
            # now every other multitarget gate which might be defined
            if self.multitarget:
                cg = compile_multitarget(gate=cg)
            if self.multicontrol:
                raise NotImplementedError("Multicontrol compilation does not work yet")

            if self.hadamard_power:
                cg=compile_h_power(gate=cg)
            if self.phase_to_z:
                cg=compile_phase_to_z(gate=cg)
            if self.power:
                cg=compile_power_gate(gate=cg)
            if self.phase:
                cg=compile_phase(gate=cg)
            if controlled:
                if self.cc_max:
                    cg=compile_to_cc(gate=cg)
                if self.controlled_exponential_pauli:
                    cg = compile_exponential_pauli_gate(gate=cg)
                if self.hadamard_power:
                    cg= compile_h_power(gate=cg)
                if self.controlled_power:
                    cg=compile_power_gate(gate=cg)
                if self.controlled_phase:
                    cg= compile_controlled_phase(gate=cg)
                if self.toffoli:
                    cg=compile_toffoli(gate=cg)
                    if self.phase:
                        cg=compile_phase(gate=cg)
                if self.controlled_rotation:
                    cg = compile_controlled_rotation(gate=cg)
                if self.cc_max:
                    cg = compile_to_cc(gate=cg)
            compiled += cg
            #print('out of compile comes ', cg)
        compiled.n_qubits = max(compiled.n_qubits, n_qubits)
        return compiled


def compiler(f):
    """
    Decorator for compile functions
    Make them applicable for single gates as well as for whole circuits
    Note that all arguments need to be passed as keyword arguments
    """

    def wrapper(gate, **kwargs):
        if hasattr(gate, "gates"):
            result = QCircuit()
            for g in gate.gates:
                result += f(gate=g, **kwargs)
            return result

        elif hasattr(gate, 'U'):
            cU = QCircuit()
            for g in gate.U.gates:
                cU += f(gate=g, **kwargs)
            inkwargs = {'H': gate.H, 'U': cU}
            return type(gate)(U=cU, H=gate.H)
        elif hasattr(gate, 'transformation'):
            compiled = []
            for E in gate.args:
                if hasattr(E, 'name'):
                    compiled.append(E)
                else:
                    cU = QCircuit()
                    for g in E.U.gates:
                        cU += f(gate=g, **kwargs)
                    # inkwargs={'U':cU,'H':E.H}
                    compiled.append(type(E)(U=cU, H=E.H))
            # nukwargs={'args':compiled,'transformation':gate._transformation}
            return type(gate)(args=compiled, transformation=gate._transformation)
        else:
            return f(gate=gate, **kwargs)

    return wrapper


def change_basis(target, axis, daggered=False):
    if isinstance(axis, str):
        axis = RotationGateImpl.string_to_axis[axis.lower()]

    if axis == 0:
        return H(target=target)
    elif axis == 1 and daggered:
        return Rx(angle=-numpy.pi / 2, target=target)
    elif axis == 1:
        return Rx(angle=numpy.pi / 2, target=target)
    else:
        return QCircuit()




@compiler
def compile_multitarget(gate) -> QCircuit:
    targets = gate.target

    if hasattr(gate, "generator") or hasattr(gate, "generators") or hasattr(gate, "paulistring"):
        return QCircuit.wrap_gate(gate)

    if isinstance(gate, ExponentialPauliGateImpl) or isinstance(gate, TrotterizedGateImpl):
        return QCircuit.wrap_gate(gate)

    if len(targets) == 1:
        return QCircuit.wrap_gate(gate)

    if isinstance(gate, MeasurementImpl):
        return QCircuit.wrap_gate(gate)

    if gate.name.lower() in ["swap", "iswap"]:
        return QCircuit.wrap_gate(gate)

    result = QCircuit()
    for t in targets:
        gx = copy.deepcopy(gate)
        gx._target = (t,)
        result += gx

    return result

@compiler
def compile_controlled_rotation(gate: RotationGateImpl, angles: list = None) -> QCircuit:
    """
    Recompilation of a controlled-rotation gate
    Basis change into Rz then recompilation of controled Rz, then change basis back
    :param gate: The rotational gate
    :param angles: new angles to set, given as a list of two. If None the angle in the gate is used (default)
    :return: set of gates wrapped in QCircuit class
    """

    if not gate.is_controlled():
        return QCircuit.wrap_gate(gate)

    if not isinstance(gate, RotationGateImpl):
        return QCircuit.wrap_gate(gate)

    if angles is None:
        angles = [gate.parameter / 2, -gate.parameter / 2]

    if len(gate.target) > 1:
        return compile_controlled_rotation(gate=compile_multitarget(gate=gate), angles=angles)

    target = gate.target
    control = gate.control
    result = QCircuit()
    result += change_basis(target=target, axis=gate._axis)
    result += RotationGateImpl(axis="z", target=target, angle=angles[0])
    result += QGateImpl(name="X", target=target, control=control)
    result += RotationGateImpl(axis="Z", target=target, angle=angles[1])
    result += QGateImpl(name="X", target=target, control=control)
    result += change_basis(target=target, axis=gate._axis, daggered=True)

    result.n_qubits = result.max_qubit() + 1
    return result

@compiler
def compile_to_cc(gate) -> QCircuit:
    if not gate.is_controlled:
        return QCircuit.wrap_gate(gate)
    cl=len(gate.control)
    target=gate.target
    control=gate.control
    if cl <= 2:
        return QCircuit.wrap_gate(gate)
    name=gate.name
    back=QCircuit()
    if name in ['X','x','Y','y','Z','z','H','h']:
        if isinstance(gate, PowerGateImpl):
            power=gate.parameter
        else:
            power=1.0
        new=PowerGateImpl(name=name,power=power,target=target,control=control)
        back += compile_power_gate(gate=new,cut=True)
    elif isinstance(gate, RotationGateImpl):
        partial=compile_controlled_rotation(gate=gate)
        back += compile_to_cc(gate=partial)
    elif isinstance(g, PhaseGateImpl):
        partial=compile_controlled_phase(gate=gate)
        back += compile_to_cc(gate=partial)
    else:
        print(gate)
        raise TequilaException('frankly, what the fuck is this gate?')
    return back
@compiler
def compile_toffoli(gate) -> QCircuit:
    if gate.name.lower is not 'x':
        return QCircuit.wrap_gate(gate)
    control=gate.control
    c1=control[1]
    c0=control[0]
    target=gate.target
    result=QCircuit()
    result+=H(target)
    result+=CNOT(c1,target)
    result+=T(target).dagger()
    result+=CNOT(c0,target)
    result+=T(target)
    result+=CNOT(c1,target)
    result+=T(target).dagger()
    result+=CNOT(c0,target)
    result+=T(c1)
    result+=T(target)
    result+=CNOT(c0,c1)
    result+=H(target)
    result+=T(c0)
    result+=T(c1).dagger()
    result+=CNOT(c0,c1)

    return(result)

@compiler
def compile_power_gate(gate,cut=False) -> QCircuit:
    if not isinstance(gate, PowerGateImpl):
        return QCircuit.wrap_gate(gate)
    if gate.name.lower() in ['h','hadamard']:
        return QCircuit.wrap_gate(gate=gate)
    if not gate.is_controlled():
        return compile_power_base(gate=gate)

    return power_recursor(gate=gate,cut=cut)

@compiler
def power_recursor(gate,cut=False) -> QCircuit:
    '''
    if not hasattr(gate,'power'):
        return QCircuit.wrap_gate(gate)
    '''
    result = QCircuit()
    cl=0
    if gate.is_controlled():
        cl=len(gate.control)
    if cl is 0:
        return compile_power_base(gate=gate)
    elif cl is 1:
        return get_axbxc_decomp(gate=gate)

    elif cl is 2 and not cut:
        v = type(gate)(name=gate.name, power=gate.parameter / 2, target=gate.target, control=gate.control[1])
        result += get_axbxc_decomp(v)
        result += CNOT(gate.control[0], gate.control[1])
        vdag = type(gate)(name=gate.name, power=gate.parameter / 2, target=gate.target, control=gate.control[1]).dagger()
        result += get_axbxc_decomp(vdag)
        result += CNOT(gate.control[0], gate.control[1])
        again= type(gate)(name=gate.name, power=gate.parameter / 2, target=gate.target, control=gate.control[0])
        result += get_axbxc_decomp(again)

    elif cl is 2 and cut:
        if gate.name in ['CCx','CCNOT','CCX','X']:
            return QCircuit.wrap_gate(gate)
        else:
            v = type(gate)(name=gate.name, power=gate.parameter / 2, target=gate.target, control=gate.control[1])
            result += get_axbxc_decomp(v)
            result += CNOT(gate.control[0], gate.control[1])
            vdag = type(gate)(name=gate.name, power=gate.parameter / 2, target=gate.target,
                              control=gate.control[1]).dagger()
            result += get_axbxc_decomp(vdag)
            result += CNOT(gate.control[0], gate.control[1])
            again = type(gate)(name=gate.name, power=gate.parameter / 2, target=gate.target, control=gate.control[0])
            result += get_axbxc_decomp(again)

    else:
        v = type(gate)(name=gate.name, power=gate.parameter / 2, target=gate.target, control=gate.control[-1])
        result += get_axbxc_decomp(v)
        result += CNOT(target=gate.control[cl-1], control=gate.control[0:cl-1])
        vdag = type(gate)(name=gate.name, power=gate.parameter / 2, target=gate.target, control=gate.control[-1]).dagger()
        result += get_axbxc_decomp(vdag)
        result += CNOT(target=gate.control[cl-1], control=gate.control[0:cl-1])
        rebuild= type(gate)(name=gate.name, power=gate.parameter / 2, target=gate.target, control=gate.control[:cl - 1])
        result += power_recursor(gate=rebuild,cut=cut)

    return result

@compiler
def compile_power_base(gate):
    if not isinstance(gate, PowerGateImpl):
        return QCircuit.wrap_gate(gate)
    power=gate.parameter
    if gate.name in['H','h','Hadamard','hadamard']:
        return compile_h_power(gate=gate)
    if gate.name is 'X':
        ### off by global phase of Exp[ pi power /2]
        '''
        if we wanted to do it formally we would use the following
        a=-numpy.pi/2
        b=numpy.pi/2
        theta = power*numpy.pi

        result = QCircuit()
        result+= Rz(angle=b,target=gate.target)
        result+= Ry(angle=theta,target=gate.target)
        result+= Rz(angle=a,target=gate.target)
        '''
        result =Rx(angle =power*numpy.pi,target=gate.target)
    elif gate.name is 'Y':
        ### off by global phase of Exp[ pi power /2]
        theta=power*numpy.pi

        result = QCircuit()
        result+= Ry(angle=theta,target=gate.target)
    elif gate.name is 'Z':
        ### off by global phase of Exp[ pi power /2]
        a=0
        b=power*numpy.pi
        theta=0
        result = QCircuit()
        result+= Rz(angle=b,target=gate.target)
    else:
        raise TequilaException('passed a gate with name ' +gate.name + ', which cannot be handled!')
    return result

@compiler
def get_axbxc_decomp(gate):
    if not isinstance(gate, PowerGateImpl) or gate.name not in ['X','Y','Z']:
        return QCircuit.wrap_gate(gate)
    power=gate.parameter
    target=gate.target
    result=QCircuit()
    if gate.name is 'X':
        a=-numpy.pi/2
        b=numpy.pi/2
        theta=power*numpy.pi

        '''
        result+=Phase(numpy.pi*power/2,gate.control)
        result+=Rz(-(a-b)/2,target)
        result+=CNOT(gate.control,target)
        #result+=Rz(-(a+b)/2,target)
        result+=Ry(-theta/2,target)
        result+=CNOT(gate.control,target)
        result+=Ry(theta/2,target)
        result+=Rz(a,target=target)
        '''

        '''
        result+=Rz((a-b)/2,target)
        result+=CNOT(gate.control,target)
        #result+=Rz(-(a+b)/2,target)
        result+=Ry(-theta/2,target)
        result+=CNOT(gate.control,target)
        result+=Ry(theta/2,target)
        result+=Rz(a,target)
        result += Phase(numpy.pi * power / 2, gate.control)
        '''
        result +=Rx(angle=theta,target=target,control=gate.control)
        result += Phase(numpy.pi * power / 2, gate.control)

    elif gate.name is 'Y':
        ### off by global phase of Exp[ pi power /2]

        theta = power * numpy.pi

        '''
        result+=Phase(numpy.pi*power/2,gate.control)
        result+=CNOT(gate.control,target)
        result+=Ry(-theta/2,target)
        result+=CNOT(gate.control,target)
        result+=Ry(theta/2,target)
        '''
        a=0
        b=0
        #result+=Rz((a-b)/2,target)
        result+=CNOT(gate.control,target)
        #result+=Rz(-(a+b)/2,target)
        result+=Ry(-theta/2,target)
        result+=CNOT(gate.control,target)
        result+=Ry(theta/2,target)
        #result+=Rz(a,target)
        result+=Phase(numpy.pi*power/2,gate.control)



    elif gate.name is 'Z':
        a= 0
        b = power * numpy.pi
        theta=0


        result+=Rz(b/2,target)
        result+=CNOT(gate.control,target)
        result+=Rz(-b/2,target)
        result+=CNOT(gate.control,target)
        #result+=Rz(a,target)
        result+=Phase(numpy.pi*power/2,gate.control)


        '''
        result+=Rz(b/2,target)
        result+=CNOT(gate.control,target)
        result+=Rz(-b/2,target)
        result+=CNOT(gate.control,target)
        '''
    return result

@compiler
def compile_h_power(gate) -> QCircuit:
    if not isinstance(gate, PowerGateImpl) or gate.name not in ['H','h','hadamard']:
        return QCircuit.wrap_gate(gate)

    if not gate.is_controlled():
        return hadamard_base(gate=gate)
    return hadamard_recursor(gate=gate)

@compiler
def hadamard_base(gate) ->QCircuit:
    if not isinstance(gate, PowerGateImpl) or gate.name not in ['H','h','hadamard']:
        return QCircuit.wrap_gate(gate)
    power=gate.parameter
    a=power.wrap(a_calc)
    b=power.wrap(b_calc)
    theta=power.wrap(theta_calc)

    result = QCircuit()

    result += Rz(angle=b, target=gate.target)
    result += Ry(angle=theta, target=gate.target)
    result += Rz(angle=a, target=gate.target)

    return result

@compiler
def hadamard_axbxc(gate) -> QCircuit:
    if not isinstance(gate, PowerGateImpl) or gate.name not in ['H','h','hadamard']:
        return QCircuit.wrap_gate(gate)
    power=gate.parameter
    target=gate.target
    a=power.wrap(a_calc)
    b=power.wrap(b_calc)
    theta=power.wrap(theta_calc)
    phase=power*jnp.pi/2

    result = QCircuit()

    result += Rz((a - b) / 2, target)
    result += CNOT(gate.control, target)
    result+=Rz(-(a+b)/2,target)
    result += Ry(-theta / 2, target)
    result += CNOT(gate.control, target)
    result += Ry(theta / 2, target)
    result += Rz(a, target)
    result += Phase(numpy.pi * power / 2, gate.control)

    return result

@compiler
def hadamard_recursor(gate) -> QCircuit:
    if not isinstance(gate, PowerGateImpl) or gate.name not in ['H','h','hadamard']:
        return QCircuit.wrap_gate(gate)
    result = QCircuit()
    cl=0
    if gate.is_controlled():
        cl=len(gate.control)
    if cl is 0:
        return hadamard_base(gate)
    if cl is 1:
        return hadamard_axbxc(gate)

    if cl is 2:
        v = type(gate)(name=gate.name, power=gate.parameter / 2, target=gate.target, control=gate.control[1])
        result += hadamard_axbxc(v)
        result += CNOT(gate.control[0], gate.control[1])
        vdag = type(gate)(name=gate.name, power=gate.parameter / 2, target=gate.target, control=gate.control[1]).dagger()
        result +=hadamard_axbxc(vdag)
        result += CNOT(gate.control[0], gate.control[1])
        again= type(gate)(name=gate.name, power=gate.parameter / 2, target=gate.target, control=gate.control[0])
        result += hadamard_axbxc(again)

    else:
        v = type(gate)(name=gate.name, power=gate.parameter / 2, target=gate.target, control=gate.control[-1])
        result += hadamard_axbxc(v)
        result += CNOT(target=gate.control[cl-1], control=gate.control[0:cl-1])
        vdag = type(gate)(name=gate.name, power=gate.parameter / 2, target=gate.target, control=gate.control[-1]).dagger()
        result += hadamard_axbxc(vdag)
        result += CNOT(target=gate.control[cl-1], control=gate.control[0:cl-1])
        rebuild= type(gate)(name=gate.name, power=gate.parameter / 2, target=gate.target, control=gate.control[:cl - 1])
        result += hadamard_recursor(rebuild)
    return result

def exp(x):
    return jnp.exp( 1j*pi*x)

def root_exp(x):
    return jnp.sqrt(exp(x))

def neg_half_exp(x):
    return jnp.exp( -1j*pi*x/2)

def exp_min_1(x):
    return exp(x)-1

def top_a(x):
    return root_exp(x)*exp_min_1(x)*neg_half_exp(x)

def under_right(x):
    return 3 +2*jnp.sqrt(2)+exp(x)

def bottom(x):
    return jnp.sqrt(exp_min_1(x)*under_right(x))

def my_cosecant(x):
    return 1/jnp.sin(pi*x/2)

def back_log_in(x):
    return -1 + 2*(my_cosecant(x)**2)

def first_log_a(x):
    return 4*jnp.log(top_a(x)/bottom(x))

def second_log_a(x):
    return jnp.log(back_log_in(x))

def a_calc(x):
    return jnp.real((-(0.5)*1j*(2*jnp.arcsinh(1)+first_log_a(x)+second_log_a(x))))



def top_right_in(x):
    return ((3+jnp.cos(pi*x))*(jnp.sin(pi *x /2)**2))**(1/4)

def top_b(x):
    return -(2**(3/4))*root_exp(x)*top_right_in(x)

def log_b(x):
    return 2*jnp.log(top_b(x)/bottom(x))

def b_calc(x):
    return jnp.real((-1j*(jnp.arcsinh(1)+log_b(x))))

def in_the_arc(x):
    return -2/(jnp.sqrt(3+jnp.cos(pi*x)))

def theta_calc(x):
    return jnp.real(2*jnp.arccos(1/in_the_arc(x)))



@compiler
def compile_phase(gate) -> QCircuit:
    if not isinstance(gate, PhaseGateImpl):
        return QCircuit.wrap_gate(gate)
    phase=gate.parameter
    result =QCircuit()
    if len(gate.control) is 0:
        return Rz(angle=phase,target=gate.target)

    if len(gate.control) is 1:
        result += Rz(angle=phase / 2, target=gate.control, control=None)
        result += Rz(angle=phase, target=gate.target, control=gate.control)
        return result
    else:
        return compile_controlled_phase(gate)

@compiler
def compile_phase_to_z(gate) -> QCircuit:
    if not isinstance(gate, PhaseGateImpl):
        return QCircuit.wrap_gate(gate)
    phase=gate.parameter
    return Z(power=phase/pi,target=gate.target,control=gate.control)


@compiler
def compile_controlled_phase(gate)->QCircuit:
    if not isinstance(gate, PhaseGateImpl):
        return QCircuit.wrap_gate(gate)
    if len(gate.control) == 0:
        return QCircuit.wrap_gate(gate)
    count=len(gate.control)
    result =QCircuit()
    phase=gate.parameter

    if count is 1:
        result+=H(target=gate.target)
        result+=CNOT(gate.control,gate.target)
        result+=H(target=gate.target)
        result+=Phase(gate.parameter + numpy.pi, target=gate.target)
    elif count == 2:
        result += Rz(angle=phase/(2**2),target=gate.control[0])
        result += Rz(angle=phase/(2**(1)),target=gate.control[1],control=gate.control[0])
        result += Rz(angle=phase,target=gate.target,control=gate.control)

    elif count >= 3:
        result += Rz(angle=phase / (2 ** count), target=gate.control[0])
        for i in range(1,count):
            result += Rz(angle=phase / (2 ** (count - i)), target=gate.control[i], control=gate.control[0:i])
        result += Rz(angle=phase, target=gate.target, control=gate.control)
    return result
@compiler
def compile_swap(gate) -> QCircuit:
    if gate.name.lower() == "swap":
        if len(gate.target) != 2:
            raise TequilaCompilerException("SWAP gates needs two targets")
        if hasattr(gate, "power") and gate.parameter != 1:
            raise TequilaCompilerException("SWAP gate with power can not be compiled into CNOTS")

        c = []
        if gate.control is not None:
            c = gate.control
        return X(target=gate.target[0], control=gate.target[1] + c) \
               + X(target=gate.target[1], control=gate.target[0] + c) \
               + X(target=gate.target[0], control=gate.target[1] + c)

    else:
        return QCircuit.wrap_gate(gate)


@compiler
def compile_exponential_pauli_gate(gate) -> QCircuit:
    """
    Returns the circuit: exp(i*angle*paulistring)
    primitively compiled into X,Y Basis Changes and CNOTs and Z Rotations
    :param paulistring: The paulistring in given as tuple of tuples (openfermion format)
    like e.g  ( (0, 'Y'), (1, 'X'), (5, 'Z') )
    :param angle: The angle which parametrizes the gate -> should be real
    :returns: the above mentioned circuit as abstract structure
    """

    if hasattr(gate, "paulistring"):

        angle = gate.paulistring.coeff * gate.parameter

        circuit = QCircuit()

        # the general circuit will look like:
        # series which changes the basis if necessary
        # series of CNOTS associated with basis changes
        # Rz gate parametrized on the angle
        # series of CNOT (inverted direction compared to before)
        # series which changes the basis back
        ubasis = QCircuit()
        ubasis_t = QCircuit()
        cnot_cascade = QCircuit()

        last_qubit = None
        previous_qubit = None
        for k, v in gate.paulistring.items():
            pauli = v
            qubit = [k]  # wrap in list for targets= ...

            # see if we need to change the basis
            axis = 2
            if pauli.upper() == "X":
                axis = 0
            elif pauli.upper() == "Y":
                axis = 1
            ubasis += change_basis(target=qubit, axis=axis)
            ubasis_t += change_basis(target=qubit, axis=axis, daggered=True)

            if previous_qubit is not None:
                cnot_cascade += X(target=qubit, control=previous_qubit)
            previous_qubit = qubit
            last_qubit = qubit

        reversed_cnot = cnot_cascade.dagger()

        # assemble the circuit
        circuit += ubasis
        circuit += cnot_cascade
        circuit += Rz(target=last_qubit, angle=angle, control=gate.control)
        circuit += reversed_cnot
        circuit += ubasis_t

        return circuit

    else:
        return QCircuit.wrap_gate(gate)


def do_compile_trotterized_gate(generator, steps, factor, randomize, control):
    assert (generator.is_hermitian())
    circuit = QCircuit()
    factor = factor / steps
    for index in range(steps):
        paulistrings = generator.paulistrings
        if randomize:
            numpy.random.shuffle(paulistrings)
        for ps in paulistrings:
            if len(ps._data) ==0:
                print("ignoring constant term in trotterized gate")
                continue
            coeff = to_float(ps.coeff)
            circuit += ExpPauli(paulistring=ps.naked(), angle=factor * coeff, control=control)

    return circuit

@compiler
def compile_gaussian_gate(gate, compile_exponential_pauli:bool = False):
    if not hasattr(gate, "generator"):
        return QCircuit.wrap_gate(gate)
    if not hasattr(gate, "shift"):
        return QCircuit.wrap_gate(gate)

    return do_compile_trotterized_gate(generator=gate.generator, steps=gate.steps, randomize=False, factor=gate.parameter, control=gate.control)

@compiler
def compile_trotterized_gate(gate, compile_exponential_pauli: bool = False):
    if not hasattr(gate, "generators") or not hasattr(gate, "steps"):
        return QCircuit.wrap_gate(gate)

    c = 1.0
    result = QCircuit()
    if gate.join_components:
        for step in range(gate.steps):
            if gate.randomize_component_order:
                numpy.random.shuffle(gate.generators)
            for i, g in enumerate(gate.generators):
                if gate.angles is not None:
                    c = gate.angles[i]
                result += do_compile_trotterized_gate(generator=g, steps=1, factor=c / gate.steps,
                                                      randomize=gate.randomize, control=gate.control)
    else:
        if gate.randomize_component_order:
            numpy.random.shuffle(gate.generators)
        for i, g in enumerate(gate.generators):
            if gate.angles is not None:
                c = gate.angles[i]
            result += do_compile_trotterized_gate(generator=g, steps=gate.steps, factor=c, randomize=gate.randomize,
                                                  control=gate.control)

    if compile_exponential_pauli:
        return compile_exponential_pauli_gate(result)
    else:
        return result
