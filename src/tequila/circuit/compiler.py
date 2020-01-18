"""
Primitive Compiler from Qubit-Operators to evolution operators
Replace with fancier external packages at some point
"""
from tequila import TequilaException
from tequila.circuit.circuit import QCircuit
from tequila.circuit.gates import Rx, H, X, Rz, ExpPauli
from tequila.circuit._gates_impl import RotationGateImpl, QGateImpl, MeasurementImpl
from tequila.utils import to_float
from tequila import Variable
from tequila import Objective
from tequila.objective.objective import ExpectationValueImpl

import numpy, copy, typing


class TequilaCompilerException(TequilaException):
    pass


class Compiler:

    def __init__(self,
                 multitarget=True,
                 multicontrol=False,
                 trotterized=True,
                 exponential_pauli=True,
                 controlled_exponential_pauli=True,
                 controlled_rotation=True,
                 swap=True
                 ):
        self.multitarget = multitarget
        self.multicontrol = multicontrol
        self.trotterized = trotterized
        self.exponential_pauli = exponential_pauli
        self.controlled_exponential_pauli = controlled_exponential_pauli
        self.controlled_rotation = controlled_rotation
        self.swap = swap

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

        compiled = QCircuit()
        for gate in abstract_circuit.gates:
            cg = gate
            controlled = gate.is_controlled()

            # order matters
            # first the real multi-target gates
            if controlled or self.trotterized:
                cg = compile_trotterized_gate(gate=cg)
            if controlled or self.exponential_pauli:
                cg = compile_exponential_pauli_gate(gate=cg)
            if self.swap:
                cg = compile_swap(gate=cg)
            # now every other multitarget gate which might be defined
            if self.multitarget:
                cg = compile_multitarget(gate=cg)
            if self.multicontrol:
                raise NotImplementedError("Multicontrol compilation does not work yet")
            if controlled:
                if self.controlled_rotation:
                    cg = compile_controlled_rotation(gate=cg)
                if self.controlled_exponential_pauli:
                    cg = compile_exponential_pauli_gate(gate=cg)

            compiled += cg

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

    if not hasattr(gate, "angle"):
        return QCircuit.wrap_gate(gate)

    if angles is None:
        angles = [gate.angle / 2, -gate.angle / 2]

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
def compile_swap(gate) -> QCircuit:
    if gate.name.lower() == "swap":
        if len(gate.target) != 2:
            raise TequilaCompilerException("SWAP gates needs two targets")
        if hasattr(gate, "power") and gate.power != 1:
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

    if hasattr(gate, "angle") and hasattr(gate, "paulistring"):

        angle = gate.paulistring.coeff * gate.angle

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
            coeff = to_float(ps.coeff)
            circuit += ExpPauli(paulistring=ps.naked(), angle=factor * coeff, control=control)

    return circuit


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
