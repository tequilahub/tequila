"""
Primitive Compiler from Qubit-Operators to evolution operators
Replace with fancier external packages at some point
"""
from openvqe import OpenVQEException
from openvqe.circuit.circuit import QCircuit
from openvqe.circuit.gates import Rx, H, X, Rz, ExpPauli
from openvqe.circuit._gates_impl import RotationGateImpl, QGateImpl, MeasurementImpl
from openvqe.objective import Objective

import numpy, copy


class OpenVQECompilerException(OpenVQEException):
    pass


def compiler(f):
    """
    Decorator for compile functions
    Make them applicable for single gates as well as for whole circuits
    Note that all arguments need to be passed as keyword arguments
    """

    def wrapper(gate, **kwargs):
        if hasattr(gate, "gates"):
            result = QCircuit(weight=gate.weight)
            for g in gate.gates:
                result += f(gate=g, **kwargs)
            return result
        elif hasattr(gate, "unitaries") and hasattr(gate, "observable"):
            compiled = []
            for U in gate.unitaries:
                cU = QCircuit(weight=U.weight)
                for g in U.gates:
                    cU += f(gate=g, **kwargs)
                compiled.append(cU)
            return Objective(observable=gate.observable, unitaries=compiled)
        else:
            return f(gate=gate, **kwargs)

    return wrapper


def change_basis(target, axis, daggered=False):
    if isinstance(axis, str):
        axis = RotationGateImpl.string_to_axis[axis.lower()]

    if axis == 0:
        return H(target=target, frozen=True)
    elif axis == 1 and daggered:
        return Rx(angle=-numpy.pi / 2, target=target, frozen=True)
    elif axis == 1:
        return Rx(angle=numpy.pi / 2, target=target, frozen=True)
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
        gx.target = (t,)
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
        angles = [gate.angle / 2.0, -gate.angle / 2.0]

    assert (len(angles) == 2)

    if len(gate.target) > 1:
        return compile_controlled_rotation(gate=compile_multitarget(gate=gate), angles=angles)

    target = gate.target
    control = gate.control

    result = QCircuit()
    result += change_basis(target=target, axis=gate._axis)
    result += RotationGateImpl(axis="z", target=target, angle=angles[0], frozen=gate.frozen)
    result += QGateImpl(name="X", target=target, control=control)
    result += RotationGateImpl(axis="Z", target=target, angle=angles[1], frozen=gate.frozen)
    result += QGateImpl(name="X", target=target, control=control)
    result += change_basis(target=target, axis=gate._axis, daggered=True)

    result.n_qubits = result.max_qubit() + 1
    return result


@compiler
def compile_swap(gate) -> QCircuit:
    if gate.name.lower() == "swap":
        if len(gate.target) != 2:
            raise OpenVQECompilerException("SWAP gates needs two targets")
        if hasattr(gate, "power") and power != 1:
            raise OpenVQECompilerException("SWAP gate with power can not be compiled into CNOTS")

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
        angle = gate.angle

        if not numpy.isclose(numpy.imag(angle()), 0.0):
            raise OpenVQEException("angle is not real, angle=" + str(angle))

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
        reversed_cnot = QCircuit()

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
        circuit += Rz(target=last_qubit, angle=angle, control=gate.control, frozen=gate.frozen)
        circuit += reversed_cnot
        circuit += ubasis_t

        return circuit

    else:
        return QCircuit.wrap_gate(gate)


def do_compile_trotterized_gate(generator, steps, factor, randomize, control, threshold, frozen):

    assert (generator.is_hermitian())
    circuit = QCircuit()
    factor = factor / steps
    for index in range(steps):
        paulistrings = generator.paulistrings
        if randomize:
            numpy.random.shuffle(paulistrings)
        for ps in paulistrings:
            value = ps.coeff
            # don't make circuit for too small values
            if len(ps) != 0 and not numpy.isclose(value, 0.0, atol=threshold):
                circuit += ExpPauli(paulistring=ps, angle=factor * value, control=control, frozen=frozen)

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
                result += do_compile_trotterized_gate(generator=g, steps=1, factor=c / gate.steps, randomize=gate.randomize, control=gate.control, frozen=gate.frozen, threshold=gate.threshold)
    else:
        if gate.randomize_component_order:
            numpy.random.shuffle(gate.generators)
        for i, g in enumerate(gate.generators):
            if gate.angles is not None:
                c = gate.angles[i]
            result += do_compile_trotterized_gate(generator=g, steps=gate.steps, factor=c, randomize=gate.randomize, control=gate.control, frozen=gate.frozen, threshold=gate.threshold)

    if compile_exponential_pauli:
        return compile_exponential_pauli_gate(result)
    else:
        return result

