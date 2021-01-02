from tequila.circuit.circuit import QCircuit
from tequila.objective.objective import Variable, assign_variable
from tequila.circuit._gates_impl import RotationGateImpl, PowerGateImpl, QGateImpl, \
    ExponentialPauliGateImpl, TrotterizedGateImpl, GeneralizedRotationImpl, PhaseGateImpl, TrotterParameters
import typing, numbers
from tequila.hamiltonian.qubit_hamiltonian import PauliString, QubitHamiltonian
import numpy as np
import functools


def wrap_gate(func):
    @functools.wraps(func)
    def doit(*args, **kwargs):
        return QCircuit.wrap_gate(func(*args, **kwargs))

    return doit


def RotationGate(axis: int, angle: typing.Union[typing.Hashable, numbers.Number], target: typing.Union[list, int],
                 control: typing.Union[list, int] = None):
    """
    Notes
    ----------
    Initialize an abstract rotation gate of the form

    .. math::
        R_{\\text{axis}}(\\text{angle}) = e^{-i\\frac{\\text{angle}}{2} \\sigma_{\\text{axis}}}


    Parameters
    ----------
    axis
        integer 1 for x, 2 for y, 3 for z
    angle
        Hashable type (will be treated as Variable) or Numeric type (static angle)
    target
        integer or list of integers
    control
        integer or list of integers

    Returns
    -------
    QCircuit object with this RotationGate
    """
    return QCircuit.wrap_gate(RotationGateImpl(axis=axis, angle=angle, target=target, control=control))


def PowerGate(name: str, target: typing.Union[list, int], power: bool = None, control: typing.Union[list, int] = None):
    """
    Initialize a (potentially parametrized) gate which is supported on the backend

    Parameters
    ----------
    name: str
        name of the gate on the backend
    target
        int or list of int
    power
        numeric type (fixed exponent) or hashable type (parametrized exponent)
    control
        int or list of int

    Returns
    -------

    """
    return QCircuit.wrap_gate(PowerGateImpl(name=name, power=power, target=target, control=control))


def Phase(phi: typing.Union[typing.Hashable, numbers.Number], target: typing.Union[list, int],
          control: typing.Union[list, int] = None) -> QCircuit:
    """
    Notes
    ----------
    Initialize an abstract phase gate which acts as

    .. math::
        S(\\phi) = \\begin{pmatrix} 1 & 0 \\\\ 0 & e^{i\\phi} \\end{pmatrix}

    Parameters
    ----------
    phi
        defines the phase, can be numeric type (static gate) or hashable non-numeric type (parametrized gate)
    target
        int or list of int
    control
        int or list of int

    Returns
    -------
    QCircuit object

    """
    return QCircuit.wrap_gate(PhaseGateImpl(phase=phi, target=target, control=control))


def S(target: typing.Union[list, int], control: typing.Union[list, int] = None) -> QCircuit:
    """
    Notes
    ----------

    .. math::
        S = \\begin{pmatrix} 1 & 0 \\\\ 0 & e^{i\\frac{\\pi}{2}} \\end{pmatrix}

    Parameters
    ----------
    target
        int or list of int
    control
        int or list of int

    Returns
    -------
    QCircuit object
    """
    return Phase(np.pi / 2, target=target, control=control)


def T(target: typing.Union[list, int], control: typing.Union[list, int] = None):
    """
    Notes
    ----------
    Fixed phase gate

    .. math::
        T = \\begin{pmatrix} 1 & 0 \\\\ 0 & e^{i\\frac{\\pi}{4}} \\end{pmatrix}

    Parameters
    ----------
    target
        int or list of int
    control
        int or list of int

    Returns
    -------
    QCircuit object

    """
    return Phase(np.pi / 4, target=target, control=control)


@wrap_gate
def QGate(name, target: typing.Union[list, int], control: typing.Union[list, int] = None):
    return QGateImpl(name=name, target=target, control=control)


def Rx(angle, target: typing.Union[list, int], control: typing.Union[list, int] = None) -> QCircuit:
    """
    Notes
    ----------
    Rx gate of the form

    .. math::
        R_{x}(\\text{angle}) = e^{-i\\frac{\\text{angle}}{2} \\sigma_{x}}


    Parameters
    ----------
    angle
        Hashable type (will be treated as Variable) or Numeric type (static angle)
    target
        integer or list of integers
    control
        integer or list of integers

    Returns
    -------
    QCircuit object with this RotationGate

    """
    return QCircuit.wrap_gate(RotationGateImpl(axis=0, angle=angle, target=target, control=control))


@wrap_gate
def Ry(angle, target: typing.Union[list, int], control: typing.Union[list, int] = None) -> QCircuit:
    """
    Notes
    ----------
    Ry gate of the form

    .. math::
        R_{y}(\\text{angle}) = e^{-i\\frac{\\text{angle}}{2} \\sigma_{y}}


    Parameters
    ----------
    angle
        Hashable type (will be treated as Variable) or Numeric type (static angle)
    target
        integer or list of integers
    control
        integer or list of integers

    Returns
    -------
    QCircuit object with this RotationGate
    """
    return QCircuit.wrap_gate(RotationGateImpl(axis=1, angle=angle, target=target, control=control))


@wrap_gate
def Rz(angle, target: typing.Union[list, int], control: typing.Union[list, int] = None) -> QCircuit:
    """
    Notes
    ----------
    Rz gate of the form

    .. math::
        R_{z}(\\text{angle}) = e^{-i\\frac{\\text{angle}}{2} \\sigma_{z}}


    Parameters
    ----------
    angle
        Hashable type (will be treated as Variable) or Numeric type (static angle)
    target
        integer or list of integers
    control
        integer or list of integers

    Returns
    QCircuit object with this RotationGate
    -------
    """
    return QCircuit.wrap_gate(RotationGateImpl(axis=2, angle=angle, target=target, control=control))


def X(target: typing.Union[list, int], control: typing.Union[list, int] = None, power=None) -> QCircuit:
    """
    Notes
    ----------
    Pauli X Gate

    Parameters
    ----------
    target
        int or list of int
    control
        int or list of int
    power
        numeric type (fixed exponent) or hashable type (parametrized exponent)

    Returns
    -------
    QCircuit object
    """
    return _initialize_power_gate(name="X", power=power, target=target, control=control)


@wrap_gate
def H(target: typing.Union[list, int], control: typing.Union[list, int] = None, power=None) -> QCircuit:
    """
    Notes
    ----------
    Hadamard gate

    Parameters
    ----------
    target
        int or list of int
    control
        int or list of int
    power
        numeric type (fixed exponent) or hashable type (parametrized exponent)

    Returns
    -------
    QCircuit object

    """
    return _initialize_power_gate(name="H", power=power, target=target, control=control)


def Y(target: typing.Union[list, int], control: typing.Union[list, int] = None, power=None) -> QCircuit:
    """
    Notes
    ----------
    Pauli Y Gate

    Parameters
    ----------
    target:
        int or list of int
    control
        int or list of int
    power
        numeric type (fixed exponent) or hashable type (parametrized exponent)

    Returns
    -------
    QCircuit object

    """
    return _initialize_power_gate(name="Y", power=power, target=target, control=control)


def Z(target: typing.Union[list, int], control: typing.Union[list, int] = None, power=None) -> QCircuit:
    """
    Notes
    ----------
    Pauli Z Gate

    Parameters
    ----------
    target
        int or list of int
    control
        int or list of int
    power
        numeric type (fixed exponent) or hashable type (parametrized exponent)

    Returns
    -------
    QCircuit object

    """
    return _initialize_power_gate(name="Z", power=power, target=target, control=control)


def _initialize_power_gate(name: str, target: typing.Union[list, int], control: typing.Union[list, int] = None,
                           power=None) -> QCircuit:
    if power is None or power in [1, 1.0]:
        return QCircuit.wrap_gate(QGateImpl(name=name, target=target, control=control))
    else:
        return QCircuit.wrap_gate(PowerGateImpl(name=name, power=power, target=target, control=control))


def ExpPauli(paulistring: typing.Union[PauliString, str], angle, control: typing.Union[list, int] = None):
    """Exponentiated Pauligate:
    
    ExpPauli(PauliString, angle) = exp(-i* angle/2* PauliString)

    Parameters
    ----------
    paulistring :
        given as PauliString structure or as string or dict or list
        if given as string: Format should be like X(0)Y(3)Z(2)
        if given as list: Format should be like [(0,'X'),(3,'Y'),(2,'Z')]
        if given as dict: Format should be like { 0:'X', 3:'Y', 2:'Z' }
    angle :
        the angle (will be multiplied by paulistring coefficient if there is one)
    control :
        control qubits
    paulistring: typing.Union[PauliString :
        
    str] :
        
    control: typing.Union[list :
        
    int] :
         (Default value = None)

    Returns
    -------
    type
        Gate wrapped in circuit

    """

    if isinstance(paulistring, str):
        ps = PauliString.from_string(string=paulistring)
    elif isinstance(paulistring, list):
        ps = PauliString.from_openfermion(key=list)
    elif isinstance(paulistring, dict):
        ps = PauliString(data=paulistring)
    else:
        ps = paulistring

    # Failsave: If the paulistring contains just one pauli matrix
    # it is better to initialize a rotational gate due to strange conventions in some simulators
    if len(ps.items()) == 1:
        target, axis = tuple(ps.items())[0]
        return QCircuit.wrap_gate(
            RotationGateImpl(axis=axis, target=target, angle=ps.coeff * assign_variable(angle), control=control))
    else:
        return QCircuit.wrap_gate(ExponentialPauliGateImpl(paulistring=ps, angle=angle, control=control))

def Rp(paulistring: typing.Union[PauliString, str], angle, control: typing.Union[list, int] = None):
    """
    Same as ExpPauli
    """
    return ExpPauli(paulistring=paulistring, angle=angle, control=control)

def GeneralizedRotation(angle: typing.Union[typing.List[typing.Hashable], typing.List[numbers.Real]],
                        generator: QubitHamiltonian,
                        control: typing.Union[list, int] = None,
                        eigenvalues_magnitude: float = 0.5,
                        steps: int = 1) -> QCircuit:
    """

    Notes
    --------
    
    A gates which is shift-rule differentiable
     - its generator only has two distinguishable eigenvalues
     - it is then differentiable by the shift rule
     - eigenvalues_magnitude needs to be given upon initialization (this is "r" and the default is r=1/2)
     - the generator will not be verified to fullfill the properties
     Compiling will be done in analogy to a trotterized gate with steps=1 as default

    The gate will act in the same way as rotations and exppauli gates

    .. math::
        U_{G}(\\text{angle}) = e^{-i\\frac{\\text{angle}}{2} G}
    
    Parameters
    ----------
    angle
        numeric type or hashable symbol or tequila objective
    generator
        tequila QubitHamiltonian or any other structure with paulistrings
    control
        list of control qubits
    eigenvalues_magnitude
        magnitude of eigenvalues, in most papers referred to as "r" (default 0.5)
    steps
        possible Trotterization steps (default 1)

    Returns
    -------
    The gate wrapped in a circuit
    """

    return QCircuit.wrap_gate(
        GeneralizedRotationImpl(angle=assign_variable(angle), generator=generator, control=control, eigenvalues_magnitude=eigenvalues_magnitude, steps=steps))


def Trotterized(generators: typing.List[QubitHamiltonian],
                steps: int,
                angles: typing.Union[
                    typing.List[typing.Hashable], typing.List[numbers.Real], typing.List[Variable]] = None,
                control: typing.Union[list, int] = None,
                parameters: TrotterParameters = None) -> QCircuit:
    """

    Parameters
    ----------
    generators :
        list of generators
    angles :
        coefficients for each generator
    steps :
        trotter steps
    control :
        control qubits
    parameters :
        Additional Trotter parameters, if None then defaults are used
    generators: typing.List[QubitHamiltonian] :
        
    steps: int :
        
    angles: typing.Union[typing.List[typing.Hashable] :
        
    typing.List[numbers.Real] :
        
    typing.List[Variable]] :
         (Default value = None)
    control: typing.Union[list :
        
    int] :
         (Default value = None)
    parameters: TrotterParameters :
         (Default value = None)

    Returns
    -------
    QCircuit

    """

    # convenience
    if not (isinstance(generators, list) or isinstance(generators, tuple)):
        generators = [generators]
    if not (isinstance(angles, list) or isinstance(angles, tuple)):
        angles = [angles]

    if parameters is None:
        parameters = TrotterParameters()

    assigned_angles = [assign_variable(angle) for angle in angles]

    return QCircuit.wrap_gate(
        TrotterizedGateImpl(generators=generators, angles=assigned_angles, steps=steps, control=control,
                            **parameters.__dict__))


"""
Convenience for Two Qubit Gates
iSWAP will only work with cirq, the others will be recompiled
"""


@wrap_gate
def SWAP(first: int, second: int, control: typing.Union[int, list] = None, power: float = None) -> QCircuit:
    """
    Notes
    ----------
    SWAP gate, order of targets does not matter

    Parameters
    ----------
    first: int
        target qubit
    second: int
        target qubit
    control
        int or list of ints
    power
        numeric type (fixed exponent) or hashable type (parametrized exponent)

    Returns
    -------
    QCircuit

    """
    return _initialize_power_gate(name="SWAP", target=[first, second], control=control, power=power)


# @wrap_gate
# def iSWAP(q0: int, q1: int, control: typing.Union[int, list] = None):
#     return initialize_power_gate(name="ISWAP", target=[q0, q1], control=control)


"""
Convenience Initialization Routines for controlled gates
All following the patern: Gate(control_qubit, target_qubit, possible_parameter)
"""


def CNOT(control: int, target: int) -> QCircuit:
    """
    Convenience CNOT initialization

    Parameters
    ----------
    control: int
        control qubit
    target: int
        target qubit

    Returns
    -------
    QCircuit object
    """
    return X(target=target, control=control)


def Toffoli(first: int, second: int, target: int) -> QCircuit:
    """
    Convenience Toffoli initialization

    Parameters
    ----------
    first: int
        first control qubit
    second: int
        second control qubit
    target: int
        target qubit

    Returns
    -------
    QCircuit object

    """
    return X(target=target, control=[first, second])


def CX(control: int, target: int) -> QCircuit:
    """
    Convenience initialization CX (CNOT)

    Parameters
    ----------
    control: int
        control qubit
    target: int
        target qubit

    Returns
    -------
    QCircuit object
    """
    return X(target=target, control=control)


def CY(control: int, target: int) -> QCircuit:
    """
    Convenience initialization CY (controlled Pauli Y)

    Parameters
    ----------
    control: int
        control qubit
    target: int
        target qubit

    Returns
    -------
    QCircuit object
    """
    return Y(target=target, control=control)


def CZ(control: int, target: int) -> QCircuit:
    """
    Convenience initialization CZ (controlled Pauli Z)

    Parameters
    ----------
    control: int
        control qubit
    target: int
        target qubit

    Returns
    -------
    QCircuit object
    """
    return Z(target=target, control=control)


def CRx(control: int, target: int, angle: float) -> QCircuit:
    """
    Convenience initialization CRx (controlled Pauli X Rotation)

    Parameters
    ----------
    control: int
        control qubit
    target: int
        target qubit
    angle:
        Hashable type (will be treated as Variable) or Numeric type (static angle)

    Returns
    -------
    QCircuit object
    """
    return Rx(target=target, control=control, angle=angle)


def CRy(control: int, target: int, angle: float) -> QCircuit:
    """
    Convenience initialization CRy (controlled Pauli Y Rotation)

    Parameters
    ----------
    control: int
        control qubit
    target: int
        target qubit
    angle:
        Hashable type (will be treated as Variable) or Numeric type (static angle)

    Returns
    -------
    QCircuit object
    """
    return Ry(target=target, control=control, angle=angle)


def CRz(control: int, target: int, angle: float) -> QCircuit:
    """
    Convenience initialization CRz (controlled Pauli Z Rotation)

    Parameters
    ----------
    control: int
        control qubit
    target: int
        target qubit
    angle:
        Hashable type (will be treated as Variable) or Numeric type (static angle)

    Returns
    -------
    QCircuit object
    """
    return Rz(target=target, control=control, angle=angle)


def U(theta, phi, lambd, target: typing.Union[list, int], control: typing.Union[list, int] = None):
    """
    Notes
    ----------
    Convenient gate, one of the abstract gates defined by OpenQASM.

    .. math::
        U(\\theta, \\phi, \\lambda) = R_z(\\phi + 3\\pi)R_x(\\pi/2)R_z(\\theta + \\pi)R_x(\\pi/2)R_z(\\lambda)

    Parameters
    ----------
    theta
        first parameter angle
    phi
        second parameter angle
    lamnd
        third parameter angle
    target
        int or list of int
    control
        int or list of int

    Returns
    -------
    QCircuit object

    """

    theta = assign_variable(theta)
    phi = assign_variable(phi)
    lambd = assign_variable(lambd)
    pi_half = assign_variable(np.pi / 2)

    return Rz(angle=phi + 3 * np.pi, target=target, control=control) + \
           Rx(angle=pi_half,         target=target, control=control) + \
           Rz(angle=theta + np.pi,   target=target, control=control) + \
           Rx(angle=pi_half,         target=target, control=control) + \
           Rz(angle=lambd,           target=target, control=control)


def u1(lambd, target: typing.Union[list, int], control: typing.Union[list, int] = None):
    """
    Notes
    ----------
    Convenient gate, one of the abstract gates defined by Quantum Experience Standard Header.
    Changes the phase of a carrier without applying any pulses.

    .. math::
        u1(\\lambda) = U(0, 0, \\lambda) = R_z(\\lambda)

    Parameters
    ----------
    lambd
        parameter angle
    target
        int or list of int
    control
        int or list of int

    Returns
    -------
    QCircuit object

    """

    lambd = assign_variable(lambd)

    return U(theta=0, phi=0, lambd=lambd, target=target, control=control)


def u2(phi, lambd, target: typing.Union[list, int], control: typing.Union[list, int] = None):
    """
    Notes
    ----------
    Convenient gate, one of the abstract gates defined by Quantum Experience Standard Header.
    Uses a single \\pi/2-pulse.

    .. math::
        u2(\\phi, \\lambda) = U(\\pi/2, \\phi, \\lambda) = R_z(\\phi + \\pi/2)R_x(\\pi/2)R_z(\\lambda - \\pi/2)

    Parameters
    ----------
    phi
        first parameter angle
    lambd
        second parameter angle
    target
        int or list of int
    control
        int or list of int

    Returns
    -------
    QCircuit object

    """

    phi = assign_variable(phi)
    lambd = assign_variable(lambd)

    return U(theta=np.pi/2, phi=phi, lambd=lambd, target=target, control=control)


def u3(theta, phi, lambd, target: typing.Union[list, int], control: typing.Union[list, int] = None):
    """
    Notes
    ----------
    Convenient gate, one of the abstract gates defined by Quantum Experience Standard Header
    The most general single-qubit gate.
    Uses a pair of \\pi/2-pulses.

    .. math::
        u3(\\theta, \\phi, \\lambda) = U(\\theta, \\phi, \\lambda)

    Parameters
    ----------
    theta
        first parameter angle
    phi
        second parameter angle
    lambd
        third parameter angle
    target
        int or list of int
    control
        int or list of int

    Returns
    -------
    QCircuit object

    """

    theta = assign_variable(theta)
    phi = assign_variable(phi)
    lambd = assign_variable(lambd)

    return U(theta=theta, phi=phi, lambd=lambd, target=target, control=control)


if __name__ == "__main__":
    G = CRx(1, 0, 2.0)

    print(G)
