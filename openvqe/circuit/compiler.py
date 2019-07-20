"""
Primitive Compiler from Qubit-Operators to evolution operators
Replace with fancier external packages at some point
"""

from openvqe.circuit.circuit import QCircuit, QGate, H, CNOT, Rx, Ry, Rz
import numpy
from openfermion import QubitOperator


def compile_trotter_evolution(cluster_operator: QubitOperator, steps: int = 1, anti_hermitian=True):
    circuit = QCircuit()
    factor = 1.0 / steps
    if anti_hermitian:
        factor = 1.0j / steps
    for index in range(steps):
        for key, value in cluster_operator.terms.items():
            if key == ():
                # dont implement the constant part
                continue
            elif not numpy.isclose(value, 0.0, rtol=1.e-8, atol=1.e-8):
                # don;t make circuit for too small values
                # @todo include ampltidude_neglect_threshold into parameters
                circuit += exponential_pauli_gate(paulistring=key, angle=value * factor)
    return circuit


def exponential_pauli_gate(paulistring, angle):
    """
    Returns the circuit: exp(i*angle*paulistring)
    primitively compiled into X,Y Basis Changes and CNOTs and Z Rotations
    :param paulistring: The paulistring in given as tuple of tuples (openfermion format)
    like e.g  ( (0, 'Y'), (1, 'X'), (5, 'Z') )
    :param angle: The angle which parametrizes the gate -> should be real
    :returns: the above mentioned circuit as abstract structure
    """

    if not numpy.isclose(numpy.imag(angle), 0.0):
        raise Warning("angle is not real, angle=" + str(angle))

    circuit = QCircuit()

    # the general circuit will look like:
    # series which changes the basis if necessary
    # series of CNOTS associated with basis changes
    # Rz gate parametrized on the angle
    # series of CNOT (inverted direction compared to before)
    # series which changes the basis back
    change_basis = QCircuit()
    change_basis_back = QCircuit()
    cnot_cascade = QCircuit()
    reversed_cnot = QCircuit()

    last_qubit = None
    previous_qubit = None
    for pq in paulistring:
        pauli = pq[1]
        qubit = [pq[0]]  # wrap in list for targets= ...

        # see if we need to change the basis
        if pauli.upper() == "X":
            change_basis += H(qubit)
            change_basis_back += H(qubit)
        elif pauli.upper() == "Y":
            change_basis += Rx(target=qubit, angle=numpy.pi / 2)
            change_basis_back += Rx(target=qubit, angle=-numpy.pi / 2)

        if previous_qubit is not None:
            cnot_cascade += CNOT(target=qubit, control=previous_qubit)
        previous_qubit = qubit
        last_qubit = qubit

    reversed_cnot = cnot_cascade.make_dagger()

    # assemble the circuit
    circuit += change_basis
    circuit += cnot_cascade
    # factor 2 is since gates are defined with angle/2
    circuit += Rz(target=last_qubit, angle=2.0 * angle)
    circuit += reversed_cnot
    circuit += change_basis_back

    return circuit
