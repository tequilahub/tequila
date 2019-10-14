"""
Convenience initialization
Using PX, PY, PZ notation to not confuse with circuits
"""

from openvqe.hamiltonian import QubitHamiltonian


def pauli(qubit, type):
    if type in QubitHamiltonian.axis_to_string:
        type = QubitHamiltonian.axis_to_string(type)
    else:
        type = type.upper()
    return QubitHamiltonian(type + str(qubit))


def PX(qubit):
    return QubitHamiltonian.init_from_string("X" + str(qubit))


def PY(qubit):
    return QubitHamiltonian.init_from_string("Y" + str(qubit))


def PZ(qubit):
    return QubitHamiltonian.init_from_string("Z" + str(qubit))


def PI(qubit):
    return QubitHamiltonian.init_unit()


def Qp(qubit):
    return 0.5 * (PI(qubit=qubit) + PZ(qubit=qubit))


def Qm(qubit):
    return 0.5 * (PI(qubit=qubit) - PZ(qubit=qubit))


def Sp(qubit):
    return 0.5 * (PX(qubit=qubit) + 1.j * PY(qubit=qubit))


def Sm(qubit):
    return 0.5 * (PX(qubit=qubit) - 1.j * PY(qubit=qubit))