"""
Convenience initialization
Of Pauli Operators
"""
import typing
from tequila.hamiltonian import QubitHamiltonian
from tequila import BitString
import numpy


def pauli(qubit, type):
    if type in QubitHamiltonian.axis_to_string:
        type = QubitHamiltonian.axis_to_string(type)
    else:
        type = type.upper()
    return QubitHamiltonian(type + str(qubit))


def X(qubit):
    return QubitHamiltonian.init_from_string("X" + str(qubit))


def Y(qubit):
    return QubitHamiltonian.init_from_string("Y" + str(qubit))


def Z(qubit):
    return QubitHamiltonian.init_from_string("Z" + str(qubit))


def I(*args, **kwargs):
    return QubitHamiltonian.init_unit()


def Qp(qubit):
    return 0.5 * (I(qubit=qubit) + Z(qubit=qubit))


def Qm(qubit):
    return 0.5 * (I(qubit=qubit) - Z(qubit=qubit))


def Sp(qubit):
    return 0.5 * (X(qubit=qubit) + 1.j * Y(qubit=qubit))


def Sm(qubit):
    return 0.5 * (X(qubit=qubit) - 1.j * Y(qubit=qubit))


def Projector(wfn, threshold=0.0) -> QubitHamiltonian:
    """
    :param wfn: a QubitWaveFunction
    :param threshold: neglect all close to zero with given threshold
    :return: The projector |wfn><wfn|
    """
    H = QubitHamiltonian.init_zero()
    for k1, v1 in wfn.items():
        for k2, v2 in wfn.items():
            c = v1 * v2
            if not numpy.isclose(c, 0.0, atol=threshold):
                H += c * decompose_transfer_operator(bra=k1, ket=k2)
    return H


def decompose_transfer_operator(ket: BitString, bra: BitString, qubits: typing.List[int] = None) -> QubitHamiltonian:
    """
    Decompose |ket><bra| into paulistrings
    """

    opmap = {
        (0, 0): Qp,
        (0, 1): Sp,
        (1, 0): Sm,
        (1, 1): Qm
    }

    nbits = None
    if qubits is not None:
        nbits = len(qubits)

    if isinstance(bra, int):
        bra = BitString.from_int(integer=bra, nbits=nbits)
    if isinstance(ket, int):
        ket = BitString.from_int(integer=ket, nbits=nbits)

    b_arr = bra.array
    k_arr = ket.array
    assert (len(b_arr) == len(k_arr))
    n_qubits = len(k_arr)

    if qubits is None:
        qubits = range(n_qubits)

    assert (n_qubits <= len(qubits))

    result = QubitHamiltonian.init_unit()
    for q, b in enumerate(b_arr):
        k = k_arr[q]
        result *= opmap[(k, b)](qubit=qubits[q])

    return result
