"""
Convenience initialization
of Pauli Operators. Resulting structures can be added and multiplied together.
Currently uses OpenFermion as backend (QubitOperators)
"""
import typing
from tequila.hamiltonian import QubitHamiltonian
from tequila import BitString, TequilaException
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila.tools import list_assignment
import numpy

def from_string(string, openfermion_format=False):
    return QubitHamiltonian.from_string(string=string, openfermion_format=openfermion_format)

def pauli(qubit, type) -> QubitHamiltonian:
    """
    Parameters
    ----------
    qubit: int or list of ints

    type: str or int or list of string or int:
        define if X, Y or Z (0,1,2)

    Returns
    -------
    QubitHamiltonian
    """

    def assign_axis(axis):
        if axis in QubitHamiltonian.axis_to_string:
            return QubitHamiltonian.axis_to_string[axis]
        elif hasattr(axis, "upper"):
            return axis.upper()
        else:
            raise TequilaException("unknown initialization for pauli operator: {}".format(axis))

    if not isinstance(qubit, typing.Iterable):
        qubit = [qubit]
        type = [type]

    type = [assign_axis(x) for x in type]

    init_string = "".join("{}{} ".format(t, q) for t, q in zip(type, qubit))

    return QubitHamiltonian.from_string(string=init_string, openfermion_format=True)


def X(qubit) -> QubitHamiltonian:
    """
    Initialize a single Pauli X Operator

    Parameters
    ----------
    qubit: int or list of ints
        qubit(s) on which the operator should act

    Returns
    -------
    QubitHamiltonian

    """
    qubit = list_assignment(qubit)
    return pauli(qubit=qubit, type=["X"] * len(qubit))


def Y(qubit) -> QubitHamiltonian:
    """
    Initialize a single Pauli Y Operator

    Parameters
    ----------
    qubit: int or list of ints
        qubit(s) on which the operator should act

    Returns
    -------
    QubitHamiltonian

    """
    qubit = list_assignment(qubit)
    return pauli(qubit=qubit, type=["Y"] * len(qubit))


def Z(qubit) -> QubitHamiltonian:
    """
    Initialize a single Pauli Z Operator

    Parameters
    ----------
    qubit: int or list of ints
        qubit(s) on which the operator should act

    Returns
    -------
    QubitHamiltonian

    """
    qubit = list_assignment(qubit)
    return pauli(qubit=qubit, type=["Z"] * len(qubit))


def I(*args, **kwargs) -> QubitHamiltonian:
    """
    Initialize unit Operator

    Returns
    -------
    QubitHamiltonian

    """
    return QubitHamiltonian.unit()


def Zero(*args, **kwargs) -> QubitHamiltonian:
    """
    Initialize 0 Operator

    Returns
    -------
    QubitHamiltonian

    """
    return QubitHamiltonian.zero()


def Qp(qubit) -> QubitHamiltonian:
    """
    Notes
    ----------
    Initialize

    .. math::
        \\frac{1}{2} \\left( 1 - \\sigma_z \\right)

    Parameters
    ----------
    qubit: int or list of ints
        qubit(s) on which the operator should act

    Returns
    -------
    QubitHamiltonian

    """
    qubit = list_assignment(qubit)
    result = I()
    for q in qubit:
        result *= 0.5 * (I(qubit=q) + Z(qubit=q))
    return result


def Qm(qubit) -> QubitHamiltonian:
    """
    Notes
    ----------
    Initialize

    .. math::
        \\frac{1}{2} \\left( 1 + \\sigma_z \\right)

    Parameters
    ----------
    qubit: int or list of ints
        qubit(s) on which the operator should act

    Returns
    -------
    QubitHamiltonian

    """
    qubit = list_assignment(qubit)
    result = I()
    for q in qubit:
        result *= 0.5 * (I(qubit=q) - Z(qubit=q))
    return result


def Sp(qubit) -> QubitHamiltonian:
    """
    Notes
    ----------
    Initialize

    .. math::
        \\frac{1}{2} \\left( \\sigma_x + i\\sigma_y \\right)

    Parameters
    ----------
    qubit: int or list of ints
        qubit(s) on which the operator should act

    Returns
    -------
    QubitHamiltonian

    """
    qubit = list_assignment(qubit)
    result = I()
    for q in qubit:
        result *= 0.5 * (X(qubit=q) + 1.j * Y(qubit=q))
    return result


def Sm(qubit) -> QubitHamiltonian:
    """
    Notes
    ----------
    Initialize

    .. math::
        \\frac{1}{2} \\left( \\sigma_x - i \\sigma_y \\right)

    Parameters
    ----------
    qubit: int or list of ints
        qubit(s) on which the operator should act

    Returns
    -------
    QubitHamiltonian

    """
    qubit = list_assignment(qubit)
    result = I()
    for q in qubit:
        result *= 0.5 * (X(qubit=q) - 1.j * Y(qubit=q))
    return result


def Projector(wfn, threshold=0.0, n_qubits=None) -> QubitHamiltonian:
    """
    Notes
    ----------
    Initialize a projector given by

    .. math::
        H = \\lvert \\Psi \\rangle \\langle \\Psi \\rvert

    Parameters
    ----------
    wfn: QubitWaveFunction or int, or string, or array :
        The wavefunction onto which the projector projects
        Needs to be passed down as tequilas QubitWaveFunction type
        See the documentation on how to initialize a QubitWaveFunction from
        integer, string or array (can also be passed down diretly as one of those types)


    threshold: float: (Default value = 0.0)
        neglect small parts of the operator

    n_qubits: only needed when an integer is given as wavefunction

    Returns
    -------

    """

    wfn = QubitWaveFunction(state=wfn, n_qubits=n_qubits)

    H = QubitHamiltonian.zero()
    for k1, v1 in wfn.items():
        for k2, v2 in wfn.items():
            c = v1.conjugate() * v2
            if not numpy.isclose(c, 0.0, atol=threshold):
                H += c * decompose_transfer_operator(bra=k1, ket=k2)
    assert (H.is_hermitian())
    return H


def KetBra(ket: QubitWaveFunction, bra: QubitWaveFunction, hermitian: bool = False, threshold: float = 1.e-6,
           n_qubits=None):
    """
    Notes
    ----------
    Initialize the general KetBra operator
    .. math::
        H = \\lvert ket \\rangle \\langle bra \\rvert

    e.g.
    wfn1 = tq.QubitWaveFunction.from_string("1.0*|00> + 1.0*|11>").normalize()
    wfn2 = tq.QubitWaveFunction.from_string("1.0*|00>")
    operator = tq.paulis.KetBra(ket=wfn1, bra=wfn1)
    initializes the transfer operator from the all-zero state to a Bell state

    Parameters
    ----------
    ket: QubitWaveFunction:
         QubitWaveFunction which defines the ket element
         can also be given as string or array or integer
    bra: QubitWaveFunction:
         QubitWaveFunction which defines the bra element
         can also be given as string or array or integer
    hermitian: bool: (Default False)
         if True the hermitian version H + H^\dagger is returned
    threshold: float: (Default 1.e-6)
         elements smaller than the threshold will be ignored
    n_qubits: only needed if ket and/or bra are passed down as integers

    Returns
    -------
    QubitHamiltonian:
        a tequila QubitHamiltonian (not necessarily hermitian) representing the KetBra operator desired.

    """
    H = QubitHamiltonian.zero()
    ket = QubitWaveFunction(state=ket, n_qubits=n_qubits)
    bra = QubitWaveFunction(state=bra, n_qubits=n_qubits)

    for k1, v1 in bra.items():
        for k2, v2 in ket.items():
            c = v1.conjugate() * v2
            if not numpy.isclose(c, 0.0, atol=threshold):
                H += c * decompose_transfer_operator(bra=k1, ket=k2)
    if hermitian:
        return H.split()[0]
    else:
        return H.simplify(threshold=threshold)


def decompose_transfer_operator(ket: BitString, bra: BitString, qubits: typing.List[int] = None) -> QubitHamiltonian:
    """
    Notes
    ----------
    Create the operator

    Note that this is operator is not necessarily hermitian
    So be careful when using it as a generator for gates

    e.g.
    decompose_transfer_operator(ket="01", bra="10", qubits=[2,3])
    gives the operator

    .. math::
        \\lvert 01 \\rangle \\langle 10 \\rvert_{2,3}

    acting on qubits 2 and 3

    Parameters
    ----------
    ket: pass an integer, string, or tequila BitString
    bra: pass an integer, string, or tequila BitString
    qubits: pass the qubits onto which the operator acts

    Returns
    -------

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

    result = QubitHamiltonian.unit()
    for q, b in enumerate(b_arr):
        k = k_arr[q]
        result *= opmap[(k, b)](qubit=qubits[q])

    return result
