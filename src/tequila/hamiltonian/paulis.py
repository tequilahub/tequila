"""
Convenience initialization
of Pauli Operators. Resulting structures can be added and multiplied together.
Currently uses OpenFermion as backend (QubitOperators)
"""
import typing
from tequila.hamiltonian import QubitHamiltonian
from tequila import BitString
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
import numpy


def pauli(qubit, type) -> QubitHamiltonian:
    """
    Parameters
    ----------
    qubit: int

    type: str or int:
        define if X, Y or Z (0,1,2)

    Returns
    -------
    QubitHamiltonian
    """
    if type in QubitHamiltonian.axis_to_string:
        type = QubitHamiltonian.axis_to_string(type)
    else:
        type = type.upper()
    return QubitHamiltonian(type + str(qubit))


def X(qubit) -> QubitHamiltonian:
    """
    Initialize a single Pauli X Operator

    Parameters
    ----------
    qubit: int
        qubit on which the operator should act

    Returns
    -------
    QubitHamiltonian

    """
    return QubitHamiltonian.init_from_string("X" + str(qubit))


def Y(qubit) -> QubitHamiltonian:
    """
    Initialize a single Pauli Y Operator

    Parameters
    ----------
    qubit: int
        qubit on which the operator should act

    Returns
    -------
    QubitHamiltonian

    """
    return QubitHamiltonian.init_from_string("Y" + str(qubit))


def Z(qubit) -> QubitHamiltonian:
    """
    Initialize a single Pauli Z Operator

    Parameters
    ----------
    qubit: int
        qubit on which the operator should act

    Returns
    -------
    QubitHamiltonian

    """
    return QubitHamiltonian.init_from_string("Z" + str(qubit))


def I(*args, **kwargs) -> QubitHamiltonian:
    """
    Initialize unit Operator

    Returns
    -------
    QubitHamiltonian

    """
    return QubitHamiltonian.init_unit()


def Zero(*args, **kwargs) -> QubitHamiltonian:
    """
    Initialize 0 Operator

    Returns
    -------
    QubitHamiltonian

    """
    return QubitHamiltonian.init_zero()


def Qp(qubit) -> QubitHamiltonian:
    """
    Notes
    ----------
    Initialize

    .. math::
        \\frac{1}{2} \\left( 1 - \\sigma_z \\right)

    Parameters
    ----------
    qubit: int
        qubit on which the operator should act

    Returns
    -------
    QubitHamiltonian

    """
    return 0.5 * (I(qubit=qubit) + Z(qubit=qubit))


def Qm(qubit) -> QubitHamiltonian:
    """
    Notes
    ----------
    Initialize

    .. math::
        \\frac{1}{2} \\left( 1 + \\sigma_z \\right)

    Parameters
    ----------
    qubit: int
        qubit on which the operator should act

    Returns
    -------
    QubitHamiltonian

    """
    return 0.5 * (I(qubit=qubit) - Z(qubit=qubit))


def Sp(qubit) -> QubitHamiltonian:
    """
    Notes
    ----------
    Initialize

    .. math::
        \\frac{1}{2} \\left( \\sigma_x + i\\sigma_y \\right)

    Parameters
    ----------
    qubit: int
        qubit on which the operator should act

    Returns
    -------
    QubitHamiltonian

    """
    return 0.5 * (X(qubit=qubit) + 1.j * Y(qubit=qubit))


def Sm(qubit) -> QubitHamiltonian:
    """
    Notes
    ----------
    Initialize

    .. math::
        \\frac{1}{2} \\left( \\sigma_x + i \\sigma_y \\right)

    Parameters
    ----------
    qubit: int
        qubit on which the operator should act

    Returns
    -------
    QubitHamiltonian

    """
    return 0.5 * (X(qubit=qubit) - 1.j * Y(qubit=qubit))


def Projector(wfn, threshold=0.0) -> QubitHamiltonian:
    """
    Notes
    ----------
    Initialize a projector given by

    .. math::
        H = \\lvert \\Psi \\rangle \\langle \\Psi \\rvert

    Parameters
    ----------
    wfn: QubitWaveFunction :
        The wavefunction onto which the projector projects
        Needs to be passed down as tequilas QubitWaveFunction type
        See the documentation on how to initialize a QubitWaveFunction from
        integer, string or array

    threshold: float: (Default value = 0.0)
        neglect small parts of the operator

    Returns
    -------

    """
    if isinstance(wfn, str):
        wfn = QubitWaveFunction.from_string(string=wfn)
    elif isinstance(wfn, numpy.ndarray):
        wfn = QubitWaveFunction.from_array(arr=wfn)
    H = QubitHamiltonian.init_zero()
    for k1, v1 in wfn.items():
        for k2, v2 in wfn.items():
            c = v1.conjugate() * v2
            if not numpy.isclose(c, 0.0, atol=threshold):
                H += c * decompose_transfer_operator(bra=k1, ket=k2)
    assert (H.is_hermitian())
    return H


def KetBra(ket: QubitWaveFunction, bra: QubitWaveFunction, hermitian: bool = False, threshold: float = 1.e-6):
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
         can also be given as string or array
    bra: QubitWaveFunction:
         QubitWaveFunction which defines the bra element
         can also be given as string or array
    hermitian: bool: (Default False)
         if True the hermitian version H + H^\dagger is returned
    threshold: float: (Default 1.e-6)
         elements smaller than the threshold will be ignored

    Returns
    -------
    a tequila QubitHamiltonian (not necessarily hermitian)

    """
    H = QubitHamiltonian.init_zero()
    if isinstance(ket, str):
        ket = QubitWaveFunction.from_string(string=ket)
    elif isinstance(ket, numpy.ndarray):
        ket = QubitWaveFunction.from_array(arr=ket)
    if isinstance(bra, str):
        bra = QubitWaveFunction.from_string(string=bra)
    elif isinstance(bra, numpy.ndarray):
        ket = QubitWaveFunction.from_array(arr=bra)
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

    result = QubitHamiltonian.init_unit()
    for q, b in enumerate(b_arr):
        k = k_arr[q]
        result *= opmap[(k, b)](qubit=qubits[q])

    return result
