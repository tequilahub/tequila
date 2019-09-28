from openvqe.hamiltonian import QubitHamiltonian, PX, PY, PZ, PauliString, Qp, Qm, Sp, Sm, PI, \
    decompose_transfer_operator
from numpy import random
from openvqe import BitString


def test_paulistring_conversion():
    X1 = QubitHamiltonian.init_from_string("x0")
    X2 = PX(0)
    print("X2=", X2)
    keys = [i for i in X2.hamiltonian.terms.keys()]
    pwx = PauliString.init_from_openfermion(key=keys[0], coeff=X2.hamiltonian.terms[keys[0]])
    X3 = QubitHamiltonian.init_from_paulistring(pwx)
    assert (X1 == X2)
    assert (X2 == X3)

    H = PX(0) * PY(1) * PZ(2) + PX(3) * PY(4) * PZ(5)
    PS = []
    for key, value in H.items():
        PS.append(PauliString.init_from_openfermion(key, value))
    PS2 = H.paulistrings
    assert (PS == PS2)

    H = make_random_pauliword(complex=True)
    for i in range(5):
        H += make_random_pauliword(complex=True)
    PS = []
    for key, value in H.items():
        PS.append(PauliString.init_from_openfermion(key, value))
    PS2 = H.paulistrings
    assert (PS == PS2)


def test_simple_arithmetic():
    qubit = random.randint(0, 5)
    primitives = [PX, PY, PZ]
    assert (PX(qubit).conjugate() == PX(qubit))
    assert (PY(qubit).conjugate() == -1 * PY(qubit))
    assert (PZ(qubit).conjugate() == PZ(qubit))
    assert (PX(qubit).transpose() == PX(qubit))
    assert (PY(qubit).transpose() == -1 * PY(qubit))
    assert (PZ(qubit).transpose() == PZ(qubit))
    for P in primitives:
        assert (P(qubit) * P(qubit) == QubitHamiltonian())
        n = random.randint(0, 10)
        nP = QubitHamiltonian.init_zero()
        for i in range(n):
            nP += P(qubit)
        assert (n * P(qubit) == nP)

    for i, Pi in enumerate(primitives):
        i1 = (i + 1) % 3
        i2 = (i + 2) % 3
        assert (Pi(qubit) * primitives[i1](qubit) == 1j * primitives[i2](qubit))
        assert (primitives[i1](qubit) * Pi(qubit) == -1j * primitives[i2](qubit))

        for qubit2 in random.randint(6, 10, 5):
            if qubit2 == qubit: continue
            P = primitives[random.randint(0, 2)]
            assert (Pi(qubit) * primitives[i1](qubit) * P(qubit2) == 1j * primitives[i2](qubit) * P(qubit2))
            assert (P(qubit2) * primitives[i1](qubit) * Pi(qubit) == -1j * P(qubit2) * primitives[i2](qubit))


def test_special_operators():
    # sigma+ sigma- as well as Q+ and Q-
    assert (Sp(0) * Sp(0) == QubitHamiltonian.init_zero())
    assert (Sm(0) * Sm(0) == QubitHamiltonian.init_zero())

    assert (Qp(0) * Qp(0) == Qp(0))
    assert (Qm(0) * Qm(0) == Qm(0))
    assert (Qp(0) * Qm(0) == QubitHamiltonian.init_zero())
    assert (Qm(0) * Qp(0) == QubitHamiltonian.init_zero())

    assert (Sp(0) * Sm(0) == Qp(0))
    assert (Sm(0) * Sp(0) == Qm(0))

    assert (Sp(0) + Sm(0) == PX(0))
    assert (Qp(0) + Qm(0) == PI(0))


def test_transfer_operators():
    assert (decompose_transfer_operator(ket=0, bra=0) == Qp(0))
    assert (decompose_transfer_operator(ket=0, bra=1) == Sp(0))
    assert (decompose_transfer_operator(ket=1, bra=0) == Sm(0))
    assert (decompose_transfer_operator(ket=1, bra=1) == Qm(0))

    assert (decompose_transfer_operator(ket=BitString.from_binary(binary="00"), bra=BitString.from_binary("00")) == Qp(
        0) * Qp(1))
    assert (decompose_transfer_operator(ket=BitString.from_binary(binary="01"), bra=BitString.from_binary("01")) == Qp(
        0) * Qm(1))
    assert (decompose_transfer_operator(ket=BitString.from_binary(binary="01"), bra=BitString.from_binary("10")) == Sp(
        0) * Sm(1))
    assert (decompose_transfer_operator(ket=BitString.from_binary(binary="00"), bra=BitString.from_binary("11")) == Sp(
        0) * Sp(1))

    assert (decompose_transfer_operator(ket=0, bra=0, qubits=[1]) == Qp(1))
    assert (decompose_transfer_operator(ket=0, bra=1, qubits=[1, 2, 3]) == Sp(1))
    assert (decompose_transfer_operator(ket=1, bra=0, qubits=[1]) == Sm(1))
    assert (decompose_transfer_operator(ket=1, bra=1, qubits=[1]) == Qm(1))


def test_conjugation():
    primitives = [PX, PY, PZ]
    factors = [1, -1, 1j, -1j, 0.5 + 1j]
    string = QubitHamiltonian.init_unit()
    cstring = QubitHamiltonian.init_unit()
    for repeat in range(10):
        for q in random.randint(0, 7, 5):
            ri = random.randint(0, 2)
            P = primitives[ri]
            sign = 1
            if ri == 1:
                sign = -1
            factor = factors[random.randint(0, len(factors) - 1)]
            cfactor = factor.conjugate()
            string *= factor * P(qubit=q)
            cstring *= cfactor * sign * P(qubit=q)

        assert (string.conjugate() == cstring)


def test_transposition():
    primitives = [PX, PY, PZ]
    factors = [1, -1, 1j, -1j, 0.5 + 1j]

    assert ((PX(0) * PX(1) * PY(2)).transpose() == -1 * PX(0) * PX(1) * PY(2))
    assert ((PX(0) * PX(1) * PZ(2)).transpose() == PX(0) * PX(1) * PZ(2))

    for repeat in range(10):
        string = QubitHamiltonian.init_unit()
        tstring = QubitHamiltonian.init_unit()
        for q in range(5):
            ri = random.randint(0, 2)
            P = primitives[ri]
            sign = 1
            if ri == 1:
                sign = -1
            factor = factors[random.randint(0, len(factors) - 1)]
            string *= factor * P(qubit=q)
            tstring *= factor * sign * P(qubit=q)

        assert (string.transpose() == tstring)


def make_random_pauliword(complex=True):
    primitives = [PX, PY, PZ]
    result = QubitHamiltonian.init_unit()
    for q in random.choice(range(10), 5, replace=False):
        P = primitives[random.randint(0, 2)]
        real = random.uniform(0, 1)
        imag = 0
        if complex:
            imag = random.uniform(0, 1)
        factor = real + imag * 1j
        result *= factor * P(q)
    return result


def test_dagger():
    assert (PX(0).dagger() == PX(0))
    assert (PY(0).dagger() == PY(0))
    assert (PZ(0).dagger() == PZ(0))

    for repeat in range(10):
        string = make_random_pauliword(complex=False)
        assert (string.dagger() == string)
        assert ((1j * string).dagger() == -1j * string)
