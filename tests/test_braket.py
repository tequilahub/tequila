import operator
import pytest
import tequila as tq
import numpy as np
from tequila.circuit.gates import PauliGate
from tequila.objective.objective import make_overlap, make_transition
from tequila.tools.random_generators import make_random_circuit, make_random_hamiltonian


def test_simple_overlap():
    """
    Function that tests if make_overlap function is working correctly.
    It creates a simple circuit in order to cwheck that both real and imaginary
    part are calculated in the right way.

    Returns
    -------
    None.

    """
    # two circuits to test
    U0 = tq.gates.Rz(angle=1.0, target=1)  # tq.gates.H(target=1) + tq.gates.CNOT (1 ,2)
    U1 = tq.gates.Rz(angle=2, target=1)  # tq.gates.X(target=[1,2])#

    objective_real, objective_im = make_overlap(U0=U0, U1=U1)

    Ex = tq.simulate(objective_real)
    Ey = tq.simulate(objective_im)

    exp_val = Ex + 1.0j * Ey

    # print('Evaluated overlap between the two states: {}\n'.format(exp_val))

    # we want the overlap of the wavefunctions
    # # to test we can compute it manually
    wfn0 = tq.simulate(U0)
    wfn1 = tq.simulate(U1)

    test = wfn0.inner(wfn1)
    print("Correct overlap between the two states: {}".format(test))

    # print('The two result are approximately the same?',np.isclose(test, exp_val, atol=1.e-4))

    assert np.isclose(test, exp_val, atol=1.0e-4)

    return


def test_random_overlap():
    """
    Function that tests if make_overlap function is working correctly.
    It creates circuits with random number of qubits, random rotations and
    random angles.

    Returns
    -------
    None.

    """

    # make random circuits

    # np.random.seed(111)
    n_qubits = np.random.randint(1, high=5)
    U = {k: tq.make_random_circuit(n_qubits) for k in range(2)}

    objective_real, objective_im = make_overlap(U0=U[0], U1=U[1])

    Ex = tq.simulate(objective_real)
    Ey = tq.simulate(objective_im)

    exp_val = Ex + 1.0j * Ey

    # we want the overlap of the wavefunctions
    # # to test we can compute it manually
    wfn0 = tq.simulate(U[0])
    wfn1 = tq.simulate(U[1])

    test = wfn0.inner(wfn1)

    # print(test, '\n', exp_val)

    assert np.isclose(test, exp_val, atol=1.0e-4)

    return


def test_simple_transition():
    """
    Function that tests if make_transition function is working correctly.
    It creates a simple circuit in order to check that both real and imaginary
    part of the transition elementare calculated in the right way for a given
    Hamiltonian.

    Returns
    -------
    None.

    """
    # two circuits to test
    U0 = tq.gates.H(target=1) + tq.gates.CNOT(1, 2)  # tq.gates.Rx(angle=1.0, target=1)#
    U1 = tq.gates.X(target=[1, 2]) + tq.gates.Ry(angle=2, target=1)  #

    # defining the hamiltonian
    H = tq.QubitHamiltonian("1.0*Y(0)X(1)+0.5*Y(1)Z(0)")
    # print('Hamiltonian',H,'\n')

    # calculating the transition element
    trans_real, trans_im = make_transition(U0=U0, U1=U1, H=H)

    tmp_real = tq.simulate(trans_real)
    tmp_im = tq.simulate(trans_im)

    trans_el = tmp_real + 1.0j * tmp_im

    # print('Evaluated transition element between the two states: {}'.format( trans_el))

    # # to test we can compute it manually
    # print()
    correct_trans_el = 0.0 + 0.0j

    wfn0 = tq.simulate(U0)

    for ps in H.paulistrings:
        c_k = ps.coeff

        U_k = PauliGate(ps)
        wfn1 = tq.simulate(U1 + U_k)

        tmp = wfn0.inner(wfn1)
        # print('contribution',c_k*tmp)
        correct_trans_el += c_k * tmp

    # print('Correct transition element value: {}'.format(correct_trans_el))

    # print('The two result are approximately the same?',np.isclose(correct_trans_el, trans_el, atol=1.e-4))

    assert np.isclose(correct_trans_el, trans_el, atol=1.0e-4)

    return


def test_random_transition():
    """
    Function that tests if make_transition function is working correctly.
    It creates circuits with random number of qubits, random rotations and
    random angles and a random Hamiltonian with random number of (random)
    Pauli strings.

    Returns
    -------
    None.

    """

    # np.random.seed(111)
    n_qubits = np.random.randint(1, high=5)
    # print(n_qubits)

    U = {k: tq.make_random_circuit(n_qubits) for k in range(2)}

    # print(U[0])
    # print(U[1])

    # make random hamiltonian
    paulis = ["X", "Y", "Z"]
    n_ps = np.random.randint(1, high=2 * n_qubits + 1)

    H = make_random_hamiltonian(n_qubits, paulis=paulis, n_ps=n_ps)

    trans_real, trans_im = make_transition(U0=U[0], U1=U[1], H=H)

    tmp_real = tq.simulate(trans_real)
    tmp_im = tq.simulate(trans_im)

    trans_el = tmp_real + 1.0j * tmp_im

    correct_trans_el = 0.0 + 0.0j

    wfn0 = tq.simulate(U[0])
    # print()
    # print(wfn0)

    for ps in H.paulistrings:
        c_k = ps.coeff

        U_k = PauliGate(ps)
        wfn1 = tq.simulate(U[1] + U_k)

        tmp = wfn0.inner(wfn1)
        correct_trans_el += c_k * tmp

    wfn1 = tq.simulate(U[1])
    # print(wfn1)

    correct_trans_el_2nd = wfn0.inner(H(wfn1))
    # print()
    # print(correct_trans_el, '\n',trans_el, '\n', correct_trans_el_2nd)

    assert np.isclose(correct_trans_el, trans_el, atol=1.0e-4)

    assert np.isclose(correct_trans_el, correct_trans_el_2nd, atol=1.0e-4)

    return


def test_braket():
    """_summary_"""
    # make random circuits

    # np.random.seed(111)
    n_qubits = np.random.randint(1, high=5)

    U = {k: tq.make_random_circuit(n_qubits) for k in range(2)}

    ######## Testing self overlap #########
    self_overlap = tq.BraKet(ket=U[0])
    self_overlap_sim = tq.simulate(self_overlap)

    assert np.isclose(self_overlap_sim.real, 1, atol=1.0e-4)
    assert np.isclose(self_overlap_sim.imag, 0, atol=1.0e-4)

    ######## Testing expectation value #########
    # make random hamiltonian
    paulis = ["X", "Y", "Z"]
    n_ps = np.random.randint(1, high=2 * n_qubits + 1)

    H = make_random_hamiltonian(n_qubits, paulis=paulis, n_ps=n_ps)

    exp_value_tmp = tq.ExpectationValue(H=H, U=U[0])
    br_exp_value = tq.BraKet(ket=U[0], operator=H)

    exp_value = tq.simulate(exp_value_tmp)
    br_exp_value = tq.simulate(br_exp_value)

    # print(exp_value, br_exp_value)
    assert np.isclose(exp_value, br_exp_value, atol=1.0e-4)

    ######## Testing overlap #########

    objective_real, objective_im = make_overlap(U[1], U[0])

    Ex = tq.simulate(objective_real)
    Ey = tq.simulate(objective_im)

    overlap = Ex + 1.0j * Ey

    br_objective = tq.BraKet(ket=U[0], bra=U[1])

    br_overlap = tq.simulate(br_objective)

    assert np.isclose(br_overlap, overlap, atol=1.0e-4)

    ######## Testing transition element #########

    trans_real, trans_im = make_transition(U0=U[1], U1=U[0], H=H)

    tmp_real = tq.simulate(trans_real)
    tmp_im = tq.simulate(trans_im)

    trans_el = tmp_real + 1.0j * tmp_im

    br_trans = tq.BraKet(ket=U[0], bra=U[1], operator=H)

    br_trans_el = tq.simulate(br_trans)

    assert np.isclose(br_trans_el, trans_el, atol=1.0e-4)

    return


def test_braket_gradient_optimization():
    """
    Function that tests that an objective built from transition elements can be optimized.

    The other braket tests only evaluate fixed circuits with tq.simulate, so nothing here
    covered the optimizer path, where the objective has to be a real scalar.

    A single transition element is complex, but a sum over both index orders is real by
    construction. For the circuits below <U1|H|U0> = cos((a+b)/2), so the objective is
    2*cos((a+b)/2) with a minimum of -2 wherever a+b = 2*pi.

    Returns
    -------
    None.

    """
    a, b = tq.Variable("a"), tq.Variable("b")
    U0 = tq.gates.Ry(angle=a, target=0)
    U1 = tq.gates.Ry(angle=b, target=0)
    H = tq.paulis.Z(0)

    objective = tq.braket(ket=U0, bra=U1, operator=H) + tq.braket(ket=U1, bra=U0, operator=H)

    # the imaginary parts of the two orders cancel, the objective is real
    value = complex(tq.simulate(objective, variables={"a": 0.7, "b": 1.3}))
    assert np.isclose(value.real, 2 * np.cos(1.0), atol=1.0e-4)
    assert np.isclose(value.imag, 0.0, atol=1.0e-6)

    for start_a, start_b in [(1.0, 1.0), (0.2, 0.4), (2.5, 0.1), (0.5, 2.8)]:
        # the finite difference step is set explicitly: the default of scipy is smaller than
        # the resolution of a float32 backend (jax without x64), where the objective then
        # looks flat and the optimizer stops at the starting point
        result = tq.minimize(
            objective,
            initial_values={"a": start_a, "b": start_b},
            gradient="2-point",
            method_options={"finite_diff_rel_step": 1.0e-4},
            silent=True,
        )

        assert np.isclose(result.energy, -2.0, atol=1.0e-4)
        assert np.isclose(np.cos((result.variables[a] + result.variables[b]) / 2.0), -1.0, atol=1.0e-4)

    return


def test_braket_complex_objective_not_optimizable():
    """
    Function that tests that a genuinely complex objective is rejected by the optimizer.

    A transition element between two different states is complex and can not be ordered,
    so optimizing it is not defined and has to be reported instead of silently discarding
    the imaginary part. The error has to name the imaginary part: casting the value with
    float() also raises a TypeError, but one that says nothing about the cause.

    Returns
    -------
    None.

    """
    U0 = tq.gates.Ry(angle="a", target=0) + tq.gates.Rz(angle="p", target=0)
    U1 = tq.gates.Ry(angle="b", target=0)
    initial_values = {"a": 0.7, "b": 1.3, "p": 0.9}

    objective = tq.braket(ket=U0, bra=U1)

    assert abs(complex(tq.simulate(objective, variables=initial_values)).imag) > 1.0e-6

    with pytest.raises(TypeError, match="imaginary part"):
        tq.minimize(objective, initial_values=initial_values, silent=True)

    return


def test_braket_analytic_gradient():
    """
    Function that tests that tq.grad over a transition element agrees with finite differences.

    <U1|H|U0> = cos((a+b)/2) for the circuits below, so the derivative with respect to
    both a and b is -0.5*sin((a+b)/2).

    Returns
    -------
    None.

    """
    a, b = tq.Variable("a"), tq.Variable("b")
    U0 = tq.gates.Ry(angle=a, target=0)
    U1 = tq.gates.Ry(angle=b, target=0)
    objective = tq.braket(ket=U0, bra=U1, operator=tq.paulis.Z(0))

    variables = {"a": 0.7, "b": 1.3}
    expected = -0.5 * np.sin((variables["a"] + variables["b"]) / 2.0)

    for variable in [a, b]:
        gradient = complex(tq.simulate(tq.grad(objective, variable), variables=variables)).real
        assert np.isclose(gradient, expected, atol=1.0e-4)

    return


def test_braket_different_qubits():
    """
    Function that tests a braket whose bra and ket act on different qubits.

    Both circuits are simulated and combined by an inner product, so they have to be
    evaluated in the same register. All other braket tests build bra and ket over the
    same qubits, where a mismatch cannot show up.

    Returns
    -------
    None.

    """
    a, b = 0.7, 1.3
    U0 = tq.gates.Ry(angle=a, target=0)
    U1 = tq.gates.Ry(angle=b, target=1)

    # <U1|U0> = <0|Ry(a)|0> * <Ry(b)|0> = cos(a/2)*cos(b/2)
    value = complex(tq.simulate(tq.BraKet(ket=U0, bra=U1)))

    assert np.isclose(value.real, np.cos(a / 2) * np.cos(b / 2), atol=1.0e-4)
    assert np.isclose(value, tq.simulate(U1).inner(tq.simulate(U0)), atol=1.0e-4)

    return


def test_braket_operator_outside_circuits():
    """
    Function that tests a braket whose operator acts on a qubit that neither circuit touches.

    That qubit stays in |0>, so Z acting on it contributes a factor of one and the result
    is the plain overlap.

    Returns
    -------
    None.

    """
    a, b = 0.7, 1.3
    U0 = tq.gates.Ry(angle=a, target=0)
    U1 = tq.gates.Ry(angle=b, target=0)

    value = complex(tq.simulate(tq.BraKet(ket=U0, bra=U1, operator=tq.paulis.Z(5))))

    assert np.isclose(value.real, np.cos((a - b) / 2), atol=1.0e-4)

    return


def test_braket_sampling():
    """
    Function that tests that a braket can be evaluated with a finite number of samples.

    Returns
    -------
    None.

    """
    a, b = 0.7, 1.3
    U0 = tq.gates.Ry(angle=a, target=0)
    U1 = tq.gates.Ry(angle=b, target=0)

    objective = tq.BraKet(ket=U0, bra=U1, operator=tq.paulis.Z(0))

    sampled = complex(tq.simulate(objective, samples=20000))

    assert np.isclose(sampled.real, np.cos((a + b) / 2), atol=5.0e-2)

    return


def test_braket_count_measurements():
    """
    Function that tests that a braket reports the measurements its decomposition needs.

    Real and imaginary part are measured separately, each with one Hadamard test per
    Pauli string, so counting the Pauli strings alone underestimates the effort.

    Returns
    -------
    None.

    """
    U0 = tq.gates.Ry(angle=0.7, target=0)
    U1 = tq.gates.Ry(angle=1.3, target=0)
    H = tq.paulis.Z(0) + tq.paulis.X(0)

    braket = tq.BraKet(ket=U0, bra=U1, operator=H).args[-1]
    real, imaginary = braket.compile()

    assert braket.count_measurements() == real.count_measurements() + imaginary.count_measurements()

    return


def test_real_and_imag_braket():
    """
    Function that tests that RealBraKet and ImagBraKet are usable objectives.

    Both are real valued, so unlike the complex braket they can be simulated,
    differentiated and optimized directly.

    Returns
    -------
    None.

    """
    a, b = tq.Variable("a"), tq.Variable("b")
    U0 = tq.gates.Ry(angle=a, target=0)
    U1 = tq.gates.Ry(angle=b, target=0)
    H = tq.paulis.Z(0)
    variables = {"a": 0.7, "b": 1.3}

    real = tq.RealBraKet(ket=U0, bra=U1, operator=H)
    imaginary = tq.ImagBraKet(ket=U0, bra=U1, operator=H)

    # <U1|Z|U0> = cos((a+b)/2), which is real
    assert np.isclose(tq.simulate(real, variables=variables), np.cos(1.0), atol=1.0e-4)
    assert np.isclose(tq.simulate(imaginary, variables=variables), 0.0, atol=1.0e-4)

    gradient = tq.simulate(tq.grad(real, a), variables=variables)
    assert np.isclose(gradient, -0.5 * np.sin(1.0), atol=1.0e-4)

    result = tq.minimize(real, initial_values=variables, silent=True)
    assert np.isclose(result.energy, -1.0, atol=1.0e-4)

    return


def test_braket_non_hermitian_operator():
    """
    Function that tests a transition element with a non Hermitian operator.

    <bra|O|ket> is well defined for any O, so the backend must not restrict itself to
    Hermitian operators the way an expectation value does.

    Returns
    -------
    None.

    """
    a, b = 0.7, 1.3
    U0 = tq.gates.Ry(angle=a, target=0)
    U1 = tq.gates.Ry(angle=b, target=0)

    # <U1|Z|U0> = cos((a+b)/2), so with the operator i*Z the result is purely imaginary
    value = complex(tq.simulate(tq.BraKet(ket=U0, bra=U1, operator=tq.QubitHamiltonian("1.0j*Z(0)"))))

    assert np.isclose(value.real, 0.0, atol=1.0e-4)
    assert np.isclose(value.imag, np.cos((a + b) / 2), atol=1.0e-4)

    return
