import pytest
import numpy as np
import openfermion
from tequila.hamiltonian.custom_jw_transform import custom_jw_transform
from tequila.hamiltonian.qubit_hamiltonian import QubitHamiltonian
from numpy.linalg import eig


@pytest.mark.parametrize("fermions, qubits, qubit_op1, qubit_op2",
                         [("3", [1, 2, 3, 4], "Z1 Z2 Z3 X4", "Z1 Z2 Z3 Y4"),
                          ("1", [0, 3], "Z0 X3", "Z0 Y3"),
                          ("4", [0, 1, 2, 3, 4],
                           "Z0 Z1 Z2 Z3 X4", "Z0 Z1 Z2 Z3 Y4")])
def test_custom_jw(fermions, qubits, qubit_op1, qubit_op2):
    actual1 = custom_jw_transform(fermions, qubits)
    expected1 = QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op1, coefficient=0.5)
    ) + QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op2, coefficient=0.5j))
    assert actual1 == expected1


@pytest.mark.parametrize("fermions, qubits, qubit_op1, qubit_op2",
                         [("3^", [1, 2, 3, 4], "Z1 Z2 Z3 X4", "Z1 Z2 Z3 Y4"),
                          ("2^", [0, 3, 5], "Z0 Z3 X5", "Z0 Z3 Y5"),
                          ("5^", [0, 2, 3, 4, 6, 7], "Z0 Z2 Z3 Z4 Z6 X7",
                           "Z0 Z2 Z3 Z4 Z6 Y7")])
def test_custom_jw_dg(fermions, qubits, qubit_op1, qubit_op2):
    actual1 = custom_jw_transform(fermions, qubits)
    expected1 = QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op1, coefficient=0.5)
    ) + QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op2, coefficient=-0.5j))
    assert actual1 == expected1


@pytest.mark.parametrize("fermions, qubits",
                         [("3",  [1, 2, 3]),
                          ("1", [2]),
                          ("4^", [0, 4])])
def test_custom_jw_short_qubits_list(fermions, qubits):
    with pytest.raises(Exception) as e_info:
        custom_jw_transform(fermions, qubits)


@pytest.mark.parametrize("fermions, qubits, qubit_op1, qubit_op2",
                         [("3", [1, 2, 3, 4, 5], "Z1 Z2 Z3 X4", "Z1 Z2 Z3 Y4"),
                          ("1", [1, 2, 5], "Z1 X2", "Z1 Y2"),
                          ("4", [1, 2, 3, 4, 5, 6, 7],
                           "Z1 Z2 Z3 Z4 X5", "Z1 Z2 Z3 Z4 Y5")])
def test_custom_jw_long_qubits_list(fermions, qubits, qubit_op1, qubit_op2):
    actual1 = custom_jw_transform(fermions, qubits)
    expected1 = QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op1, coefficient=0.5)
    ) + QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op2, coefficient=0.5j))
    assert actual1 == expected1


@pytest.mark.parametrize("fermions, qubits, qubit_op1, qubit_op2",
                         [("3^", [1, 2, 3, 4, 5], "Z1 Z2 Z3 X4", "Z1 Z2 Z3 Y4"),
                          ("1^", [1, 2, 5], "Z1 X2", "Z1 Y2"),
                          ("4^", [1, 2, 3, 4, 5, 6, 7],
                           "Z1 Z2 Z3 Z4 X5", "Z1 Z2 Z3 Z4 Y5")])
def test_custom_jw_long_qubits_list_dg(fermions, qubits, qubit_op1, qubit_op2):
    actual1 = custom_jw_transform(fermions, qubits)
    expected1 = QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op1, coefficient=0.5)
    ) + QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op2, coefficient=-0.5j))
    assert actual1 == expected1


@pytest.mark.parametrize("fermions, qubits, qubit_map, qubit_op1, qubit_op2",
                         [("3", [1, 2, 3, 4], {"3": 3},
                           "Z1 Z2 Z4 X3", "Z1 Z2 Z4 Y3"),
                          ("1", [1, 2, 3, 4], {"1": 4}, "Z1 X4", "Z1 Y4"),
                          ("4", [1, 2, 3, 4, 6], {"4": 2},
                           "Z1 Z3 Z4 Z6 X2", "Z1 Z3 Z4 Z6 Y2")])
def test_custom_jw_qubit_map(
        fermions, qubits, qubit_map, qubit_op1, qubit_op2):
    actual = custom_jw_transform(fermions, qubits, qubit_map)
    expected = QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op1, coefficient=0.5)
    ) + QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op2, coefficient=0.5j))

    assert actual == expected


@pytest.mark.parametrize("fermions, qubits, qubit_map, qubit_op1, qubit_op2",
                         [("3^", [1, 2, 3, 4], {"3^": 3},
                           "Z1 Z2 Z4 X3", "Z1 Z2 Z4 Y3"),
                          ("1^", [1, 2, 3, 4], {"1^": 4}, "Z1 X4", "Z1 Y4"),
                          ("4^", [1, 2, 3, 4, 6], {"4^": 2},
                           "Z1 Z3 Z4 Z6 X2", "Z1 Z3 Z4 Z6 Y2")])
def test_custom_jw_qubit_map_dg(
        fermions, qubits, qubit_map, qubit_op1, qubit_op2):
    actual = custom_jw_transform(fermions, qubits, qubit_map)
    expected = QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op1, coefficient=0.5)
    ) + QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op2, coefficient=-0.5j))

    assert actual == expected


@pytest.mark.parametrize("fermions, qubits, qubit_map, qubits_op1, qubit_op2",
                         [("3", [1, 2, 3, 4], {"4": 3},
                           "Z1 Z2 Z3 X4", "Z1 Z2 Z3 Y4"),
                          ("1", [1, 2, 3, 4], {"2": 4}, "Z1 X2", "Z1 Y2"),
                          ("4", [1, 2, 3, 4, 6], {"4^": 2},
                           "Z1 Z2 Z3 Z4 X6", "Z1 Z2 Z3 Z4 Y6")])
def test_custom_jw_no_key(fermions, qubits, qubit_map, qubits_op1, qubit_op2):
    actual = custom_jw_transform(fermions, qubits, qubit_map)
    expected = QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubits_op1, coefficient=0.5)
    ) + QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op2, coefficient=0.5j))
    assert actual == expected


@pytest.mark.parametrize("fermions, qubits, qubit_map, qubits_op1, qubit_op2",
                         [("3^", [1, 2, 3, 4], {"4^": 3},
                           "Z1 Z2 Z3 X4", "Z1 Z2 Z3 Y4"),
                          ("1^", [1, 2, 3, 4], {"1": 4}, "Z1 X2", "Z1 Y2"),
                          ("4^", [1, 2, 3, 4, 6], {"5^": 2},
                           "Z1 Z2 Z3 Z4 X6", "Z1 Z2 Z3 Z4 Y6")])
def test_custom_jw_no_key_dg(fermions, qubits, qubit_map, qubits_op1, qubit_op2):
    actual = custom_jw_transform(fermions, qubits, qubit_map)
    expected = QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubits_op1, coefficient=0.5)
    ) + QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op2, coefficient=-0.5j))
    assert actual == expected


@pytest.mark.parametrize("fermions, qubits, qubit_map",
                         [("3", [1, 2, 3, 4], {"3": 5}),
                          ("1", [0, 1], {"1": 3}),
                          ("4", [0, 1, 2, 3, 5], {"4": 6})])
def test_custom_jw_no_qubit(fermions, qubits, qubit_map):
    with pytest.raises(Exception) as e_info:
        custom_jw_transform(fermions, qubits, qubit_map)


@pytest.mark.parametrize("fermions, qubits, qubit_map",
                         [("3", [2, 2, 2, 2], {"3": 2}),
                          ("1", [1, 1], {"1": 1}),
                          ("4^", [0, 1, 0, 2, 1], {"4^": 2})])
def test_custom_jw_duplicate_qubits(fermions, qubits, qubit_map):
    with pytest.raises(Exception) as e_info:
        custom_jw_transform(fermions, qubits, qubit_map)


@pytest.mark.parametrize("fermions, qubits, qubit_op1, qubit_op2,"
                         " qubit_op3, qubit_op4",
                         [("3 4^", [[2, 3, 4, 1], [1, 2, 3, 4, 5]],
                           "Z2 Z3 Z4 X1", "Z2 Z3 Z4 Y1",
                           "Z1 Z2 Z3 Z4 X5", "Z1 Z2 Z3 Z4 Y5"),
                          ("1 1^", [[2, 3, 1], [1, 2, 4, 5]],
                           "Z2 X3", "Z2 Y3",
                           "Z1 X2", "Z1 Y2"),
                          ("2 4^", [[2, 3, 1], [1, 2, 3, 4, 5]],
                           "Z2 Z3 X1", "Z2 Z3 Y1",
                           "Z1 Z2 Z3 Z4 X5", "Z1 Z2 Z3 Z4 Y5")])
def test_custom_jw_multiple_no_map(fermions, qubits, qubit_op1,
                                   qubit_op2, qubit_op3, qubit_op4):
    actual = custom_jw_transform(fermions, qubits)
    expected = QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op1, coefficient=0.5)
    ) + QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op2, coefficient=0.5j))
    expected *= (QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op3, coefficient=0.5)
    ) + QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op4, coefficient=-0.5j)))
    assert actual == expected


@pytest.mark.parametrize("fermions, qubits, qubit_map, qubit_op1, qubit_op2,"
                         " qubit_op3, qubit_op4, qubit_op5, qubit_op6",
                         [("3 4^ 1", [[2, 3, 4, 1], [1, 2, 3, 4, 5], [0, 1]],
                           {"3": 2}, "Z3 Z4 Z1 X2", "Z3 Z4 Z1 Y2",
                           "Z1 Z2 Z3 Z4 X5", "Z1 Z2 Z3 Z4 Y5",
                           "Z0 X1", "Z0 Y1"),
                          ("1 1^ 2", [[2, 3, 1], [1, 2, 4, 5], [1, 2, 3]],
                           {"1": 2, "1^": 4}, "Z3 X2", "Z3 Y2",
                           "Z1 X4", "Z1 Y4",
                           "Z1 Z2 X3", "Z1 Z2 Y3"),
                          ("2 4^ 1", [[2, 3, 1], [1, 2, 3, 4, 5], [2, 3]],
                           {"2": 3, "4^": 3}, "Z2 Z1 X3", "Z2 Z1 Y3",
                           "Z1 Z2 Z4 Z5 X3", "Z1 Z2 Z4 Z5 Y3",
                           "Z2 X3", "Z2 Y3")])
def test_custom_jw_multiple_with_map(
        fermions, qubits, qubit_map, qubit_op1, qubit_op2,
        qubit_op3, qubit_op4, qubit_op5, qubit_op6):
    actual = custom_jw_transform(fermions, qubits, qubit_map)
    expected = QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op1, coefficient=0.5)
    ) + QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op2, coefficient=0.5j))
    expected *= (QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op3, coefficient=0.5)
    ) + QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op4, coefficient=-0.5j)))
    expected *= ((QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op5, coefficient=0.5)
    ) + QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op6, coefficient=0.5j))))

    assert actual == expected


@pytest.mark.parametrize("fermions, qubits, qubit_map",
                         [("3 4^", [[2, 3, 4, 1]], {"3": 2}),
                          ("1 1^", [[2, 3, 1]], {"1": 2, "1^": 4}),
                          ("2 4^", [[2, 3, 1]],
                           {"2": 3, "4^": 3})])
def test_custom_jw_multiple_short_qubits_list(fermions, qubits, qubit_map):
    with pytest.raises(Exception) as e_info:
        custom_jw_transform(fermions, qubits, qubit_map)


@pytest.mark.parametrize(
    "fermions1, fermions2, qubits1, qubits2, qubit_map1, qubit_map2,"
    " qubit_op1, qubit_op2, qubit_op3, qubit_op4, "
    "qubit_op5, qubit_op6, qubit_op7, qubit_op8", [
        ("3 4", "2 1", [[2, 3, 4, 1], [1, 2, 3, 4, 5]],  [[2, 3, 1], [2, 3]],
         {"3": 2}, {"2": 3, "4^": 3}, "Z3 Z4 Z1 X2", "Z3 Z4 Z1 Y2",
         "Z1 Z2 Z3 Z4 X5", "Z1 Z2 Z3 Z4 Y5", "Z2 Z1 X3", "Z2 Z1 Y3",
         "Z2 X3", "Z2 Y3"),
        ("1 2", "5 1", [[2, 3, 4], [1, 2, 3, 4, 5]],
         [[2, 3, 1, 0, 4, 6], [2, 3]],  {"2": 2}, {"5": 3, "1": 3},
         "Z2 X3", "Z2 Y3", "Z1 Z3 X2", "Z1 Z3 Y2",
         "Z2 Z1 Z0 Z4 Z6 X3", "Z2 Z1 Z0 Z4 Z6 Y3",
         "Z2 X3", "Z2 Y3"),
    ])
def test_custom_jw_multiple_addition(
        fermions1, fermions2, qubits1, qubits2, qubit_map1, qubit_map2,
        qubit_op1, qubit_op2, qubit_op3, qubit_op4,
        qubit_op5, qubit_op6, qubit_op7, qubit_op8):
    result1 = custom_jw_transform(fermions1, qubits1, qubit_map1)
    result2 = custom_jw_transform(fermions2, qubits2, qubit_map2)
    actual = result1 + result2
    expected1 = QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op1, coefficient=0.5)
    ) + QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op2, coefficient=0.5j))
    expected1 *= QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op3, coefficient=0.5)
    ) + QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op4, coefficient=0.5j))
    expected2 = QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op5, coefficient=0.5)
    ) + QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op6, coefficient=0.5j))
    expected2 *= QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op7, coefficient=0.5)
    ) + QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op8, coefficient=0.5j))
    expected = expected1 + expected2
    assert actual == expected


@pytest.mark.parametrize(
    "fermions1, fermions2, qubits1, qubits2, qubit_map1, qubit_map2,"
    " qubit_op1, qubit_op2, qubit_op3, qubit_op4, "
    "qubit_op5, qubit_op6, qubit_op7, qubit_op8", [
        ("3 4^", "2 1^", [[2, 3, 4, 1], [1, 2, 3, 4, 5]],  [[2, 3, 1], [2, 3]],
         {"3": 2}, {"2": 3, "4^": 3}, "Z3 Z4 Z1 X2", "Z3 Z4 Z1 Y2",
         "Z1 Z2 Z3 Z4 X5", "Z1 Z2 Z3 Z4 Y5", "Z2 Z1 X3", "Z2 Z1 Y3",
         "Z2 X3", "Z2 Y3"),
        ("1 2^", "5 1^", [[2, 3, 4], [1, 2, 3, 4, 5]],
         [[2, 3, 1, 0, 4, 6], [2, 3]],  {"2^": 2}, {"5": 3, "1^": 3},
         "Z2 X3", "Z2 Y3", "Z1 Z3 X2", "Z1 Z3 Y2",
         "Z2 Z1 Z0 Z4 Z6 X3", "Z2 Z1 Z0 Z4 Z6 Y3",
         "Z2 X3", "Z2 Y3"),
    ])
def test_custom_jw_multiple_addition_dg(
        fermions1, fermions2, qubits1, qubits2, qubit_map1, qubit_map2,
        qubit_op1, qubit_op2, qubit_op3, qubit_op4,
        qubit_op5, qubit_op6, qubit_op7, qubit_op8):
    result1 = custom_jw_transform(fermions1, qubits1, qubit_map1)
    result2 = custom_jw_transform(fermions2, qubits2, qubit_map2)
    actual = result1 + result2
    expected1 = QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op1, coefficient=0.5)
    ) + QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op2, coefficient=0.5j))
    expected1 *= QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op3, coefficient=0.5)
    ) + QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op4, coefficient=-0.5j))
    expected2 = QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op5, coefficient=0.5)
    ) + QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op6, coefficient=0.5j))
    expected2 *= QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op7, coefficient=0.5)
    ) + QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op8, coefficient=-0.5j))
    expected = expected1 + expected2
    assert actual == expected


@pytest.mark.parametrize("fermions, qubit_op1, qubit_op2",
                         [("1", "Z0 X1", "Z0 Y1"),
                          ("2", "Z0 Z1 X2", "Z0 Z1 Y2"),
                          ("5", "Z0 Z1 Z2 Z3 Z4 X5", "Z0 Z1 Z2 Z3 Z4 Y5")])
def test_custom_jw_no_qubits(fermions, qubit_op1, qubit_op2):
    actual = custom_jw_transform(fermions)
    expected = QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op1, coefficient=0.5)
    ) + QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op2, coefficient=0.5j))
    assert actual == expected


@pytest.mark.parametrize("fermions, qubit_op1, qubit_op2",
                         [("1^", "Z0 X1", "Z0 Y1"),
                          ("2^", "Z0 Z1 X2", "Z0 Z1 Y2"),
                          ("5^", "Z0 Z1 Z2 Z3 Z4 X5", "Z0 Z1 Z2 Z3 Z4 Y5")])
def test_custom_jw_no_qubits_dg(fermions, qubit_op1, qubit_op2):
    actual = custom_jw_transform(fermions)
    expected = QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op1, coefficient=0.5)
    ) + QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op2, coefficient=-0.5j))
    assert actual == expected


@pytest.mark.parametrize("fermions, qubit_op1, qubit_op2, qubit_op3, qubit_op4",
                         [("1 2", "Z0 X1", "Z0 Y1", "Z0 Z1 X2", "Z0 Z1 Y2"),
                          ("2 4", "Z0 Z1 X2", "Z0 Z1 Y2", "Z0 Z1 Z2 Z3 X4",
                           "Z0 Z1 Z2 Z3 Y4"),
                          ("5 1", "Z0 Z1 Z2 Z3 Z4 X5", "Z0 Z1 Z2 Z3 Z4 Y5",
                           "Z0 X1", "Z0 Y1")])
def test_custom_jw_no_qubits_multiple(fermions, qubit_op1,
                                      qubit_op2, qubit_op3, qubit_op4):
    actual = custom_jw_transform(fermions)
    expected = QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op1, coefficient=0.5)
    ) + QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op2, coefficient=0.5j))
    expected *= QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op3, coefficient=0.5)
    ) + QubitHamiltonian(qubit_operator=openfermion.QubitOperator(
        term=qubit_op4, coefficient=0.5j))
    assert actual == expected


@pytest.mark.parametrize("fermions, qubits",
                         [("1 1^", [[0, 1], [2, 3]]),
                          ("2^ 2", [[0, 1, 2], [2, 3, 4]]),
                          ("3 3^", [[0, 2, 1, 4], [2, 3, 4, 5]])])
def test_custom_jw_hermitian_eigen_value(fermions, qubits):
    actual = custom_jw_transform(fermions, qubits).to_matrix()

    a = np.array(actual)
    w, v = eig(a)
    for k in w:
        if round(k) != -1 and round(k) != 0 and round(k) != 1:
            assert False
    assert True


@pytest.mark.parametrize("fermions, qubits",
                         [("1 2^", [[0, 1], [2, 3, 4]]),
                          ("2 3", [[0, 1, 2], [2, 3, 4, 1]]),
                          ("1 3^", [[0, 2, 1], [2, 3, 4, 5]])])
def test_custom_jw_zero_eigen_value(fermions, qubits):
    actual = custom_jw_transform(fermions, qubits).to_matrix()

    a = np.array(actual)
    w, v = eig(a)
    for k in w:
        if round(k) != -1 and round(k) != 0 and round(k) != 1:
            assert False
    assert True
