from tequila.wavefunction import QubitWaveFunction
from tequila.utils.keymap import KeyMapSubregisterToRegister
from tequila import BitString, BitStringLSB
from tequila.circuit import QCircuit, gates
from tequila import ExpectationValue
from tequila.simulators.simulator_api import simulate
from tequila import INSTALLED_SAMPLERS
from tequila.hamiltonian import QubitHamiltonian, PauliString

import pytest, numpy
from numpy import isclose


def test_keymaps():
    initial_state = 0

    small = QubitWaveFunction.from_int(i=int("0b1111", 2))
    large = QubitWaveFunction.from_int(i=int("0b01010101", 2))
    large.n_qubits = 8

    keymap = KeyMapSubregisterToRegister(register=[0, 1, 2, 3, 4, 5, 6, 7], subregister=[1, 3, 5, 7])

    assert (small.apply_keymap(keymap=keymap, initial_state=initial_state).isclose(large))


def test_endianness():
    tests = ["000111",
             "111000",
             "101010",
             "010101",
             "10010010001",
             "111100101000010"]

    for string in tests:
        bits = len(string)
        i1 = BitString.from_int(int(string, 2))
        i2 = BitString.from_binary(binary=string)
        assert (i1 == i2)

        i11 = BitStringLSB.from_int(int(string, 2))
        i22 = BitStringLSB.from_binary(binary=string[::-1])
        assert (i11 == i22)
        assert (i11.integer == i1.integer)
        assert (i22.integer == i2.integer)
        assert (i11.integer == i2.integer)
        assert (i1 == BitString.from_bitstring(i11))
        assert (i1 == BitString.from_bitstring(i22))
        assert (i2 == BitString.from_bitstring(i11))

@pytest.mark.skipif(condition="cirq" not in INSTALLED_SAMPLERS or "qiskit" not in INSTALLED_SAMPLERS, reason="need cirq and qiskit for cross validation")
def test_endianness_simulators():
    tests = ["000111",
             "111000",
             "101010",
             "010101",
             "10010010001",
             "111100101000010"]

    for string in tests:
        binary = BitString.from_binary(binary=string)
        c = QCircuit()
        for i, v in enumerate(binary):
            if v == 1:
                c += gates.X(target=i)
            if v == 0:
                c += gates.Z(target=i)

        wfn_cirq = simulate(c, initial_state=0, backend="cirq")
        counts_cirq = simulate(c, samples=1, backend="cirq")
        counts_qiskit = simulate(c, samples=1, backend="qiskit")
        print("counts_cirq  =", type(counts_cirq))
        print("counts_qiskit=", type(counts_qiskit))
        print("counts_cirq  =", counts_cirq)
        print("counts_qiskit=", counts_qiskit)
        assert (counts_cirq.isclose(counts_qiskit))
        assert (wfn_cirq.state == counts_cirq.state)


@pytest.mark.parametrize("backend", INSTALLED_SAMPLERS)
@pytest.mark.parametrize("case", [("X(0)Y(1)Z(4)", 0.0), ("Z(0)", 1.0), ("Z(0)Z(1)Z(3)", 1.0), ("Z(0)Z(1)Z(2)Z(3)Z(5)", -1.0)])
def test_paulistring_sampling(backend, case):
    print(case)
    H = QubitHamiltonian.from_paulistrings(PauliString.from_string(case[0]))
    U = gates.X(target=1) + gates.X(target=3) + gates.X(target=5)
    E = ExpectationValue(H=H, U=U)
    result = simulate(E,backend=backend, samples=1)
    assert isclose(result, case[1], 1.e-4)


@pytest.mark.parametrize("backend", INSTALLED_SAMPLERS)
@pytest.mark.parametrize("case", [("X(0)Y(1)Z(4)", 0.0), ("Z(0)X(1)X(5)", -1.0), ("Z(0)X(1)X(3)", 1.0), ("Z(0)X(1)Z(2)X(3)X(5)", -1.0)])
def test_paulistring_sampling_2(backend, case):
    H = QubitHamiltonian.from_paulistrings(PauliString.from_string(case[0]))
    U = gates.H(target=1) + gates.H(target=3) + gates.X(target=5) + gates.H(target=5)
    E = ExpectationValue(H=H, U=U)
    result = simulate(E,backend=backend, samples=1)
    assert (isclose(result, case[1], 1.e-4))

@pytest.mark.parametrize("array", [numpy.random.uniform(0.0,1.0,i) for i in [2,4,8,16]])
def test_qubitwavefunction_array(array):
    print(array)
    wfn = QubitWaveFunction.from_array(arr=array)
    array2 = wfn.to_array()
    assert numpy.allclose(array,array2)
