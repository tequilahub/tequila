from openvqe.simulator import SimulatorBase
from openvqe.qubit_wavefunction import QubitWaveFunction
from openvqe.keymap import KeyMapSubregisterToRegister
from openvqe import BitString, BitNumbering, BitStringLSB


def test_keymaps():
    initial_state = 0

    small = QubitWaveFunction.from_int(i=int("0b1111", 2))
    large = QubitWaveFunction.from_int(i=int("0b01010101", 2))
    large.n_qubits = 8

    keymap = KeyMapSubregisterToRegister(register=[0, 1, 2, 3, 4, 5, 6, 7], subregister=[1, 3, 5, 7])

    assert (small.apply_keymap(keymap=keymap, initial_state=initial_state) == large)


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


from openvqe.circuit import QCircuit, gates
from openvqe.simulator.simulator_cirq import SimulatorCirq
from openvqe.simulator.simulator_qiskit import SimulatorQiskit


def test_endianness_simulators():
    tests = ["000111",
             "111000",
             "101010",
             "010101",
             "10010010001",
             "111100101000010"]

    for string in tests:
        number = int(string, 2)
        binary = BitString.from_binary(binary=string)
        c = QCircuit()
        for i, v in enumerate(binary):
            if v == 1:
                c *= gates.X(target=i)

        c *= gates.Measurement(name="", target=[x for x in range(len(string))])

        wfn_cirq = SimulatorCirq().simulate_wavefunction(abstract_circuit=c, initial_state=0).wavefunction
        counts_cirq = SimulatorCirq().run(abstract_circuit=c, samples=1).measurements
        counts_qiskit = SimulatorQiskit().run(abstract_circuit=c, samples=1).measurements
        print("counts_cirq  =", type(counts_cirq))
        print("counts_qiskit=", type(counts_qiskit))
        print("counts_cirq  =", counts_cirq)
        print("counts_qiskit=", counts_qiskit)
        assert (counts_cirq == counts_qiskit)
        assert (wfn_cirq.state == counts_cirq[''].state)

