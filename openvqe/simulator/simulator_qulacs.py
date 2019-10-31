import qulacs
from openvqe import OpenVQEException, BitString, QubitWaveFunction, BitNumbering
from openvqe.simulator import Simulator, SimulatorReturnType
from openvqe.simulator.simulator import BackendHandler
from openvqe.circuit import QCircuit

"""
todo: overwrite simulate_objective for this simulator, might be faster

Qulacs uses different Rotational Gate conventions: Rx(angle) = exp(i angle/2 X) instead of exp(-i angle/2 X)
"""

class OpenVQEQulacsException(OpenVQEException):
    def __str__(self):
        return "Error in qulacs backend:" + self.message


class BackenHandlerQulacs(BackendHandler):

    recompile_swap = False
    recompile_multitarget = True
    recompile_controlled_rotation = True

    def fast_return(self, abstract_circuit):
        False

    def initialize_circuit(self, qubit_map, *args, **kwargs):
        n_qubits = len(qubit_map)
        return qulacs.QuantumCircuit(n_qubits)

    def add_gate(self, gate, circuit, qubit_map, *args, **kwargs):
        getattr(circuit, "add_" + gate.name.upper() + "_gate")(qubit_map[gate.target[0]])

    def add_controlled_gate(self, gate, qubit_map, circuit, *args, **kwargs):
        assert (len(gate.control) == 1)
        if gate.name.upper() == "X":
            getattr(circuit, "add_CNOT_gate")(qubit_map[gate.control[0]], qubit_map[gate.target[0]])
        if gate.name.upper() == "Z":
            getattr(circuit, "add_CZ_gate")(qubit_map[gate.control[0]], qubit_map[gate.target[0]])

    def add_rotation_gate(self, gate, qubit_map, circuit, *args, **kwargs):
        getattr(circuit, "add_" + gate.name.upper() + "_gate")(qubit_map[gate.target[0]], -gate.angle())

    def add_controlled_rotation_gate(self, gate, qubit_map, circuit, *args, **kwargs):
        raise OpenVQEQulacsException("No controlled rotation supported")

    def add_power_gate(self, gate, qubit_map, circuit, *args, **kwargs):
        if gate.power() == 1:
            return self.add_gate(gate=gate, qubit_map=qubit_map, circuit=circuit)
        elif gate.power() == 0.5:
            getattr(circuit, "add_sqrt" + gate.name.upper() + "_gate")(qubit_map[gate.target[0]])
        else:
            raise OpenVQEQulacsException("Only sqrt gates supported as power gates")

    def add_controlled_power_gate(self, gate, qubit_map, circuit, *args, **kwargs):
        raise OpenVQEQulacsException("no controlled power gates")

    def add_measurement(self, gate, qubit_map, circuit, *args, **kwargs):
        raise OpenVQEQulacsException("only full wavefunction simulation, no measurements")

    def make_qubit_map(self, abstract_circuit: QCircuit):
        qubit_map = dict()
        for i,q in enumerate(abstract_circuit.qubits):
            qubit_map[q] = i
        return qubit_map



class SimulatorQulacs(Simulator):

    numbering: BitNumbering = BitNumbering.LSB

    backend_handler = BackenHandlerQulacs()

    def do_simulate_wavefunction(self, abstract_circuit: QCircuit, initial_state=0):
        circuit = self.create_circuit(abstract_circuit=abstract_circuit)

        qubits = dict()
        count = 0
        for q in abstract_circuit.qubits:
            qubits[q] = count
            count += 1

        n_qubits = len(abstract_circuit.qubits)

        state = qulacs.QuantumState(n_qubits)
        state.set_computational_basis(initial_state)
        circuit.update_quantum_state(state)

        wfn = QubitWaveFunction.from_array(arr=state.get_vector(), numbering=self.numbering)
        return SimulatorReturnType(backend_result=state, wavefunction=wfn, circuit=circuit, abstract_circuit=abstract_circuit)

