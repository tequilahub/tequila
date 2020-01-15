import qulacs
import typing, numbers
from tequila import TequilaException
from tequila.utils.bitstrings import BitNumbering, BitString, BitStringLSB
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila.simulators.simulatorbase import SimulatorBase, SimulatorReturnType
from tequila.simulators.simulatorbase import BackendHandler
from tequila.circuit import QCircuit

"""
todo: overwrite simulate_objective for this simulators, might be faster

Qulacs uses different Rotational Gate conventions: Rx(angle) = exp(i angle/2 X) instead of exp(-i angle/2 X)
"""


class TequilaQulacsException(TequilaException):
    def __str__(self):
        return "Error in qulacs backend:" + self.message


class BackenHandlerQulacs(BackendHandler):
    recompile_swap = False
    recompile_multitarget = True
    recompile_controlled_rotation = False
    recompile_exponential_pauli = True

    def fast_return(self, abstract_circuit):
        False

    def initialize_circuit(self, qubit_map, *args, **kwargs):
        n_qubits = len(qubit_map)
        return qulacs.QuantumCircuit(n_qubits)

    def add_gate(self, gate, circuit, qubit_map, *args, **kwargs):
        getattr(circuit, "add_" + gate.name.upper() + "_gate")(qubit_map[gate.target[0]])

    def add_controlled_gate(self, gate, qubit_map, circuit, *args, **kwargs):
        # assert (len(gate.control) == 1)
        if len(gate.control) == 1 and gate.name.upper() == "X":
            getattr(circuit, "add_CNOT_gate")(qubit_map[gate.control[0]], qubit_map[gate.target[0]])
        elif len(gate.control) == 1 and gate.name.upper() == "Z":
            getattr(circuit, "add_CZ_gate")(qubit_map[gate.control[0]], qubit_map[gate.target[0]])
        else:
            try:
                qulacs_gate = getattr(qulacs.gate, gate.name.upper())(qubit_map[gate.target[0]])
                qulacs_gate = qulacs.gate.to_matrix_gate(qulacs_gate)
                for c in gate.control:
                    qulacs_gate.add_control_qubit(qubit_map[c], 1)
                circuit.add_gate(qulacs_gate)
            except:
                raise TequilaQulacsException("Qulacs does not know the controlled gate: " + str(gate))

    def add_rotation_gate(self, gate, variables, qubit_map, circuit, *args, **kwargs):
        angle = -gate.angle(variables=variables)  # minus sign due to different conventions in qulacs
        getattr(circuit, "add_" + gate.name.upper() + "_gate")(qubit_map[gate.target[0]], angle)

    def add_controlled_rotation_gate(self, gate, variables, qubit_map, circuit, *args, **kwargs):
        angle = -gate.angle(variables=variables)
        qulacs_gate = getattr(qulacs.gate, gate.name.upper())(qubit_map[gate.target[0]], angle)
        qulacs_gate = qulacs.gate.to_matrix_gate(qulacs_gate)
        for c in gate.control:
            qulacs_gate.add_control_qubit(qubit_map[c], 1)
        circuit.add_gate(qulacs_gate)

    def add_power_gate(self, gate, variables, qubit_map, circuit, *args, **kwargs):

        power = gate.power(variables=variables)

        if power == 1:
            return self.add_gate(gate=gate, qubit_map=qubit_map, circuit=circuit)
        elif power == 0.5:
            getattr(circuit, "add_sqrt" + gate.name.upper() + "_gate")(qubit_map[gate.target[0]])
        else:
            raise TequilaQulacsException("Only sqrt gates supported as power gates")

    def add_controlled_power_gate(self, gate, qubit_map, circuit, *args, **kwargs):
        raise TequilaQulacsException("no controlled power gates")

    def add_measurement(self, gate, qubit_map, circuit, *args, **kwargs):
        raise TequilaQulacsException("only full wavefunction simulation, no measurements")

    def make_qubit_map(self, abstract_circuit: QCircuit):
        qubit_map = dict()
        for i, q in enumerate(abstract_circuit.qubits):
            qubit_map[q] = i
        return qubit_map


class SimulatorQulacs(SimulatorBase):
    numbering: BitNumbering = BitNumbering.LSB

    backend_handler = BackenHandlerQulacs()

    def do_simulate_wavefunction(self, abstract_circuit: QCircuit,
                                 variables: typing.Dict[typing.Hashable, numbers.Real], initial_state=0):
        circuit = self.create_circuit(abstract_circuit=abstract_circuit, variables=variables)

        qubits = dict()
        count = 0
        for q in abstract_circuit.qubits:
            qubits[q] = count
            count += 1

        n_qubits = len(abstract_circuit.qubits)
        state = qulacs.QuantumState(n_qubits)
        lsb = BitStringLSB.from_int(initial_state, nbits=n_qubits)
        state.set_computational_basis(BitString.from_binary(lsb.binary).integer)
        circuit.update_quantum_state(state)

        wfn = QubitWaveFunction.from_array(arr=state.get_vector(), numbering=self.numbering)
        return SimulatorReturnType(backend_result=state, wavefunction=wfn, circuit=circuit,
                                   abstract_circuit=abstract_circuit)
