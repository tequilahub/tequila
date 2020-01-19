from tequila.simulators.simulatorbase import BackendCircuit, QCircuit, BackendExpectationValue
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila import TequilaException
from tequila import BitString, BitNumbering, BitStringLSB
import qiskit


class TequilaQiskitException(TequilaException):
    def __str__(self):
        return "Error in qiskit backend:" + self.message


class BackendCircuitQiskit(BackendCircuit):
    recompile_swap = True
    recompile_multitarget = True
    recompile_controlled_rotation = True

    numbering = BitNumbering.LSB

    def do_simulate(self, variables, initial_state=0, *args, **kwargs) -> QubitWaveFunction:
        simulator = qiskit.Aer.get_backend("statevector_simulator")
        if initial_state != 0:
            # need something like this
            # there is a keyword for the backend for tolerance on norm
            # circuit.initialize(normed_array)
            raise TequilaQiskitException("initial state for Qiskit not yet supported here")
        backend_result = qiskit.execute(experiments=self.circuit, backend=simulator).result()
        return QubitWaveFunction.from_array(arr=backend_result.get_statevector(self.circuit), numbering=self.numbering)

    def do_sample(self, circuit: qiskit.QuantumCircuit, samples: int, *args, **kwargs) -> QubitWaveFunction:
        simulator = qiskit.Aer.get_backend("qasm_simulator")
        return self.convert_measurements(qiskit.execute(experiments=circuit, backend=simulator, shots=samples))

    def convert_measurements(self, backend_result) -> QubitWaveFunction:
        """
        :param qiskit_counts: qiskit counts as dictionary, states are binary in little endian (LSB)
        :return: Counts in OpenVQE format, states are big endian (MSB)
        """
        qiskit_counts = backend_result.result().get_counts()
        result = QubitWaveFunction()
        # todo there are faster ways
        for k, v in qiskit_counts.items():
            converted_key = BitString.from_bitstring(other=BitStringLSB.from_binary(binary=k))
            result._state[converted_key] = v
        return result

    def fast_return(self, abstract_circuit):
        return isinstance(abstract_circuit, qiskit.QuantumCircuit)

    def initialize_circuit(self, *args, **kwargs):
        return qiskit.QuantumCircuit(self.qubit_map['q'], self.qubit_map['c'])

    def add_gate(self, gate, circuit, *args, **kwargs):
        if len(gate.target) > 1:
            raise TequilaQiskitException("multi targets need to be explicitly recompiled for Qiskit")
        gfunc = getattr(circuit, gate.name.lower())
        gfunc(self.qubit_map['q'][gate.target[0]])

    def add_controlled_gate(self, gate, circuit, *args, **kwargs):
        if len(gate.target) > 1:
            raise TequilaQiskitException("multi targets need to be explicitly recompiled for Qiskit")
        if len(gate.control) == 1:
            gfunc = getattr(circuit, "c" + gate.name.lower())
            gfunc(self.qubit_map['q'][gate.control[0]], self.qubit_map['q'][gate.target[0]])
        elif len(gate.control) == 2:
            try:
                gfunc = getattr(circuit, "cc" + gate.name.lower())
            except AttributeError:
                raise TequilaQiskitException("Double controls are currenty only supported for CCX in quiskit")
            gfunc(self.qubit_map['q'][gate.control[0]], self.qubit_map['q'][gate.control[1]], self.qubit_map['q'][gate.target[0]])
        else:
            raise TequilaQiskitException("More than two control gates currently not supported")

    def add_rotation_gate(self, gate, variables, circuit, *args, **kwargs):
        if len(gate.target) > 1:
            raise TequilaQiskitException("multi targets need to be explicitly recompiled for Qiskit")
        gfunc = getattr(circuit, gate.name.lower())
        gfunc(gate.angle(variables), self.qubit_map['q'][gate.target[0]])

    def add_controlled_rotation_gate(self, gate, variables, circuit, *args, **kwargs):
        if len(gate.target) > 1:
            raise TequilaQiskitException("multi targets need to be explicitly recompiled for Qiskit")
        if len(gate.control) == 1:
            gfunc = getattr(circuit, "c" + gate.name.lower())
            gfunc(gate.angle(variables), self.qubit_map['q'][gate.control[0]], self.qubit_map['q'][gate.target[0]])
        elif len(gate.control) == 2:
            gfunc = getattr(circuit, "cc" + gate.name.lower())
            gfunc(gate.angle(variables), self.qubit_map['q'][gate.control[0]], self.qubit_map['q'][gate.control[1]],
                  self.qubit_map['q'][gate.target[0]])
        else:
            raise TequilaQiskitException("More than two control gates currently not supported")

    def add_measurement(self, gate, circuit, *args, **kwargs):
        tq = [self.qubit_map['q'][t] for t in gate.target]
        tc = [self.qubit_map['c'][t] for t in gate.target]
        circuit.measure(tq, tc)

    def make_qubit_map(self, abstract_circuit: QCircuit):
        n_qubits = abstract_circuit.n_qubits
        q = qiskit.QuantumRegister(n_qubits, "q")
        c = qiskit.ClassicalRegister(n_qubits, "c")
        return {'q': q, 'c': c}

class BackendExpectationValueQiskit(BackendExpectationValue):
    BackendCircuitType = BackendCircuitQiskit
