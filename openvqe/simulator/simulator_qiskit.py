from openvqe.simulator.simulatorbase import SimulatorBase, QCircuit, SimulatorReturnType, BackendHandler
from openvqe.qubit_wavefunction import QubitWaveFunction
from openvqe import OpenVQEException
from openvqe.circuit.compiler import compile_multitarget, compile_controlled_rotation
from openvqe.circuit._gates_impl import MeasurementImpl
from openvqe import BitString, BitNumbering, BitStringLSB
import qiskit


class OpenVQEQiskitException(OpenVQEException):
    def __str__(self):
        return "Error in qiskit backend:" + self.message


class BackenHandlerQiskit(BackendHandler):

    recompile_swap = True
    recompile_multitarget = True
    recompile_controlled_rotation = True

    def fast_return(self, abstract_circuit):
        return isinstance(abstract_circuit, qiskit.QuantumCircuit)

    def initialize_circuit(self, qubit_map, *args, **kwargs):
        return qiskit.QuantumCircuit(qubit_map['q'], qubit_map['c'])

    def add_gate(self, gate, circuit, qubit_map, *args, **kwargs):
        if len(gate.target) > 1:
            raise OpenVQEQiskitException("multi targets need to be explicitly recompiled for Qiskit")
        gfunc = getattr(circuit, gate.name.lower())
        gfunc(qubit_map['q'][gate.target[0]])

    def add_controlled_gate(self, gate, qubit_map, circuit, *args, **kwargs):
        if len(gate.target) > 1:
            raise OpenVQEQiskitException("multi targets need to be explicitly recompiled for Qiskit")
        if len(gate.control) == 1:
            gfunc = getattr(circuit, "c" + gate.name.lower())
            gfunc(qubit_map['q'][gate.control[0]], qubit_map['q'][gate.target[0]])
        elif len(gate.control) == 2:
            try:
                gfunc = getattr(circuit, "cc" + gate.name.lower())
            except AttributeError:
                raise OpenVQEQiskitException("Double controls are currenty only supported for CCX in quiskit")
            gfunc(qubit_map['q'][gate.control[0]], qubit_map['q'][gate.control[1]], qubit_map['q'][gate.target[0]])
        else:
            raise OpenVQEQiskitException("More than two control gates currently not supported")

    def add_rotation_gate(self, gate, qubit_map, circuit, *args, **kwargs):
        if len(gate.target) > 1:
            raise OpenVQEQiskitException("multi targets need to be explicitly recompiled for Qiskit")
        gfunc = getattr(circuit, gate.name.lower())
        gfunc(gate.angle(), qubit_map['q'][gate.target[0]])

    def add_controlled_rotation_gate(self, gate, qubit_map, circuit, *args, **kwargs):
        if len(gate.target) > 1:
            raise OpenVQEQiskitException("multi targets need to be explicitly recompiled for Qiskit")
        if len(gate.control) == 1:
            gfunc = getattr(circuit, "c" + gate.name.lower())
            gfunc(gate.angle(), qubit_map['q'][gate.control[0]], qubit_map['q'][gate.target[0]])
        elif len(gate.control) == 2:
            gfunc = getattr(circuit, "cc" + gate.name.lower())
            gfunc(gate.angle(), qubit_map['q'][gate.control[0]], qubit_map['q'][gate.control[1]],
                  qubit_map['q'][gate.target[0]])
        else:
            raise OpenVQEQiskitException("More than two control gates currently not supported")

    def add_power_gate(self, gate, qubit_map, circuit, *args, **kwargs):
        raise OpenVQEQiskitException("PowerGates are not supported")

    def add_controlled_power_gate(self, gate, qubit_map, circuit, *args, **kwargs):
        raise OpenVQEQiskitException("controlled PowerGates are not supported")

    def add_measurement(self, gate, qubit_map, circuit, *args, **kwargs):
        tq = [qubit_map['q'][t] for t in gate.target]
        tc = [qubit_map['c'][t] for t in gate.target]
        circuit.measure(tq, tc)

    def make_qubit_map(self, abstract_circuit: QCircuit):
        n_qubits = abstract_circuit.n_qubits
        q = qiskit.QuantumRegister(n_qubits, "q")
        c = qiskit.ClassicalRegister(n_qubits, "c")
        return {'q': q, 'c': c}


class SimulatorQiskit(SimulatorBase):
    backend_handler = BackenHandlerQiskit()

    @property
    def numbering(self):
        return BitNumbering.LSB

    def do_run(self, circuit: qiskit.QuantumCircuit, samples):
        simulator = qiskit.Aer.get_backend("qasm_simulator")
        return qiskit.execute(experiments=circuit, backend=simulator, shots=samples)

    def do_simulate_wavefunction(self, abstract_circuit: QCircuit, initial_state=0) -> SimulatorReturnType:
        raise OpenVQEQiskitException("Qiskit can (currently) not simulate general wavefunctions")

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
        return {"": result}


if __name__ == "__main__":

    from openvqe.circuit import gates

    simulator = SimulatorQiskit()
    testc = [gates.X(target=0), gates.X(target=1), gates.X(target=1, control=0),
             gates.Rx(target=1, control=0, angle=2.0), gates.Ry(target=1, control=0, angle=2.0),
             gates.Rz(target=1, control=0, angle=2.0)]

    for c in testc:
        c_qiskit = simulator.create_circuit(abstract_circuit=c)
        print(c_qiskit.draw())
