from openvqe.simulator.simulator import Simulator, QCircuit, SimulatorReturnType
from openvqe.qubit_wavefunction import QubitWaveFunction
from openvqe import OpenVQEException
from openvqe.circuit.gates import MeasurementImpl
from openvqe import BitString, BitNumbering
from openvqe import typing
import cirq



class OpenVQECirqException(OpenVQEException):
    def __str__(self):
        return "Error in cirq backend:" + self.message


class SimulatorCirq(Simulator):

    numbering: BitNumbering = BitNumbering.MSB

    def convert_measurements(self, backend_result: cirq.TrialResult) -> typing.Dict[str, QubitWaveFunction]:
        result = dict()
        for key, value in backend_result.measurements.items():
            counter = QubitWaveFunction()
            for sample in value:
                binary = BitString.from_array(array=sample.astype(int))
                if binary in counter._state:
                    counter._state[binary] += 1
                else:
                    counter._state[binary] = 1
            result[key] = counter
        return result

    def do_run(self, circuit: cirq.Circuit, samples: int = 1) -> cirq.TrialResult:
        return cirq.Simulator().run(program=circuit, repetitions=samples)

    def create_circuit(self, abstract_circuit: QCircuit, qubit_map=None,
                       recompile_controlled_rotations=False) -> cirq.Circuit:
        """
        If the backend has its own abstract_circuit objects this can be created here
        :param abstract_circuit: The abstract circuit
        :param qubit_map: Maps qubit_map which are integers in the abstract circuit to cirq qubit objects
        :param recompile_controlled_rotations: recompiles controlled rotations (i.e. controled parametrized single qubit gates)
        in order to be able to compute gradients
        if not specified LineQubits will created automatically
        :return: cirq.Cirquit object corresponding to the abstract_circuit
        """

        # fast return
        if isinstance(abstract_circuit, cirq.Circuit):
            return abstract_circuit

        # unroll
        abstract_circuit = abstract_circuit.decompose()

        if qubit_map is None:
            n_qubits = abstract_circuit.n_qubits
            qubit_map = [cirq.LineQubit(i) for i in range(n_qubits)]

        result = cirq.Circuit()
        for g in abstract_circuit.gates:
            if isinstance(g, MeasurementImpl):
                qubits = [qubit_map[i] for i in g.target]
                m = cirq.measure(*qubits, key=g.name)
                result.append(m)
            elif g.is_parametrized() and g.control is not None and recompile_controlled_rotations:
                # here we need recompilation
                rc = abstract_circuit.compile_controlled_rotation_gate(g)
                result += self.create_circuit(abstract_circuit=rc, qubit_map=qubit_map)
            else:
                tmp = cirq.Circuit()
                gate = None

                if g.name.upper() == "CNOT":
                    gate = (cirq.CNOT(target=qubit_map[g.target[0]], control=qubit_map[g.control[0]]))
                else:
                    if g.is_parametrized():
                        if hasattr(g, "power"):
                            if g.power == 1.0:
                                gate = getattr(cirq, g.name)
                            else:
                                gate = getattr(cirq, g.name + "PowGate")(exponent=g.power)
                        elif hasattr(g, "angle"):
                            gate = getattr(cirq, g.name)(rads=g.parameter)
                        else:
                            raise OpenVQECirqException("parametrized gate: only supporting power and rotation gates")
                    else:
                        gate = getattr(cirq, g.name)

                    gate = gate.on(*[qubit_map[t] for t in g.target])

                    if g.control is not None:
                        gate = gate.controlled_by(*[qubit_map[t] for t in g.control])

                tmp.append(gate)
                result += tmp
        return result

    def do_simulate_wavefunction(self, abstract_circuit: QCircuit, initial_state=0) -> SimulatorReturnType:
        simulator = cirq.Simulator()
        circuit = self.create_circuit(abstract_circuit=abstract_circuit)
        backend_result = simulator.simulate(program=circuit, initial_state=initial_state)
        return SimulatorReturnType(abstract_circuit=abstract_circuit, circuit=circuit,
                                   wavefunction=QubitWaveFunction.from_array(arr=backend_result.final_state,
                                   numbering=self.numbering),
                                   backend_result=backend_result)

    def do_simulate_density_matrix(self, circuit: cirq.Circuit, initial_state=0):
        simulator = cirq.DensityMatrixSimulator()
        result = SimulatorReturnType(result=simulator.simulate(program=circuit, initial_state=initial_state),
                                     circuit=circuit)
        result.density_matrix = result.backend_result.final_density_matrix
        return result
