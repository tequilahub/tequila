from openvqe.simulator.simulator import Simulator, QCircuit, SimulatorReturnType
from openvqe import Objective
import cirq


class SimulatorCirq(Simulator):

    def expectation_value(self, objective: Objective):
        # this is not how it is supposed to be ... just hacked in order for this part to return something
        from openvqe.tools import expectation_value_cirq
        result = 0.0
        exv = []
        weights = []
        for unitary in objective.unitaries:
            wfn = self.simulate_wavefunction(abstract_circuit=unitary)
            exv.append(
                expectation_value_cirq(hamiltonian=objective.observable(), n_qubits=objective.observable.n_qubits(),
                                       final_state=wfn.wavefunction))
            #weights.append(unitary.weight) not there yet

        return objective.objective_function(values=exv)

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

        if qubit_map is None:
            n_qubits = abstract_circuit.max_qubit()
            qubit_map = [cirq.LineQubit(i) for i in range(n_qubits)]

        result = cirq.Circuit()
        for g in abstract_circuit.gates:

            if g.is_parametrized() and g.control is not None and recompile_controlled_rotations:
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
                        gate = getattr(cirq, g.name)(rads=g.angle)
                    else:
                        gate = getattr(cirq, g.name)

                    gate = gate.on(*[qubit_map[t] for t in g.target])

                    if g.control is not None:
                        gate = gate.controlled_by(*[qubit_map[t] for t in g.control])

                tmp.append(gate)
                result += tmp
        return result

    def do_simulate_wavefunction(self, circuit: cirq.Circuit, initial_state=0):
        simulator = cirq.Simulator()
        result = SimulatorReturnType(result=simulator.simulate(program=circuit, initial_state=initial_state),
                                     circuit=circuit)
        result.wavefunction = result.result.final_state
        return result

    def do_simulate_density_matrix(self, circuit: cirq.Circuit, initial_state=0):
        simulator = cirq.DensityMatrixSimulator()
        result = SimulatorReturnType(result=simulator.simulate(program=circuit, initial_state=initial_state),
                                     circuit=circuit)
        result.density_matrix = result.result.final_density_matrix
        return result
