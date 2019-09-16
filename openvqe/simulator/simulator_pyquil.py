from openvqe.simulator.simulator import Simulator, QCircuit, OpenVQEException, SimulatorReturnType
from openvqe.tools.convenience import number_to_binary
import pyquil

class OpenVQEPyquilException(OpenVQEException):
    def __str__(self):
        return "simulator_pyquil: "+self.message


class WavefunctionPyquil(SimulatorReturnType):

    def __post_init__(self, result):
        self.wavefunction = result.amplitudes

class SimulatorPyquil(Simulator):

    def create_circuit(self, abstract_circuit: QCircuit, qubit_map=None, recompile_controlled_rotations=False) -> pyquil.Program:
        """
        If the backend has its own abstract_circuit objects this can be created here
        :param abstract_circuit: The abstract circuit
        :param qubit_map: Maps qubit_map which are integers in the abstract circuit to other integers
        :return: pyquil.program object corresponding to the abstract_circuit
        """

        # fast return
        if isinstance(abstract_circuit, pyquil.Program):
            return abstract_circuit

        if qubit_map is None:
            n_qubits = abstract_circuit.n_qubits
            qubit_map = [i for i in range(n_qubits)]
        elif not abstract_circuit.n_qubits < len(qubit_map):
            raise OpenVQEException("qubit map does not provide enough qubits")

        result = pyquil.Program()

        for g in abstract_circuit.gates:

            if len(g.target)>1:
                raise OpenVQEPyquilException("Pyquil backend does not support multiple targets")

            if g.is_parametrized() and g.control is not None and recompile_controlled_rotations:
                # here we need recompilation
                rc = abstract_circuit.compile_controlled_rotation_gate(g)
                result += self.create_circuit(abstract_circuit=rc, qubit_map=qubit_map)
            else:
                gate = None

                if g.name.upper() == "CNOT":
                    gate = (pyquil.gates.CNOT(target=qubit_map[g.target[0]], control=qubit_map[g.control[0]]))
                else:
                    if g.is_parametrized():
                        gate = getattr(pyquil.gates, g.name.upper())(g.angle, qubit_map[g.target[0]])
                    else:
                        gate = getattr(pyquil.gates, g.name.upper())(qubit_map[g.target[0]])

                    if g.control is not None:
                        for t in g.control:
                            gate = gate.controlled(qubit_map[t])

                result += gate
        return result

    def do_simulate_wavefunction(self, circuit, initial_state=0):
        try:
            simulator = pyquil.api.WavefunctionSimulator()

            # need to initialize the initial_state with a circuit
            binary = number_to_binary(initial_state)
            iprep = pyquil.Program()
            for i, val in enumerate(reversed(binary)):
                if val>0:
                    iprep += pyquil.gates.X(i)

            result=SimulatorReturnType(result=simulator.wavefunction(iprep+circuit))
            result.wavefunction = result.result.amplitudes
            return result
        except Exception as e:
            #print(e)
            print("\n\n\n!!!!Make sure Rigettis Quantum-Virtual-Machine is running somewhere in the back!!!!\n\n\n")
            raise e