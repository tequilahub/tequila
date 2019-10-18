from openvqe.simulator.simulator import Simulator, QCircuit, OpenVQEException, \
    SimulatorReturnType
from openvqe.qubit_wavefunction import QubitWaveFunction
from openvqe import BitString, BitNumbering
from openvqe.circuit._gates_impl import MeasurementImpl
import pyquil
import subprocess


class OpenVQEPyquilException(OpenVQEException):
    def __str__(self):
        return "simulator_pyquil: " + self.message

class SimulatorPyquil(Simulator):

    @property
    def numbering(self):
        return BitNumbering.LSB

    def __init__(self, initialize_qvm:bool=True):
        if initialize_qvm:
            self.qvm=subprocess.Popen(["qvm", "-S"])
        else:
            self.qvm=None
    
    def __del__(self):
        if self.qvm is not None:
            self.qvm.terminate()

    def create_circuit(self, abstract_circuit: QCircuit, qubit_map=None,
                       recompile_controlled_rotations=False) -> pyquil.Program:
        """
        If the backend has its own abstract_circuit objects this can be created here
        :param abstract_circuit: The abstract circuit
        :param qubit_map: Maps qubit_map which are integers in the abstract circuit to other integers
        :return: pyquil.program object corresponding to the abstract_circuit
        """

        # fast return
        if isinstance(abstract_circuit, pyquil.Program):
            return abstract_circuit

        # unroll
        abstract_circuit = abstract_circuit.decompose()

        if qubit_map is None:
            n_qubits = abstract_circuit.n_qubits
            qubit_map = [i for i in range(n_qubits)]
        elif not abstract_circuit.n_qubits < len(qubit_map):
            raise OpenVQEException("qubit map does not provide enough qubits")

        result = pyquil.Program()

        for g in abstract_circuit.gates:

            if isinstance(g, MeasurementImpl):
                raise OpenVQEException("Pyquil currently only works for WavefunctionSimulation -> No Measurements")
                continue

            if len(g.target) > 1:
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

    def do_simulate_wavefunction(self, abstract_circuit: QCircuit, initial_state=0):
        try:
            simulator = pyquil.api.WavefunctionSimulator()
            circuit = self.create_circuit(abstract_circuit=abstract_circuit)
            initial_state = BitString.from_int(integer=initial_state)
            iprep = pyquil.Program()
            for i, val in enumerate(initial_state):
                if val > 0:
                    iprep += pyquil.gates.X(i)

            backend_result = simulator.wavefunction(iprep + circuit)
            return SimulatorReturnType(abstract_circuit=abstract_circuit,
                                       circuit=circuit,
                                       backend_result=backend_result,
                                       wavefunction=QubitWaveFunction.from_array(arr=backend_result.amplitudes,
                                                                                 numbering=self.numbering))

        except Exception as e:
            print("\n\n\n!!!!Make sure Rigettis Quantum-Virtual-Machine is running somewhere in the back!!!!\n\n\n")
            raise e
