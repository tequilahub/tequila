from tequila.simulators.simulatorbase import QCircuit, TequilaException, BackendCircuit, BackendExpectationValue
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila import BitString, BitNumbering
import subprocess
import sys

import pyquil


class TequilaPyquilException(TequilaException):
    def __str__(self):
        return "simulator_pyquil: " + self.message


class BackendCircuitPyquil(BackendCircuit):
    recompile_swap = True
    recompile_multitarget = True
    recompile_controlled_rotation = False
    recompile_exponential_pauli = True
    recompile_trotter = True
    recompile_phase = False

    numbering = BitNumbering.LSB

    def do_simulate(self, variables, initial_state, *args, **kwargs):
        simulator = pyquil.api.WavefunctionSimulator()
        n_qubits = self.n_qubits
        msb = BitString.from_int(initial_state, nbits=n_qubits)
        iprep = pyquil.Program()
        for i, val in enumerate(msb.array):
            if val > 0:
                iprep += pyquil.gates.X(i)

        with open('qvm.log', "a+") as outfile:
            sys.stdout = outfile
            sys.stderr = outfile
            outfile.write("\nSTART SIMULATION: \n")
            outfile.write(str(self.abstract_circuit))
            process = subprocess.Popen(["qvm", "-S"], stdout=outfile, stderr=outfile)
            backend_result = simulator.wavefunction(iprep + self.circuit)
            outfile.write("END SIMULATION: \n")
            process.terminate()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
        return QubitWaveFunction.from_array(arr=backend_result.amplitudes, numbering=self.numbering)

    def fast_return(self, abstract_circuit):
        return isinstance(abstract_circuit, pyquil.Program)

    def initialize_circuit(self, *args, **kwargs):
        return pyquil.Program()

    def add_gate(self, gate, circuit, *args, **kwargs):
        circuit += getattr(pyquil.gates, gate.name.upper())(self.qubit_map[gate.target[0]])

    def add_controlled_gate(self, gate, circuit, *args, **kwargs):
        pyquil_gate = getattr(pyquil.gates, gate.name.upper())(self.qubit_map[gate.target[0]])
        for c in gate.control:
            pyquil_gate = pyquil_gate.controlled(self.qubit_map[c])
        circuit += pyquil_gate

    def add_rotation_gate(self, gate, variables, circuit, *args, **kwargs):
        circuit += getattr(pyquil.gates, gate.name.upper())(gate.angle(variables), self.qubit_map[gate.target[0]])

    def add_controlled_rotation_gate(self, gate, variables, circuit, *args, **kwargs):
        pyquil_gate = getattr(pyquil.gates, gate.name.upper())(gate.angle(variables), self.qubit_map[gate.target[0]])
        for c in gate.control:
            pyquil_gate = pyquil_gate.controlled(self.qubit_map[c])
        circuit += pyquil_gate

    def add_power_gate(self, gate, circuit, *args, **kwargs):
        raise TequilaPyquilException("PowerGates are not supported")

    def add_controlled_power_gate(self, gate, circuit, *args, **kwargs):
        raise TequilaPyquilException("controlled PowerGates are not supported")

    def add_measurement(self, gate, circuit, *args, **kwargs):
        bits = len(gate.target)
        ro = circuit.declare('ro', 'BIT', bits)
        for i, t in enumerate(gate.target):
            circuit += pyquil.gates.MEASURE(self.qubit_map[t], ro[i])


class BackendExpectationValuePyquil(BackendExpectationValue):
    BackendCircuitType = BackendCircuitPyquil
