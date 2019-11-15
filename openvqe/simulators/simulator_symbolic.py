from openvqe.simulators.simulatorbase import SimulatorBase, SimulatorReturnType
from openvqe.qubit_wavefunction import QubitWaveFunction
from openvqe.circuit.circuit import QCircuit
from openvqe.circuit.gates import QGate, Ry, X
from openvqe import BitString
import numpy
import copy
import sympy

"""
Simple Symbolic Simulator for debugging purposes
"""

class SimulatorSymbolic(SimulatorBase):

    _convert_to_numpy = False

    def convert_to_numpy(self, value):
        self._convert_to_numpy = value
        return self

    @staticmethod
    def apply_gate(state: QubitWaveFunction, gate: QGate, qubits: dict) -> QubitWaveFunction:
        result = QubitWaveFunction()
        n_qubits = len(qubits.keys())
        for s, v in state.items():
            s.nbits = n_qubits
            result += v * SimulatorSymbolic.apply_on_standard_basis(gate=gate, basisfunction=s, qubits=qubits)
        return result

    @staticmethod
    def apply_on_standard_basis(gate: QGate, basisfunction: BitString, qubits:dict) -> QubitWaveFunction:

        basis_array = basisfunction.array
        if gate.is_controlled():
            do_apply = True
            check = [basis_array[qubits[c]] == 1 for c in gate.control]
            for c in check:
                if not c:
                    do_apply = False
            if not do_apply:
                return QubitWaveFunction.from_int(basisfunction)

        if len(gate.target) > 1:
            raise Exception("Multi-targets not supported for symbolic simulators")

        result = QubitWaveFunction()
        for tt in gate.target:
            t = qubits[tt]
            qt = basis_array[t]
            a_array = copy.deepcopy(basis_array)
            a_array[t] = (a_array[t] + 1) % 2
            current_state = QubitWaveFunction.from_int(basisfunction)
            altered_state = QubitWaveFunction.from_int(BitString.from_array(a_array))

            fac1 = None
            fac2 = None
            if gate.name == "H":
                fac1 = (sympy.Integer(-1) ** qt * sympy.sqrt(sympy.Rational(1 / 2)))
                fac2 = (sympy.sqrt(sympy.Rational(1 / 2)))
            elif gate.name.upper() == "CNOT" or gate.name.upper() == "X":
                fac2 = sympy.Integer(1)
            elif gate.name.upper() == "Y":
                fac2 = sympy.I * sympy.Integer(-1) ** (qt)
            elif gate.name.upper() == "Z":
                fac1 = sympy.Integer(-1) ** (qt)
            elif gate.name.upper() == "RX":
                angle = sympy.Rational(1 / 2) * gate.angle()
                fac1 = sympy.cos(angle)
                fac2 = -sympy.sin(angle) * sympy.I
            elif gate.name.upper() == "RY":
                angle = -sympy.Rational(1 / 2) * gate.angle()
                fac1 = sympy.cos(angle)
                fac2 = +sympy.sin(angle) * sympy.Integer(-1) ** (qt + 1)
            elif gate.name.upper() == "RZ":
                angle = sympy.Rational(1 / 2) * gate.angle()
                fac1 = sympy.exp(-angle * sympy.I * sympy.Integer(-1) ** (qt))
            else:
                raise Exception("Gate is not known to simulators, " + str(gate))

            if fac1 is None or fac1 == 0:
                result += fac2 * altered_state
            elif fac2 is None or fac2 == 0:
                result += fac1 * current_state
            elif fac1 is None and fac2 is None:
                raise Exception("???")
            else:
                result += fac1 * current_state + fac2 * altered_state

        return result

    def do_simulate_wavefunction(self, abstract_circuit: QCircuit, initial_state: int = None) -> SimulatorReturnType:
        abstract_circuit = self.create_circuit(abstract_circuit=abstract_circuit)
        qubits = dict()
        count = 0
        for q in abstract_circuit.qubits:
            qubits[q] = count
            count +=1

        n_qubits = len(abstract_circuit.qubits)

        if initial_state is None:
            initial_state = QubitWaveFunction.from_int(i=0, n_qubits=n_qubits)
        elif isinstance(initial_state, int):
            initial_state = QubitWaveFunction.from_int(initial_state, n_qubits=n_qubits)

        result = initial_state
        for g in abstract_circuit.gates:
            result = self.apply_gate(state=result, gate=g, qubits=qubits)

        wfn = QubitWaveFunction()
        if self._convert_to_numpy:
            for k,v in result.items():
                wfn[k] = numpy.complex(v)
        else:
            wfn = result

        return SimulatorReturnType(backend_result=result, wavefunction=wfn, circuit=abstract_circuit)


if __name__ == "__main__":
    circuit = X(0)
    circuit += Ry(target=1, angle=sympy.Symbol("a"))
    circuit += Ry(target=2, control=1, angle=sympy.Symbol("b"))

    simulator = SimulatorSymbolic()

    result = simulator.simulate_wavefunction(abstract_circuit=circuit)
    print("result=", result)

    circuit = X(0)
    circuit += Ry(target=0, angle=sympy.Symbol("a"))

    result = simulator.simulate_wavefunction(abstract_circuit=circuit)
    print("result=", result)
