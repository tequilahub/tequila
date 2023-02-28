from tequila.simulators.simulator_base import BackendExpectationValue, BackendCircuit
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila.circuit.circuit import QCircuit
from tequila.circuit.gates import QGate
from tequila import BitString
import numpy
import copy
import sympy

"""
Simple Symbolic Simulator for debugging purposes
"""

class BackendCircuitSymbolic(BackendCircuit):

    # compiler instructions
    compiler_arguments = {
        "trotterized": True,
        "swap": True,
        "multitarget": True,
        "controlled_rotation": False,
        "generalized_rotation": True,
        "exponential_pauli": True,
        "controlled_exponential_pauli": True,
        "phase": True,
        "power": True,
        "hadamard_power": True,
        "controlled_power": True,
        "controlled_phase": True,
        "toffoli": True,
        "phase_to_z": True,
        "cc_max": True
    }

    convert_to_numpy = True

    def create_circuit(self, abstract_circuit: QCircuit, variables=None):
        return abstract_circuit

    def update_variables(self, variables):
        pass

    @classmethod
    def apply_gate(cls, state: QubitWaveFunction, gate: QGate, qubits: dict, variables) -> QubitWaveFunction:
        result = QubitWaveFunction()
        n_qubits = len(qubits.keys())
        for s, v in state.items():
            s.nbits = n_qubits
            result += v * cls.apply_on_standard_basis(gate=gate, basisfunction=s, qubits=qubits, variables=variables)
        return result

    @classmethod
    def apply_on_standard_basis(cls, gate: QGate, basisfunction: BitString, qubits:dict, variables) -> QubitWaveFunction:

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
                angle = sympy.Rational(1 / 2) * gate.parameter(variables)
                fac1 = sympy.cos(angle)
                fac2 = -sympy.sin(angle) * sympy.I
            elif gate.name.upper() == "RY":
                angle = -sympy.Rational(1 / 2) * gate.parameter(variables)
                fac1 = sympy.cos(angle)
                fac2 = +sympy.sin(angle) * sympy.Integer(-1) ** (qt + 1)
            elif gate.name.upper() == "RZ":
                angle = sympy.Rational(1 / 2) * gate.parameter(variables)
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

    def do_simulate(self, variables, initial_state: int = None, *args, **kwargs) -> QubitWaveFunction:
        qubits = dict()
        count = 0
        for q in self.abstract_circuit.qubits:
            qubits[q] = count
            count +=1

        n_qubits = len(self.abstract_circuit.qubits)

        if initial_state is None:
            initial_state = QubitWaveFunction.from_int(i=0, n_qubits=n_qubits)
        elif isinstance(initial_state, int):
            initial_state = QubitWaveFunction.from_int(initial_state, n_qubits=n_qubits)

        result = initial_state
        for g in self.abstract_circuit.gates:
            result = self.apply_gate(state=result, gate=g, qubits=qubits, variables=variables)

        wfn = QubitWaveFunction()
        if self.convert_to_numpy:
            for k,v in result.items():
                wfn[k] = complex(v)
        else:
            wfn = result

        return wfn

class BackendExpectationValueSymbolic(BackendExpectationValue):
    BackendCircuitType = BackendCircuitSymbolic
