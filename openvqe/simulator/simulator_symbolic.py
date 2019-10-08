from openvqe.simulator.simulator import Simulator, SimulatorReturnType, QubitWaveFunction
from openvqe.circuit.circuit import QCircuit
from openvqe.circuit.gates import QGate, Ry, X
from openvqe import BitStringLSB, BitString
import copy
import sympy

"""
Simple Symbolic Simulator for debugging purposes
Works in LSB notation
"""
#todo get rid of QState and replace with QubitWavefunction for consistency, wait for Mahas code


class QState:

    def __init__(self, state=None, binary_printout=True):
        if state is None:
            self.state = {}
        else:
            self.state = state
        self.binary_printout = binary_printout

    def __eq__(self, other):
        if not hasattr(other, "state"):
            return False
        self.simplify()
        other.simplify()
        return self.state == other.state

    @staticmethod
    def initialize_from_integer(integer):
        state = {BitStringLSB.from_int(integer=integer): sympy.Integer(1.0)}
        result = QState(state=state)
        return result

    def simplify(self):
        pops = []
        for k,v in self.state.items():
            self.state[k] = sympy.simplify(v)
            try:
                if v == 0:
                    pops.append(k)
                elif sympy.comp(v, 0, tol=1.e-4):
                    pops.append(k)
            except:
                pass
        for k in pops:
            self.state.pop(k)

        return self

    def __len__(self):
        return len(self.state)

    def n_qubits(self):
        return max(self.state.keys()).nbits

    def __rmul__(self, other):
        for k in self.state.keys():
            self.state[k] *= other
        return self

    def __add__(self, other):
        self.simplify()
        other.simplify()
        result = QState(state=copy.deepcopy(self.state))
        for key, value in other.state.items():
            if key in self.state:
                result.state[key] = self.state[key] + value
            else:
                result.state[key] = value
        return result

    def __iadd__(self, other):
        self.simplify()
        other.simplify()
        for key, value in other.state.items():
            if key in self.state:
                self.state[key] = self.state[key] + value
            else:
                self.state[key] = value
        return self

    def __getitem__(self, item):
        return self.state[item]

    def items(self):
        return self.state.items()

    def __repr__(self):
        self.simplify()
        result = ""
        maxq = self.n_qubits()
        for s, v in self.state.items():
            if self.binary_printout:
                s.nbits=maxq
                result += "+(" + str(v) + ")|" + str(s.binary) + ">"
            else:
                result += "+(" + str(v) + ")|" + str(s.integer) + ">"
        return result

    def inner(self, other):
        result = 0.0
        for s, v in self.state.items():
            if s in other.state:
                result += v*other.state[s]
        return result


class SimulatorSymbolic(Simulator):

    def create_circuit(self, abstract_circuit: QCircuit) -> QCircuit:
        return abstract_circuit

    @staticmethod
    def apply_on_standard_basis(gate: QGate, qubits: BitStringLSB):
        qubits.nbits = gate.max_qubit()+1
        if gate.is_controlled():
            do_apply = True
            check = [qubits.array[c] == 1 for c in gate.control]
            for c in check:
                if not c:
                    do_apply = False
            if not do_apply:
                return QState.initialize_from_integer(qubits.integer)

        result = QState()

        if len(gate.target) >1:
            raise Exception("multi targets do not work yet for symbolicsymulator")
        for t in gate.target:
            qv = qubits.array[t]
            altered = copy.deepcopy(qubits)
            altered[t] = (altered[t] + 1) % 2
            current_state = QState.initialize_from_integer(qubits.integer)
            altered_state = QState.initialize_from_integer(altered.integer)
            fac1 = None
            fac2 = None
            if gate.name.upper() == "H":
                fac1 = (sympy.Integer(-1) ** qv * sympy.sqrt(sympy.Rational(1 / 2)))
                fac2 = (sympy.sqrt(sympy.Rational(1 / 2)))
            elif gate.name.upper() == "CNOT" or gate.name.upper() == "X":
                fac2 = sympy.Integer(1)
            elif gate.name.upper() == "Y":
                fac2 = sympy.I*sympy.Integer(-1)**(qv+1)
            elif gate.name.upper() == "Z":
                fac1 = sympy.Integer(-1)**(qv)
            elif gate.name.upper() == "RX":
                angle = sympy.Rational(1 / 2) * gate.angle
                fac1 = sympy.cos(angle)
                fac2 = -sympy.sin(angle) * sympy.I
            elif gate.name.upper() == "RY":
                angle = -sympy.Rational(1 / 2) * gate.angle
                fac1 = sympy.cos(angle)
                fac2 = +sympy.sin(angle) * sympy.Integer(-1) ** (qv+1)
            elif gate.name.upper() == "RZ":
                angle = sympy.Rational(1 / 2) * gate.angle
                fac1 = sympy.exp(-angle * sympy.I * sympy.Integer(-1) ** (qv))
            else:
                raise Exception("Gate is not known to simulator, " + str(gate))

            if fac1 is None:
                result += fac2 * altered_state
            elif fac2 is None:
                result += fac1 * current_state
            else:
                result += fac1 * current_state + fac2 * altered_state

        return result

    @staticmethod
    def apply_gate(state: QState, gate: QGate) -> QState:
        result = QState()
        maxq = max(state.n_qubits(), gate.max_qubit())
        for s, v in state.items():
            s.nbits=maxq
            result += v * SimulatorSymbolic.apply_on_standard_basis(gate=gate, qubits=s)
        return result

    def do_simulate_wavefunction(self, abstract_circuit: QCircuit, initial_state: int = None) -> SimulatorReturnType:
        if initial_state is None:
            initial_state = QState.initialize_from_integer(0)
        elif isinstance(initial_state, int):
            initial_state = QState.initialize_from_integer(initial_state)

        result = initial_state
        for g in abstract_circuit.gates:
            result = self.apply_gate(state=result, gate=g)
        wfn = QubitWaveFunction()
        for k, v in result.items():
            key = BitString.from_binary(binary=k.binary)
            wfn[key] = v

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
