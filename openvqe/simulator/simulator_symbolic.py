from openvqe.simulator.simulator import Simulator
<<<<<<< HEAD
from openvqe.circuit.circuit import QCircuit, QGateImpl, Ry, X
from openvqe.tools.convenience import number_to_binary
=======
from openvqe.circuit.circuit import QCircuit
from openvqe.circuit.gates import QGate, Ry, X
from openvqe.tools.convenience import number_to_binary, binary_to_number
>>>>>>> master
import copy
import sympy


def apply_function(function, number):
    return function(number)


class QState:

    def __init__(self, state=None, binary_printout=True):
        if state is None:
            self.state = {}
        else:
            self.state = state
        self.binary_printout = binary_printout

    @staticmethod
    def initialize_from_integer(integer):
        state = {integer: sympy.Integer(1.0)}
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
        return len(number_to_binary(number=max(self.state.keys())))

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
                result += "+(" + str(v) + ")|" + str(number_to_binary(number=s, bits=maxq)) + ">"
            else:
                result += "+(" + str(v) + ")|" + str(s) + ">"
        return result


class SimulatorSymbolic(Simulator):

    @staticmethod
<<<<<<< HEAD
    def apply_gate(state: QState, gate: QGateImpl):
        assert (gate.max_qubit() <= state.n_qubits())
=======
    def apply_on_standard_basis(gate: QGate, qubits: int):
        if gate.is_controlled():
            do_apply = True
            check = [qubits[c] == 1 for c in gate.control]
            for c in check:
                if c == False:
                    do_apply = False
            if not do_apply:
                return QState.initialize_from_integer(binary_to_number(qubits))

>>>>>>> master
        result = QState()

        # exceptions
        if gate.name.upper() == "POWSWAP":
            fac1 = sympy.Rational(1 / 2) * (sympy.Integer(1) + sympy.exp(sympy.I * sympy.pi * gate.angle))
            fac2 = sympy.Rational(1 / 2) * (sympy.Integer(1) - sympy.exp(sympy.I * sympy.pi * gate.angle))
            assert(len(gate.target)==2)
            t0 = gate.target[0]
            t1 = gate.target[1]
            current_state = QState.initialize_from_integer(binary_to_number(qubits))
            if qubits[t0] == qubits[t1]:
                return current_state
            altered = copy.deepcopy(qubits)
            altered[t0] = qubits[t1]
            altered[t1] = qubits[t0]
            altered_state = QState.initialize_from_integer(binary_to_number(altered))
            return fac1 * current_state + fac2*altered_state

        for t in gate.target:
            qv = qubits[t]
            altered = copy.deepcopy(qubits)
            altered[t] = (altered[t] + 1) % 2
            current_state = QState.initialize_from_integer(binary_to_number(qubits))
            altered_state = QState.initialize_from_integer(binary_to_number(altered))
            fac1 = None
            fac2 = None
            if gate.name.upper() == "H":
                fac1 = (sympy.Integer(-1) ** qv * sympy.sqrt(sympy.Rational(1 / 2)))
                fac2 = (sympy.sqrt(sympy.Rational(1 / 2)))
            elif gate.name.upper() == "CNOT" or gate.name.upper() == "X":
                fac2 = sympy.Integer(1)
            elif gate.name.upper() == "RX":
                angle = sympy.Rational(1 / 2) * gate.angle
                fac1 = sympy.cos(angle)
                fac2 = -sympy.sin(angle) * sympy.I
            elif gate.name.upper() == "RY":
                angle = sympy.Rational(1 / 2) * gate.angle
                fac1 = sympy.cos(angle)
                fac2 = -sympy.sin(angle) * sympy.Integer(-1) ** qv
            elif gate.name.upper() == "RZ":
                angle = sympy.Rational(1 / 2) * gate.angle
                fac1 = sympy.exp(-angle * sympy.I * sympy.Integer(-1) ** qv)
            elif gate.name.upper() == "RZ":
                angle = sympy.Rational(1 / 2) * gate.angle
                fac1 = sympy.exp(-angle * sympy.I * sympy.Integer(-1) ** qv)
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
    def apply_gate(state: QState, gate: QGate):
        result = QState()
        maxq = max(state.n_qubits(), gate.max_qubit())
        for s, v in state.items():
            result += v * SimulatorSymbolic.apply_on_standard_basis(gate=gate, qubits=number_to_binary(number=s, bits=maxq))
        return result

    def simulate_wavefunction(self, abstract_circuit: QCircuit, initial_state: QState = None):
        n_qubits = abstract_circuit.max_qubit()
        if initial_state is None:
            initial_state = QState(QBasisState(qubits=[0] * n_qubits))
        elif isinstance(initial_state, int):
            initial_state = QState.initialize_from_integer(initial_state)

        result = initial_state
        for g in abstract_circuit.gates:
            result = self.apply_gate(state=result, gate=g)
        return result


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
