from openvqe.simulator.simulator import Simulator
from openvqe.circuit.circuit import QCircuit, QGate, Ry, X
from openvqe.tools.convenience import number_to_binary
import copy
import numpy
from numpy import isclose, sqrt


class SymbolicNumber:

    def __init__(self, symbol="", factor=1):
        self.factor = factor
        self.symbol = symbol

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            return SymbolicNumber(factor=self.factor * other, symbol=copy.copy(self.symbol))
        elif self.symbol == "" or self.symbol is None:
            return SymbolicNumber(factor=self.factor * other.factor, symbol=other.symbol)
        else:
            return SymbolicNumber(factor=self.factor * other.factor, symbol=self.symbol + "*" + other.symbol)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __repr__(self):
        if isclose(self.factor, 1.0):
            return str(self.symbol)
        elif isclose(self.factor, 0.0):
            return "0"
        else:
            return str(self.factor) + self.symbol

    def __eq__(self, other):
        if isinstance(other, (int, float, complex)):
            return isclose(self.factor, other)
        elif isclose(self.factor, other.factor):
            s1 = self.symbol.split(sep="*")
            s2 = other.symbol.split(sep="*")
            if len(s1) != len(s2):
                return False
            for x in s1:
                if x not in s2:
                    return False
        else:
            return False

        return True


def apply_function(function: str, number: SymbolicNumber):
    if isinstance(number, SymbolicNumber):
        return SymbolicNumber(symbol=str(function) + "(" + str(number) + ")")
    else:
        tmp = getattr(numpy, function)(number)
        return SymbolicNumber(factor=tmp)


class QBasisState:

    def __init__(self, qubits=None, factor=None):
        if qubits is None:
            self.qubits = []
        else:
            self.qubits = qubits
        if factor is None:
            self.factor = SymbolicNumber()
        else:
            self.factor = factor

    def flip(self, qubit, factor=1):
        qubits = copy.deepcopy(self.qubits)
        qubits[qubit] = (qubits[qubit] + 1) % 2
        return QBasisState(qubits=qubits, factor=self.factor * factor)

    def copy(self, factor=1):
        return QBasisState(qubits=copy.deepcopy(self.qubits), factor=self.factor * factor)

    def __rmul__(self, other):
        return QBasisState(qubits=copy.deepcopy(self.qubits), factor=other * self.factor)

    def __len__(self):
        return len(self.qubits)

    def __repr__(self):
        return str(self.factor) + "|" + str(self.qubits) + ">"


class QState:

    def __init__(self, state=None):
        if isinstance(state, QBasisState):
            self.state = [copy.deepcopy(state)]
        elif state is None:
            self.state = []
        elif state is list:
            self.state = copy.deepcopy(state)
        elif hasattr(state, state):
            # copy constructor
            self.state = copy.deepcopy(state.state)
        else:
            raise Exception("Construction of QState failed")

    def simplify(self):
        simplified = []
        for s in self.state:
            if s.factor == 0.0:
                pass
            else:
                simplified.append(s)

        self.state = simplified

    def __len__(self):
        return len(self.state)

    def n_qubits(self):
        if len(self.state) > 0:
            return len(self.state[0])
        else:
            return 0

    def __radd__(self, other):
        state = copy.deepcopy(self.state)
        state.append(copy.deepcopy(other))
        return QState(state=state)

    def __add__(self, other):
        return QState(state=copy.deepcopy(self.state + other.state))

    def __iadd__(self, other):
        if isinstance(other, QBasisState):
            self.state.append(other)
        else:
            self.state += other.state
        return self

    def __getitem__(self, item):
        return self.state[item]

    def __repr__(self):
        self.simplify()
        result = ""
        for s in self.state:
            result += "+(" + str(s) + ")"
        return result


class SimulatorSymbolic(Simulator):

    @staticmethod
    def apply_gate(state: QState, gate: QGate):
        assert (gate.max_qubit() <= state.n_qubits())
        result = QState()
        for s in state:
            if gate.is_controlled() and s.qubits[gate.control[0]] == 0:
                result += s
            elif gate.name.upper() == "H":
                if s.qubits[gate.target[0]] == 0:
                    result += s.copy(factor=SymbolicNumber(symbol="1/sqrt(2)"))
                    result += s.flip(qubit=gate.target[0], factor=SymbolicNumber(symbol="1/sqrt(2)"))
                else:
                    result += 1.0 / sqrt(2.0) * s.copy()
                    result += -1.0 / sqrt(2.0) * s.flip(qubit=gate.target[0])
            elif gate.name.upper() == "CNOT" or gate.name.upper() == "X":
                result += s.flip(qubit=gate.target[0])
            elif gate.name.upper() == "RX":
                if s.qubits[gate.target[0]] == 0:
                    result += s.copy(factor=apply_function(function="cos", number=0.5 * gate.angle))
                    result += s.flip(qubit=gate.target[0],
                                     factor=1j * apply_function(function="sin", number=0.5 * gate.angle))
                else:
                    result += s.copy(factor=apply_function(function="cos", number=0.5 * gate.angle))
                    result += s.flip(qubit=gate.target[0],
                                     factor=-1j * apply_function(function="sin", number=0.5 * gate.angle))
            elif gate.name.upper() == "RY":
                if s.qubits[gate.target[0]] == 0:
                    result += s.copy(factor=apply_function(function="cos", number=0.5 * gate.angle))
                    result += s.flip(qubit=gate.target[0],
                                     factor=apply_function(function="sin", number=0.5 * gate.angle))
                else:
                    result += s.copy(factor=apply_function(function="cos", number=0.5 * gate.angle))
                    result += s.flip(qubit=gate.target[0],
                                     factor=-1.0 * apply_function(function="sin", number=0.5 * gate.angle))
            elif gate.name.upper() == "RZ":
                if s.qubits[gate.target[0]] == 0:
                    result += s.copy(factor=apply_function(function="cos", number=0.5 * gate.angle))
                    result += s.copy(factor=1.0j * apply_function(function="sin", number=0.5 * gate.angle))
                else:
                    result += s.copy(factor=apply_function(function="cos", number=0.5 * gate.angle))
                    result += s.copy(factor=-1.0j * apply_function(function="sin", number=0.5 * gate.angle))
            else:
                raise Exception("Gate: " + gate.name + " not yet supported in symbolic simulator")

        return result

    def simulate_wavefunction(self, abstract_circuit: QCircuit, initial_state: QState = None):
        n_qubits = abstract_circuit.max_qubit()
        if initial_state is None:
            initial_state = QState(QBasisState(qubits=[0] * n_qubits))
        elif isinstance(initial_state, int):
            initial_state = QState(QBasisState(qubits=number_to_binary(number=initial_state, bits=n_qubits)))

        result = initial_state
        for g in abstract_circuit.gates:
            result = self.apply_gate(state=result, gate=g)
        return result


if __name__ == "__main__":
    circuit = X(0)
    circuit += Ry(target=1, angle=SymbolicNumber(symbol="a"))
    circuit += Ry(target=2, control=1, angle=SymbolicNumber(symbol="b"))

    result = simulate(circuit=circuit)
    print("result=", result)

    circuit = X(0)
    circuit += Ry(target=0, angle=SymbolicNumber(symbol="a"))

    result = simulate(circuit=circuit)
    print("result=", result)

    a = SymbolicNumber(symbol="a")
    b = SymbolicNumber(symbol="b")
    a1 = SymbolicNumber(symbol="a", factor=1.0)
    a2 = SymbolicNumber(symbol="a", factor=2.0)

    print("succes:", False == (a == b))
    print("succes:", True == (a == a1))
    print("succes:", True == (a == 0.5 * a2))
