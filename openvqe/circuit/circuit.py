import numpy
import copy


class QGate:

    @staticmethod
    def list_assignement(o):
        """
        Helper function to make initialization with lists and single elements possible
        :param o: iterable object or single element
        :return: Gives back a list if a single element was given
        """
        if o is None:
            return None
        elif hasattr(o, "__get_item__"):
            return o
        elif hasattr(o, "__iter__"):
            return o
        else:
            return [o]

    def __init__(self, name, target: list, control: list = None, angle=None, frozen=None):
        self.name = name
        self.target = self.list_assignement(target)
        self.control = self.list_assignement(control)
        self.angle = angle
        if frozen is None:
            self.frozen = False
        else:
            self.frozen = frozen

    def is_frozen(self):
        return self.frozen

    def make_dagger(self):
        """
        :return: return the hermitian conjugate of the gate
        """
        if self.is_parametrized():
            if self.name in ["Rx", "Ry", "Rz"]:
                return QGate(name=copy.copy(self.name), target=copy.deepcopy(self.target),
                             control=copy.deepcopy(self.control),
                             angle=-1.0 * self.angle)
            else:
                raise Exception("dagger operation for parametrized gates currently only implemented for rotations")
        else:
            return QGate(name=copy.copy(self.name), target=copy.deepcopy(self.target),
                         control=copy.deepcopy(self.control))

    def is_controlled(self) -> bool:
        """
        :return: True if the gate is controlled
        """
        return self.control is not None

    def is_parametrized(self) -> bool:
        """
        :return: True if the gate is parametrized
        """
        return not self.angle is None

    def is_single_qubit_gate(self) -> bool:
        """
        Convenience and easier to interpret
        :return: True if the Gate only acts on one qubit (not controlled)
        """
        return (self.control is None or len(self.control) == 0) and len(self.target) == 1

    def verify(self):
        if not self.is_controlled():
            for c in target:
                if c in self.control:
                    raise Exception("control and target are the same qubit: " + self.__str__())

    def __str__(self):
        result = str(self.name) + "(target=" + str(self.target)
        if not self.is_single_qubit_gate():
            result += ", control=" + str(self.control)
        if self.is_parametrized():
            result += ", angle=" + str(self.angle)
        result += ")"
        return result

    def __repr__(self):
        """
        Todo: Add Nice stringification
        """
        return self.__str__()

    def max_qubit(self):
        """
        :return: Determine maximum qubit index needed
        """
        result = max(self.target)
        if self.control is not None:
            result = max(result, max(self.control))
        return result + 1


class QCircuit:

    def __init__(self, gates=None):
        if gates is None:
            self.gates = []
        else:
            self.gates = gates

    def __getitem__(self, item):
        """
        iteration over gates is possible
        :param item:
        :return: returns the ith gate in the circuit where i=item
        """
        return self.gates[item]

    def __setitem__(self, key: int, value: QGate):
        """
        Insert a gate at a specific position
        :param key:
        :param value:
        :return: self for chaining
        """
        self.gates[key] = value
        return self

    def make_dagger(self):
        """
        :return: Circuit in reverse with signs of rotations switched
        """
        result = QCircuit()
        for g in reversed(self.gates):
            result += g.make_dagger()
        return result

    def insert_gate(self, position: int, gate: QGate):
        if position == "random":
            self.insert_gate_at_random_position(gate=gate)
        else:
            self.gates.insert(position, gate)
        return self

    def insert_gate_at_random_position(self, gate: QGate):
        position = numpy.random.choice(len(self.gates), 1)[0]

        if isinstance(gate, QCircuit):
            for g in gate.gates:
                self.insert_gate(position=position, gate=g)
        else:
            self.insert_gate(position=position, gate=gate)

    def extract_angles(self) -> list:
        """
        Extract the angles from the circuit
        :return: angles as list of tuples where each tuple contains the angle and the position of the gate in the circuit
        """
        angles = []
        for i, g in enumerate(self.gates):
            if g.is_parametrized() and not g.is_frozen():
                angles.append((i, g.angle))

        return angles

    def change_angles(self, angles: list):
        """
        Change angles in of all parametrized gates in circuit
        inplace operation
        :param angles: list of tuples where the first entry is the position of the angle in the circuit and the second the angle itself
        :return: circuit itself for chaining
        """

        try:
            for a in angles:
                position = a[0]
                angle = a[1]
                if not self.gates[position].is_parametrized():
                    raise Exception("You are trying to change the angle of an unparametrized gate\ngate=" + str(
                        self.gates[position]) + "\nangles=(" + str(a) + ")")
                elif self.gates[position].is_frozen():
                    raise Exception("You are trying to change the angle of a frozen gate\ngate=" + str(
                        self.gates[position]) + "\nangles=(" + str(a) + ")")
                else:
                    self.gates[position].angle = angle
        except IndexError as error:
            raise Exception(str(error) + "\nFailed to assign angles, you probably provided to much angles")
        except TypeError as error:
            raise Exception(str(
                error) + "\nFailed to assign angles, you have to give a list of tuples holding position in circuit and angle value")

        return self

    def max_qubit(self):
        qmax = 0
        for g in self.gates:
            qmax = max(qmax, g.max_qubit())
        return qmax

    def __add__(self, other):
        if isinstance(other, QGate):
            other = self.wrap_gate(other)
        result = QCircuit()
        result.gates = (self.gates + other.gates).copy()
        return result

    def __iadd__(self, other):
        if isinstance(other, QGate):
            other = self.wrap_gate(other)
        if isinstance(other, QGate):
            self.gates.append(other)
        else:
            self.gates += other.gates
        return self

    def __str__(self):
        result = "circuit:\n"
        for g in self.gates:
            result += str(g) + "\n"
        return result

    def __repr__(self):
        return self.__str__()

    def make_gradient(self, index):
        """
        :param index: the index should refer to a gate which is parametrized
        :return: returns the gradient with respect to the gate at position self.gates[index]
        """
        if not self.gates[index].is_parametrized():
            raise Exception("You are trying to get the gradient w.r.t a gate which is not parametrized\ngate=" + str(
                self.gates[index]))

        if not self.gates[index].name in ["Rx", "Ry", "Rz"]:
            raise Exception("You are trying to get the gradient w.r.t a gate which is not a rotation\ngate=" + str(
                self.gates[index]))

        gates = self.gates.copy()
        # have to shift by 2 pi/4 because of the factor 2 convention
        # i.e. angle=pi means the gate is exp(i*pi/2*Pauli)
        gates[index].angle += 2 * numpy.pi/4

        return QCircuit(gates=gates)

    @staticmethod
    def wrap_gate(gate: QGate):
        """
        :param gate: Abstract Gate
        :return: wrap gate in QCircuit structure (enable arithmetic operators)
        """
        return QCircuit(gates=[gate])

    def recompile_gates(self, instruction):
        """
        Recompiles gates based on the instruction function
        inplace operation
        :param instruction: has to be callable
        :return: recompiled circuit
        """
        for g in self.gates:
            g = instruction(g)
        return self




# Convenience
def H(target: int, control: int = None):
    return QCircuit(gates=[QGate(name="H", target=target, control=control)])


# Convenience
def S(target: int, control: list = None):
    return QCircuit(gates=[QGate(name="S", target=target, control=control)])


# Convenience
def X(target: int, control: list = None):
    return QCircuit(gates=[QGate(name="X", target=target, control=control)])


# Convenience
def Y(target: int, control: list = None):
    return QCircuit(gates=[QGate(name="Y", target=target, control=control)])


# Convenience
def Z(target: int, control: list = None):
    return QCircuit(gates=[QGate(name="Z", target=target, control=control)])


# Convenience
def I(target: int):
    return QCircuit(gates=[QGate(name="I", target=target)])


# Convenience
def CNOT(control: int, target: int):
    return QCircuit(gates=[QGate(name="CNOT", target=target, control=control)])


# Convenience
def aCNOT(control: int, target: int):
    return QCircuit(gates=[
        QGate(name="X", target=control),
        QGate(name="CNOT", target=target, control=control),
        QGate(name="X", target=control)
    ])


# Convenience
def Rx(target: int, angle: float, control: int = None):
    return QCircuit(gates=[
        QGate(name="Rx", target=target, angle=angle, control=control)
    ])


# Convenience
def Ry(target: int, angle: float, control: int = None):
    return QCircuit(gates=[
        QGate(name="Ry", target=target, angle=angle, control=control)
    ])


# Convenience
def Rz(target: int, angle: float, control: int = None):
    return QCircuit(gates=[
        QGate(name="Rz", target=target, angle=angle, control=control)
    ])


# Convenience
def SWAP(target: list, control: list):
    return QCircuit(gates=[
        QGate(name="SWAP", target=target, control=control)
    ])


# Convenience
def TOFFOLI(target: list, control: list = None):
    return QCircuit(gates=[
        QGate(name="TOFFOLI", target=target, control=control)
    ])


if __name__ == "__main__":
    circuit = Ry(control=0, target=3, angle=numpy.pi / 2)
    circuit += CNOT(control=1, target=0)
    circuit += Ry(control=0, target=1, angle=numpy.pi / 2)
    circuit += aCNOT(control=2, target=0)
    circuit += CNOT(control=3, target=2)
    circuit += Ry(control=0, target=3, angle=numpy.pi / 2)
    circuit += X(0) + X(2)

    print(circuit)

    cr = circuit.make_dagger()

    from openvqe.circuit.circuit_cirq import simulate

    simulate(cr)
