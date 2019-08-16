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

    def get_gradient(self, index):
        """
        :param index: the index should refer to a gate which is parametrized
        :return: returns the tuple of circuitsgradient with respect to the gate at position self.gates[index]
        """
        if not self.gates[index].is_parametrized():
            raise Exception("You are trying to get the gradient w.r.t a gate which is not parametrized\ngate=" + str(
                self.gates[index]))
        elif self.gates[index].is_frozen():
            raise Exception("You are trying to get the gradient w.r.t a gate which is frozen out\ngate=" +str(self.gates[index]))

        list_of_tuple_of_lists=[]
        gate_list = self.gates.copy()

        new_gates=[]
        new_gates += gate_list[:index]


        rebuilt = recompile_gate(gate_list[index])

        for gate_tuple in rebuilt:
            sub_list = new_gates.copy()
            temp=[]
            for gate_set in gate_tuple:
                unit= sub_list+gate_set
                unit+= gate_list[index+1:]
                temp.append(unit)
            list_of_tuple_of_lists.append(tuple(temp))


        circuit_list=[(QCircuit(gates=entry) for entry in tuple_entry) for tuple_entry in list_of_tuple_lists]

        return circuit_list

    @staticmethod
    def wrap_gate(gate: QGate):
        """
        :param gate: Abstract Gate
        :return: wrap gate in QCircuit structure (enable arithmetic operators)
        """
        return QCircuit(gates=[gate])

    def _recompile_core(self,angle,shifted,spot,target=target,control=control):
        '''
        helper function for recursion of recompile_gate.
        '''
        temp=[]
        if spot == 0:
            temp.append(QGate(name="Rz",target=target,angle=-shifted))
            temp.append(QGate(name="CNOT",target=target,control=control))
            temp.append(QGate(name="Rz",target=target,angle=angle))
            temp.append(QGate(name="CNOT",target=target,control=control))
        if spot == 1
            temp.append(QGate(name="Rz",target=target,angle=-angle))
            temp.append(QGate(name="CNOT",target=target,control=control))
            temp.append(QGate(name="Rz",target=target,angle=shifted))
            temp.append(QGate(name="CNOT",target=target,control=control))

        return temp

    def recompile_gate(self, gate):
        """
        Recompiles gates based on the instruction function
        :param gate: the QGate to recompile
        :return: list of tuple of lists of qgates
        """

        outer_list=[]
        target=gate.target
        control=gate.control
        angle=gate.angle
        clone=copy.deepcopy(gate)
        if len(target) >1:
            raise Exception('multi-target gates do not have quadrature decompositions. I beg your forgiveness.')
            ### They may have one if they can be decomposed into single target gates and control gates, which can
            ### then be decomposed further, but we have to deal with this case by case.

        if gate.is_controlled():
            #### do the case by case for controlled gates
            if gate.name in ['Rx','Ry','Rz']:
                g_shift=0.5
                s=np.pi/2
                up_angle= (angle + s)*g_shift
                down_angle= (angle - s)*g_shift
                if gate.name is  'Rx':
                    for spot in [0,1]:
                        inner=[]
                        for ang in [up_angle,down_angle]:
                            temp=[]
                            temp.append(QGate(name="H", target=target, control=None))
                            temp += self._recompile_core(angle*g_shift,ang,spot)
                            temp.append(QGate(name="H", target=target, control=None))
                            inner.append(temp)

                        outer_list.append(tuple(inner))

                elif gate.name is 'Ry':
                    for spot in [0,1]:
                        inner=[]
                        for ang in [up_angle,down_angle]:
                            temp=[]
                            temp.append(QGate(name="Rx", target=target,angle=np.pi/2 control=None))
                            temp += self._recompile_core(angle*g_shift,ang,spot)
                            temp.append(QGate(name="Rx", target=target,angle=-np.pi/2 control=None))
                            inner.append(temp)

                        outer_list.append(tuple(inner))

                elif gate.name is 'Rz':
                    for spot in [0,1]:
                        inner=[]
                        for ang in [up_angle,down_angle]:
                            temp=[]
                            temp += self._recompile_core(angle*g_shift,ang,spot)
                            inner.append(temp)

                        outer_list.append(tuple(inner))
            else:
                raise Exception('non-rotation  gates do not yet have a quadrature decompoition')


        else:
            if gate.name in ['Rx','Ry','Rz']:
                g_shift=1
                s=np.pi/2
                up=copy.deepcopy(gate)
                up.angle= (angle + s)*g_shift

                down=copy.deepcopy(gate)
                down.angle= (angle - s)*g_shift
                outer_list.append(tuple([up,down]))
            #if gate.name in ['X','Y','Z']:
            else
                raise Exception('non-rotation gates do not yet have a quadrature decompoition')

        return outer_list


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
