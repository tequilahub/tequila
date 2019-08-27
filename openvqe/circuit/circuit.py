import numpy
import copy
from openvqe.circuit.gates import *


class QCircuit():

    def __init__(self, weight =1.0,gates=None):
        if gates is None:
            self.gates = []
        else:
            self.gates = gates
        self.weight=weight

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

    def dagger(self):
        """
        Sumner's Fork:
        I have changed this so that the call to dagger is just dagger all the way down.
        :return: Circuit in reverse with signs of rotations switched
        """
        result = QCircuit()
        for g in reversed(self.gates):
            result += g.dagger()
        return result


    def __true_weight__(self):
        '''
        returns the true weight of the circuit, taking in both the weight assigned by the user and and the weight of all the gates
        '''
        pass

    def replace_gate(self,position, gates: list, inplace: bool=False):
        '''
        if inplace=False:
        returns a transformed version of the circuit in which whatever gate was at 'position' is removed and replaced with,
        in sequence, all the gates in in gates.
        else, changes swaps out the gate at position for the gates in gates, but does not return a new object.
        Particularly useful in the post-processing of gate gradients.

        '''
        prior=self.gates[:position]
        new=gates
        posterior=self.gates[position+1:]
        #### note: this is gonna play badly with gates that would be applied simultaneously, I think.)
        new_gates =prior+new+posterior
        if inplace == False:
            return QCircuit(weight=self.weight,gates=new_gates)
        else:
            self.gates=new_gates



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
            if g.is_parametrized():
                if not g.is_frozen():
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
        :return: returns the tuple of circuit gradient with respect to the gate at position self.gates[index]
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


    def gradient(self):
        '''
        this is a preliminary function for getting quantum circuit gradients based on the methods of the non-frozen gates within the circuit itself.
        returns: list of lists of circuits which, when evaluated, are the exact gradient of the circuit w.r.t it's non-frozen parameters.
        '''
        angles=self.extract_angles()
        count=len(angles)
        gradient=[]
        gates=copy.deepcopy(self.gates)
        for i in range(count):
            sub_list=[]
            target=gates[angles[i][0]]
            g_w = target.gradient()
            for weight_gate in g_w:
                new_circuit=copy.deepcopy(self)
                new_circuit.replace_gate(position=angles[i][0],gates=weight_gate['gates'],inplace=True)
                new_circuit.weight*=weight_gate['weight']
                sub_list.append(new_circuit)
            gradient.append(sub_list)

        return gradient


    @staticmethod
    def wrap_gate(gate: QGate):
        """
        :param gate: Abstract Gate
        :return: wrap gate in QCircuit structure (enable arithmetic operators)
        """
        return QCircuit(gates=[gate])

    def _recompile_core(self,angle,shifted,spot,target,control):
        '''
        helper function for recursion of recompile_gate.
        '''
        temp=[]
        if spot == 0:
            temp.append(QGate(name="Rz",target=target,angle=-shifted))
            temp.append(QGate(name="CNOT",target=target,control=control))
            temp.append(QGate(name="Rz",target=target,angle=angle))
            temp.append(QGate(name="CNOT",target=target,control=control))
        if spot == 1:
            temp.append(QGate(name="Rz",target=target,angle=-angle))
            temp.append(QGate(name="CNOT",target=target,control=control))
            temp.append(QGate(name="Rz",target=target,angle=shifted))
            temp.append(QGate(name="CNOT",target=target,control=control))

        return temp

    def recompile_gate(self, gate):
        """
        TODO:
        remove
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
                            temp.append(QGate(name="H", target=gate.target, control=None))
                            temp += self._recompile_core(angle*g_shift,ang,spot)
                            temp.append(QGate(name="H", target=gate.target, control=None))
                            inner.append(temp)

                        outer_list.append(tuple(inner))

                elif gate.name is 'Ry':
                    for spot in [0,1]:
                        inner=[]
                        for ang in [up_angle,down_angle]:
                            temp=[]
                            temp.append(QGate(name="Rx", target=gate.target,angle=np.pi/2, control=None))
                            temp += self._recompile_core(angle*g_shift,ang,spot)
                            temp.append(QGate(name="Rx", target=gate.target,angle=-np.pi/2, control=None))
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
            else:
                raise Exception('non-rotation gates do not yet have a quadrature decompoition')

        return outer_list
