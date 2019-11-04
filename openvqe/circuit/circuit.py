from openvqe.circuit._gates_impl import QGateImpl
from openvqe import OpenVQEException
from openvqe import BitNumbering
from openvqe import copy
from opnvqe.circuit.variable import Variable,Transform as Variable,Transform

class QCircuit():

    def decompose(self):
        primitives = []
        for g in self.gates:
            if hasattr(g, "decompose"):
                primitives += g.decompose()
            else:
                primitives.append(g)
        return QCircuit(gates=primitives)


    @property
    def parameters(self):
        parameters=[]
        for g in self.gates:
            if g.is_parametrized() and not g.is_frozen():
                if type(g.parameter) is Transform:
                    gpars=g.parameter.variables
                    for p in gpars:
                        if p not in parameters:
                            parameters.append(p)
                elif type(g.parameter) is Variable:
                    parameters.append(g.parameter)

        return parameters
    
    @property
    def numbering(self) -> BitNumbering:
        return BitNumbering.LSB

    @property
    def qubits(self):
        accumulate = []
        for g in self.gates:
            accumulate += g.qubits
        return sorted(list(set(accumulate)))

    @property
    def n_qubits(self):
        if self._n_qubits is not None:
            return max(self.max_qubit()+1,self._n_qubits)
        else:
            return self.max_qubit()+1

    @n_qubits.setter
    def n_qubits(self, other):
        self._n_qubits = other
        if other<self.max_qubit()+1:
            raise OpenVQEException("You are trying to set n_qubits to " + str(other) + " but your circuit needs at least: "+ str(self.max_qubit()+1))
        return self

    @property
    def weight(self):
        if self._weight is None:
            return 1
        else:
            return self._weight

    @weight.setter
    def weight(self, value):
        self._weight = value

    def __init__(self, weight=1.0, gates=None):
        self._n_qubits = None
        if gates is None:
            self.gates = []
        else:
            self.gates = gates
        self._weight = weight

    def is_primitive(self):
        """
        Check if this is a single gate wrapped in this structure
        :return: True if the circuit is just a single gate
        """
        return len(self.gates)

    def __getitem__(self, item):
        """
        iteration over gates is possible
        :param item:
        :return: returns the ith gate in the circuit where i=item
        """
        return self.gates[item]

    def __setitem__(self, key: int, value: QGateImpl):
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
            result *= g.dagger()
        return result

    def extract_parameters(self) -> dict:
        """
        Extract all parameters from the circuit
        :return: List of all unique parameters with names as keys
        TO DO: move away from the dictionary paradigm to the vector paradigm.
        """
        parameters = dict()
        for i, g in enumerate(self.gates):
            if g.is_parametrized() and not g.is_frozen():
                if type(g.parameter) is Transform:
                    pars=g.parameter.variables
                    for par in pars:
                        parameters[par.name] = par.value
                elif type(g.parameter )is Variable:
                    parameters[g.parameter.name] = g.parameter.value
        return parameters

    def update_parameters(self, parameters: dict):
        """
        inplace operation
        :param parameters: a dict of all parameters that shall be updated (order does not matter)
        :return: self for chaining
        TODO: get rid of this
        """
        for p in self.parameters:
            if p.name in parameters:
                p.value=parameters[p.name]
        return self

    def get_indices_for_parameter(self, name: str):
        """
        Lookup all the indices of gates parameterized by a paramter with this name
        :param name: the name of the parameter
        :return: all indices as list
        """
        namex=name
        if hasattr(name, "name"):
            namex=name

        result = []
        for i,g in enumerate(self.gates):
            if g.is_parametrized() and not g.is_frozen() and g.parameter.name == namex:
                result.append(i)
        return result


    def max_qubit(self):
        """
        :return: Maximum index this circuit touches
        """
        qmax = 0
        for g in self.gates:
            qmax = max(qmax, g.max_qubit())
        return qmax

    def __mul__(self, other):
        if isinstance(other, QGateImpl):
            other = self.wrap_gate(other)
        result = QCircuit()
        result.gates = copy.deepcopy(self.gates + other.gates)
        result.weight = self.weight * other.weight
        result._n_qubits = max(max(self.max_qubit()+1,self.n_qubits), max(other.max_qubit()+1,other.n_qubits))
        return result

    def __imul__(self, other):
        if isinstance(other, QGateImpl):
            other = self.wrap_gate(other)

        if isinstance(other, list) and isinstance(other[0], QGateImpl):
            self.gates += other
        else:
            self.gates += other.gates
            self.weight *= other.weight
        self._n_qubits = max(max(self.max_qubit()+1,self.n_qubits), max(other.max_qubit()+1,other.n_qubits))
        return self

    def __rmul__(self, other):
        if isinstance(other, QCircuit):
            return self.__mul__(other)
        if isinstance(other, QGateImpl):
            return self.__mul__(other)
        else:
            return QCircuit(gates=copy.deepcopy(self.gates), weight=self.weight * other)

    def __add__(self, other):
        return self.__mul__(other=other)

    def __iadd__(self, other):
        return self.__imul__(other=other)


    def __pow__(self, power, modulo=None):
        if modulo is not None:
            raise OpenVQEException("Modulo powers for circuits/unitaries not supported")
        if not self.is_primitive():
            raise OpenVQEException("Powers are currently only supported for single gates")

        pgates = []
        for g in self.gates:
            pgates.append(g ** power)
        return QCircuit(gates=pgates, weight=self.weight ** power)

    def __ipow__(self, power, modulo=None):
        if modulo is not None:
            raise OpenVQEException("Modulo powers for circuits/unitaries not supported")
        if not self.is_primitive():
            raise OpenVQEException("Powers are currently only supported for single gates")

        self.weight = self.weight ** power
        for i, g in enumerate(self.gates):
            self.gates[i] **= power
        return self

    def __str__(self):
        result = "circuit: "
        if self.weight != 1.0:
            result += " weight=" + "{:06.2f}".format(self.weight) + " \n"
        else:
            result += "\n"
        for g in self.gates:
            result += str(g) + "\n"
        return result

    def __eq__(self, other):
        if self.weight != other.weight:
            return False
        if len(self.gates) != len(other.gates):
            return False
        for i, g in enumerate(self.gates):
            if g != other.gates[i]:
                return False
        return True

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def wrap_gate(gate: QGateImpl):
        """
        :param gate: Abstract Gate
        :return: wrap gate in QCircuit structure (enable arithmetic operators)
        """
        if isinstance(gate, QCircuit):
            return gate
        else:
            return QCircuit(gates=[gate])


