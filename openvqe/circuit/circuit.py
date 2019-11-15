from openvqe.circuit._gates_impl import QGateImpl
from openvqe import OpenVQEException
from openvqe import BitNumbering
from openvqe import copy
from openvqe.circuit.variable import Variable,Transform,has_variable

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
    def parameter_list(self):
        parameters=[]
        for g in self.gates:
            if g.is_parametrized() and not g.is_frozen():
                if hasattr(g.parameter,'f'):
                    gpars=g.parameter.var_list
                    for p in gpars:
                        if p not in parameters:
                            parameters.append(p)
                elif hasattr(g.parameter,'_name') and hasattr(g.parameter,'_value'):
                    parameters.append(g.parameter)
        return parameters
    
    @property
    def parameters(self):
        parameters = dict()
        for i, g in enumerate(self.gates):
            if g.is_parametrized() and not g.is_frozen():
                if type(g.parameter) is Transform:
                    pars=g.parameter.variables
                    for name,val in pars.items():
                        parameters[name] = val
                elif type(g.parameter )is Variable:
                    parameters[g.parameter.name] = g.parameter.value
        return parameters
    

    @property
    def numbering(self) -> BitNumbering:
        return BitNumbering.LSB

    @property
    def qubits(self):
        accumulate = []
        for g in self.gates:
            accumulate += list(g.qubits)
        return sorted(list(set(accumulate)))

    @property
    def n_qubits(self):
        return max(self.max_qubit()+1,self._min_n_qubits)

    @n_qubits.setter
    def n_qubits(self, other):
        self._min_n_qubits = other
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


    def __init__(self, gates=None,weight=1.0):
        self._n_qubits = None
        self._min_n_qubits = 0
        if gates is None:
            self.gates = []
        else:
            self.gates = list(gates)
        self._weight = weight
        #self.individuate_parameters()
        self.validate()

    def individuate_parameters(self):
        count=0
        for parameter in self.parameters:
            if hasattr(parameter,'is_default'):
                if parameter.is_default:
                    parameter.name= 'v.{}'.format(str(count))
                    count+=1

    def validate(self):
        for k,v in self.parameters.items():
            found_frozen=False
            found_live=False
            for g in self.gates:
                if g.is_parametrized():
                    if has_variable(g.parameter,{k:v}):
                        if g.is_frozen():
                            found_frozen=True
                        else:
                            found_live=True
                if found_frozen and found_live:
                    error_string='This circuit contains gates depending on parameter named {} which are frozen and unfrozen simultaneously. This breaks the gradient! please rebuild your circuit without doing so.'.format(k)
                    raise OpenVQEException(error_string)


    def is_primitive(self):
        """
        Check if this is a single gate wrapped in this structure
        :return: True if the circuit is just a single gate
        """
        return len(self.gates)==1

    def replace_gate(self,position,gates,inplace=False):
        if hasattr(gates,'__iter__'):
            gs=gates
        else:
            gs=[gates]

        new=self.gates[:position]
        new.extend(gs)
        new.extend(self.gates[(position+1):])
        if inplace is False:
            return QCircuit(gates=new,weight=self.weight)
        elif inplace is True:
            self.gates=new

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
        TODO: deprecate, now identical to the parameters property
        """
        parameters = dict()
        for i, g in enumerate(self.gates):
            if g.is_parametrized() and not g.is_frozen():
                if hasattr(g.parameter, "__iter__") or hasattr(g.parameter, "__get_item__"):
                    for parameter in g.parameter:
                        if parameter.name not in parameters:
                            parameters[parameter.name] = parameter.value
                elif g.parameter.name not in parameters:
                    parameters[g.parameter.name] = g.parameter.value

        return parameters

    def update_parameters(self, parameters: dict):
        """
        inplace operation
        :param parameters: a dict of all parameters that shall be updated (order does not matter)
        :return: self for chaining
        """
        for g in self.gates:
            if g.is_parametrized():
                for k,v in parameters.items():
                    if has_variable(g.parameter,k):
                        g.parameter.update({k:v})

        return self

    def get_indices_for_parameter(self, name: str):
        """
        Lookup all the indices of gates parameterized by a paramter with this name
        :param name: the name of the parameter
        :return: all indices as list
        """
        namex=name
        if hasattr(name, "name"):
            namex=name.name

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
            qmax = max(qmax, g.max_qubit)
        return qmax

    def __mul__(self, other):
        result = QCircuit()
        result.gates = [g.copy() for g in self.gates + other.gates]
        result.weight = self.weight * other.weight
        result._min_n_qubits = max(self._min_n_qubits, other._min_n_qubits)
        return result

    def __imul__(self, other):
        if isinstance(other, QGateImpl):
            other = self.wrap_gate(other)

        if isinstance(other, list) and isinstance(other[0], QGateImpl):
            self.gates += other
        else:
            self.gates += other.gates
            self.weight *= other.weight
        self._min_n_qubits = max(self._min_n_qubits, other._min_n_qubits)
        return self

    def __rmul__(self, other):
        if isinstance(other, QCircuit):
            return self.__mul__(other)
        if isinstance(other, QGateImpl):
            return self.__mul__(other)
        else:
            return QCircuit(gates=[g.copy() for g in self.gates], weight=self.weight * other)

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


