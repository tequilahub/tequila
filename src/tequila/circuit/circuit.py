from tequila.circuit._gates_impl import QGateImpl
from tequila import TequilaException
from tequila import BitNumbering
from tequila.utils import has_variable
import numpy, typing, numbers

class QCircuit():

    @property
    def gates(self):
        if self._gates is None:
            return []
        else:
            return self._gates

    @property
    def parameter_list(self):
        """
        this property is designed to return a list of the variables in a gate, and for the list to be equal in length
        to the number of gates. Unparametrized or Frozen gates will insert None. This can be used to experiment with the behavior
        of the circuit; changes to the object in the list (unless it is deepcopied) will change the object in the gate directly.
        This is an unprotected property and abuse will break it, but if you've come this far, you knew that.
        :returns: list
        """
        parameters = []
        for g in self.gates:
            if g.is_parametrized() and not g.is_frozen():
                if hasattr(g.parameter, 'f'):
                    gpars = g.parameter.parameter_list
                    parameters.append(gpars)
                elif hasattr(g.parameter, '_name') and hasattr(g.parameter, '_value'):
                    parameters.append(g.parameter)
            else:
                parameters.append(None)
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
        return max(self.max_qubit() + 1, self._min_n_qubits)

    @n_qubits.setter
    def n_qubits(self, other):
        self._min_n_qubits = other
        if other < self.max_qubit() + 1:
            raise TequilaException(
                "You are trying to set n_qubits to " + str(other) + " but your circuit needs at least: " + str(
                    self.max_qubit() + 1))
        return self

    def __init__(self, gates=None):

        self._n_qubits = None
        self._min_n_qubits = 0
        if gates is None:
            self._gates = []
        else:
            self._gates = list(gates)

    def validate(self):
        '''
        helper function to ensure consistency between frozen and unfrozen gates; makes sure no Variable
        parametrizes both frozen and unfrozen gates. Reproduces something alike to the parameters attribute, but note that it also looks at frozen gates.
        '''
        pars = dict()
        for i, g in enumerate(self.gates):
            if g.is_parametrized():
                for name, val in g.parameter.variables.items():
                    pars[name] = [0, 0]

        for g in self.gates:
            if g.is_parametrized():
                if g.is_frozen():
                    for k in g.parameter.variables.keys():
                        pars[k][1] = 1
                else:
                    for k in g.parameter.variables.keys():
                        pars[k][0] = 1

        if any([numpy.sum(pars[k]) > 1 for k in pars.keys()]):
            error_string = 'This circuit contains gates depending on a given parameter, some of which are frozen and others not. \n This breaks the gradient! please rebuild your circuit without doing so.'
            raise TequilaException(error_string)

    def is_primitive(self):
        """
        Check if this is a single gate wrapped in this structure
        :return: True if the circuit is just a single gate
        """
        return len(self.gates) == 1

    def replace_gate(self, position, gates):
        if hasattr(gates, '__iter__'):
            gs = gates
        else:
            gs = [gates]

        new = self.gates[:position]
        new.extend(gs)
        new.extend(self.gates[(position + 1):])
        return QCircuit(gates=new)

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

    def extract_variables(self) -> dict:
        """
        return a dict containing all the variable objects contained in any of the gates within the unitary
        including those nested within transforms.
        rtype dict: {parameter.name:parameter.value}
        """
        pars = dict()
        for i, g in enumerate(self.gates):
            if g.is_parametrized():
                pars = {**pars, **g.extract_variables()}
        return pars

    def update_variables(self, variables: dict):
        """
        inplace operation
        :param variables: a dict of all parameters that shall be updated (order does not matter)
        :return: self for chaining
        """
        for g in self.gates:
            if g.is_parametrized():
                g.update_variables(variables)

        return self

    def get_indices_for_parameter(self, name: str):
        """
        Lookup all the indices of gates parameterized by a paramter with this name
        :param name: the name of the parameter
        :return: all indices as list
        """
        namex = name
        if hasattr(name, "name"):
            namex = name.name

        result = []
        for i, g in enumerate(self.gates):
            if g.is_parametrized() and not g.is_frozen() and has_variable(g.parameter,namex):
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

    def __iadd__(self, other):
        if isinstance(other, QGateImpl):
            other = self.wrap_gate(other)

        if isinstance(other, list) and isinstance(other[0], QGateImpl):
            self._gates += other
        else:
            self._gates += other.gates
        self._min_n_qubits = max(self._min_n_qubits, other._min_n_qubits)
        return self

    def __add__(self, other):
        gates = [g.copy() for g in (self.gates + other.gates)]
        result = QCircuit(gates=gates)
        result._min_n_qubits = max(self._min_n_qubits, other._min_n_qubits)
        return result

    def __pow__(self, power, modulo=None):
        if modulo is not None:
            raise TequilaException("Modulo powers for circuits/unitaries not supported")
        if not self.is_primitive():
            raise TequilaException("Powers are currently only supported for single gates")

        pgates = []
        for g in self.gates:
            pgates.append(g ** power)
        return QCircuit(gates=pgates)

    def __str__(self):
        result = "circuit: \n"
        for g in self.gates:
            result += str(g) + "\n"
        return result

    def __eq__(self, other):
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
