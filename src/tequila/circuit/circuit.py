from tequila.circuit._gates_impl import QGateImpl
from tequila import TequilaException
from tequila import BitNumbering
import numpy, typing, numbers,copy


def new_moment(qubits):
    mom = {i: None for i in qubits}
    return mom

def moment_pair(qubits):
    return (new_moment(qubits),new_moment(qubits))






class QCircuit():

    @property
    def moments(self):
        table = {i:0 for i in self.qubits}
        moms = []
        moms.append(Moment())
        for g in self.gates:
            qus = g.qubits
            spots=[table[q] for q in qus]

            if max(spots) == len(moms):

                moms.append(Moment([g]))
            else:
                moms[max(spots)].add_gate(g)
            for q in qus:
                table[q] =max(spots)+1
        return moms

    @property
    def canonical_moments(self):
        table_u = {i: 0 for i in self.qubits}
        table_p = {i: 0 for i in self.qubits}
        moms = []
        moms.append((Moment(),Moment()))

        for g in self.gates:
            p=0
            qus = g.qubits
            if g.is_parametrized():
                if hasattr(g.parameter,'extract_variables'):
                    p=1

            if p == 0:
                spots=[table_u[q] for q in qus] + [table_p[q] for q in qus]
                if max(spots) == len(moms):
                    moms.append((Moment([g]),Moment()))
                else:
                    moms[max(spots)][0].add_gate(g)
                for q in qus:
                    table_u[q] = max(spots)+1
                    table_p[q] = max(spots)

            else:
                spots=[max(table_p[q],table_u[q]-1) for q in qus]
                if max(spots) == len(moms):
                    moms.append((Moment(),Moment([g])))
                else:
                    moms[max(spots)][1].add_gate(g)
                for q in qus:
                    table_u[q] = table_p[q]= max(spots)+1

        return moms

    @property
    def old_moments(self):
        all_m = []
        qubits = self.qubits
        all_m.append(new_moment())
        for g in self.gates:
            qus = g.qubits
            found = None
            for i, m in enumerate(all_m[::-1]):
                if any([(m[q] is not None) for q in qus]):
                    found = i
                if found is not None:
                    break
            if found is None:
                for i, q in enumerate(qus):
                    if i == 0 and q in g.target:
                        all_m[0][q] = g
                    elif i != 0 and q in g.target:
                        all_m[0][q] = 'target'
                    elif q in g.control:
                        all_m[0][q] = 'control'
            elif found is 0:
                new = new_moment(qubits)
                for i, q in enumerate(qus):
                    if i == 0 and q in g.target:
                        new[q] = g
                    elif i != 0 and q in g.target:
                        new[q] = 'target'
                    elif q in g.control:
                        new[q] = 'control'
                all_m.append(new)
            else:
                for i, q in enumerate(qus):
                    if i == 0 and q in g.target:
                        all_m[::-1][found - 1][q] = g
                    elif i != 0 and q in g.target:
                        all_m[::-1][found - 1][q] = 'target'
                    elif q in g.control:
                        all_m[::-1][found - 1][q] = 'control'
        return all_m
    @property
    def old_canonical_moments(self):
        all_m = []
        qubits = self.qubits
        all_m.append(moment_pair(qubits))


        for g in self.gates:

            qus = g.qubits
            found_p=None
            found_u=None
            p=0
            if g.is_parametrized():
                if hasattr(g.parameter,'wrap'):
                    p=1
            for i, m in enumerate(all_m[::-1]):
                if any([(m[0][q] is not None) for q in qus]):
                    found_u=i
                if any([(m[1][q] is not None) for q in qus]):
                    found_p = i

                if found_u is not None or found_p is not None:
                    break

            if found_u is None and found_p is None:
                for i, q in enumerate(qus):
                    if i == 0 and q in g.target:
                        all_m[0][p][q] = g
                    elif i != 0 and q in g.target:
                        all_m[0][p][q] = 'target'
                    elif q in g.control:
                        all_m[0][p][q] = 'control'

            elif (found_u is 0 or found_p is 0) and p is 0:
                new = moment_pair(qubits)
                for i, q in enumerate(qus):
                    if i == 0 and q in g.target:
                        new[p][q] = g
                    elif i != 0 and q in g.target:
                        new[p][q] = 'target'
                    elif q in g.control:
                        new[p][q] = 'control'
                all_m.append(new)

            elif found_p is 0 and p is 1:
                new = moment_pair(qubits)
                for i, q in enumerate(qus):
                    if i == 0 and q in g.target:
                        new[p][q] = g
                    elif i != 0 and q in g.target:
                        new[p][q] = 'target'
                    elif q in g.control:
                        new[p][q] = 'control'
                all_m.append(new)

            elif p is 1 and found_p is None:
                for i, q in enumerate(qus):
                    if i == 0 and q in g.target:
                        all_m[::-1][found_u][p][q] = g
                    elif i != 0 and q in g.target:
                        all_m[::-1][found_u][p][q] = 'target'
                    elif q in g.control:
                        all_m[::-1][found_u][p][q] = 'control'

            elif p is 1 and found_p is not None:
                for i, q in enumerate(qus):
                    if i == 0 and q in g.target:
                        all_m[::-1][found_p-1][p][q] = g
                    elif i != 0 and q in g.target:
                        all_m[::-1][found_p-1][p][q] = 'target'
                    elif q in g.control:
                        all_m[::-1][found_p-1][p][q] = 'control'

            elif p is 0 and found_p is None:
                for i, q in enumerate(qus):
                    if i == 0 and q in g.target:
                        all_m[::-1][found_u-1][p][q] = g
                    elif i != 0 and q in g.target:
                        all_m[::-1][found_u-1][p][q] = 'target'
                    elif q in g.control:
                        all_m[::-1][found_u-1][p][q] = 'control'

            elif p is 0 and found_p is not None:
                for i, q in enumerate(qus):
                    if i == 0 and q in g.target:
                        all_m[::-1][found_p-1][p][q] = g
                    elif i != 0 and q in g.target:
                        all_m[::-1][found_p-1][p][q] = 'target'
                    elif q in g.control:
                        all_m[::-1][found_p-1][p][q] = 'control'


        return all_m

    @property
    def depth(self):
        return len(self.moments)

    @property
    def canonical_depth(self):
        return 2*len(self.canonical_moments)

    @property
    def gates(self):
        if self._gates is None:
            return []
        else:
            return self._gates

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
        :return: Circuit in reverse with signs of rotations switched
        """
        result = QCircuit()
        for g in reversed(self.gates):
            result += g.dagger()
        return result

    def extract_variables(self) -> list:
        """
        return a list containing all the variable objects contained in any of the gates within the unitary
        including those nested within transforms.
        """
        variables = []
        for i, g in enumerate(self.gates):
            if g.is_parametrized():
                variables += g.extract_variables()
        return list(set(variables))

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
    @staticmethod
    def from_moments(moments: typing.List):
        c=QCircuit()
        for m in moments:
            c+=m.as_circuit()
        return c

class Moment():
    '''
    the class which represents all operations to be applied at once in a circuit.
    wraps a list of gates with a list of occupied qubits.
    Can be converted directly to a circuit.
    '''
    @property
    def occupations(self):
        occ = []
        for g in self.gates:
            for q in g.qubits:
                if q in occ:
                    raise TequilaException('cannot have doubly occupied qubits, which occurred at qubit {}'.format(q))
                else:
                    occ.append(q)
        return occ

    @property
    def gates(self):
        return self._gates

    def __init__(self, gates: typing.List[QGateImpl] = None,sort=False):
        occ = []
        if gates is not None:
            for g in list(gates):
                for q in g.qubits:
                    if q in occ:
                        raise TequilaException('cannot have doubly occupied qubits, which occurred at qubit {}'.format(str(q)))
                    else:
                        occ.append(q)
            self._gates = gates
        else:
            self._gates=[]
        if sort:
            self.sort_gates()
    def with_gate(self, gate: typing.Union[QCircuit, QGateImpl]):
        prev = self.occupations
        newq=gate.qubits
        overlap = []
        for n in newq:
            if n in prev:
                overlap.append(n)

        gates=copy.deepcopy(self.gates)
        if len(overlap) is 0:
            gates.append(gate)
        else:
            for i,g in enumerate(gates):
                if any([q in overlap for q in g.qubits]):
                    del g
            gates.append(gate)

        return Moment(gates=gates)

    def with_gates(self,gates):
        gl=list(gates)
        first_overlap=[]
        for g in gl:
            for q in g.qubits:
                if q not in first_overlap:
                    first_overlap.append(q)
                else:
                    raise TequilaException('cannot have a moment with multiple operations acting on the same qubit!')


        new=self.with_gate(gl[0])
        for g in gl[1:]:
            new=new.with_gate(g)
        new.sort_gates()
        return new

    def add_gate(self,gate: typing.Union[QCircuit,QGateImpl]):
        prev = self.occupations
        newq=gate.qubits
        for n in newq:
            if n in prev:
                raise TequilaException('cannot add gate {} to moment; qubit {} already occupied.'.format(str(gate),str(n)))

        self._gates.append(gate)
        self.sort_gates()
        return self
    def as_circuit(self):
        c=QCircuit()
        for g in self.gates:
            c+=g
        return c

    def extract_variables(self) -> list:
        """
        return a list containing all the variable objects contained in any of the gates within the unitary
        including those nested within transforms.
        """
        variables = []
        for i, g in enumerate(self.gates):
            if g.is_parametrized():
                variables += g.extract_variables()
        return list(set(variables))

    def sort_gates(self):
        sd={}
        for gate in self.gates:
            q=min(gate.qubits)
            sd[q]=gate
        sl=[sd[k] for k in sorted(sd.keys())]
        self._gates=sl

    def __str__(self):
        result = "Moment: \n"
        for g in self.gates:
            result += str(g) + "\n"
        return result

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not hasattr(other,'gates'):
            return False
        if len(self.gates) != len(other.gates):
            return False
        for i, g in enumerate(self.gates):
            if g != other.gates[i]:
                return False
        return True

    def max_qubit(self):
        """
        :return: Maximum index this moment touches
        """
        qmax = 0
        for g in self.gates:
            qmax = max(qmax, g.max_qubit)
        return qmax

    def __add__(self, other):
        if hasattr(other,'gates'):
            gates = [g.copy() for g in (self.gates + other.gates)]
            result = QCircuit(gates=gates)
            result._min_n_qubits = max(self.as_circuit()._min_n_qubits, other._min_n_qubits)
        else:
            if isinstance(other,QGateImpl):
                self.add_gate(other)
            else:
                raise TequilaException('cannot add moments to types other than QCircuit,Moment,or Gate; recieved summand of type {}'.format(str(type(other))))
        return result
