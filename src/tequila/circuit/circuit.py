from __future__ import annotations
from tequila.circuit._gates_impl import QGateImpl, assign_variable, list_assignment
from tequila.utils.exceptions import TequilaException, TequilaWarning
from tequila.utils.bitstrings import BitNumbering
import typing
import copy
from collections import defaultdict
import warnings

from .qpic import export_to

class QCircuit():
    """
    Fundamental class representing an abstract circuit.

    Attributes
    ----------
    canonical_depth:
        the depth of the circuit, if converted to alternating parametrized and unparametrized layers.
    canonical_moments:
        returns the circuit as a list of Moment objects alternating between parametrized and unparametrized layers.
    depth:
        returns the gate depth of the circuit.
    gates:
        returns the gates in the circuit, as a list.
    moments:
        returns the circuit as a list of Moment objects.
    n_qubits:
        the number of qubits on which the circuit operates.
    numbering:
        returns the numbering convention use by tequila circuits.
    qubits:
        returns a list of qubits acted upon by the circuit.


    Methods
    -------
    make_parameter_map:


    """

    def export_to(self, *args, **kwargs):
        """
        Export to png, pdf, qpic, tex with qpic backend
        Convenience: see src/tequila/circuit/qpic.py - export_to for more
        Parameters
        """
        # this way we allow calling U.export_to("asd.png") instead of having to specify U.export_to(filename="asd.png")
        if "circuit" not in kwargs:
            kwargs["circuit"]=self
        return export_to(*args, **kwargs)

    @property
    def moments(self):
        """
        Divide self into subcircuits representing layers of simultaneous gates. Attempts to minimize gate depth.
        Returns
        -------
        list:
            list of Moment objects.
        """
        table = {i: 0 for i in self.qubits}
        moms = []
        moms.append(Moment())
        for g in self.gates:
            qus = g.qubits
            spots = [table[q] for q in qus]

            if max(spots) == len(moms):

                moms.append(Moment([g]))
            else:
                moms[max(spots)].add_gate(g)
            for q in qus:
                table[q] = max(spots) + 1
        for mom in moms:
            mom.sort_gates()
        return moms

    @property
    def canonical_moments(self):
        """
        Divide self into subcircuits of alternating unparametrized and parametrized layers.
        Returns
        -------
        list of Moment objects.
        """
        table_u = {i: 0 for i in self.qubits}
        table_p = {i: 0 for i in self.qubits}
        moms = []
        moms.append((Moment(), Moment()))

        for g in self.gates:
            p = 0
            qus = g.qubits
            if g.is_parametrized():
                if hasattr(g.parameter, 'extract_variables'):
                    p = 1

            if p == 0:
                spots = [table_u[q] for q in qus] + [table_p[q] for q in qus]
                if max(spots) == len(moms):
                    moms.append((Moment([g]), Moment()))
                else:
                    moms[max(spots)][0].add_gate(g)
                for q in qus:
                    table_u[q] = max(spots) + 1
                    table_p[q] = max(spots)

            else:
                spots = [max(table_p[q], table_u[q] - 1) for q in qus]
                if max(spots) == len(moms):
                    moms.append((Moment(), Moment([g])))
                else:
                    moms[max(spots)][1].add_gate(g)
                for q in qus:
                    table_u[q] = table_p[q] = max(spots) + 1
        noms = []
        for m in moms:
            noms.extend([m[0], m[1]])

        for nom in noms:
            nom.sort_gates()
        return noms

    @property
    def depth(self):
        """
        gate depth of the abstract circuit.
        Returns
        -------
        int: the depth.

        """
        return len(self.moments)

    @property
    def canonical_depth(self):
        """
        gate depth of the abstract circuit in alternating layer form.
        Returns
        -------
        int: depth of the alternating layer form.
        """
        return len(self.canonical_moments)

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
                "You are trying to set n_qubits to " + str(
                    other) + " but your circuit needs at least: " + str(
                    self.max_qubit() + 1))
        return self

    def __init__(self, gates=None, parameter_map=None):
        """
        init
        Parameters
        ----------
        gates:
            (Default value = None)
            the gates to include in the circuit.
        parameter_map:
            (Default value = None)
            mapping to indicate where in the circuit certain parameters appear.
        """
        self._n_qubits = None
        self._min_n_qubits = 0
        if gates is None:
            self._gates = []
        else:
            self._gates = list(gates)

        if parameter_map is None:
            self._parameter_map = self.make_parameter_map()
        else:
            self._parameter_map = parameter_map

    def make_parameter_map(self) -> dict:
        """
        Returns
        -------
            ParameterMap of the circuit: A dictionary with
            keys: variables in the circuit
            values: list of all gates and their positions in the circuit
            e.g. result[Variable("a")] = [(3, Rx), (5, Ry), ...]
        """
        parameter_map = defaultdict(list)
        for idx, gate in enumerate(self.gates):
            if gate.is_parametrized():
                variables = gate.extract_variables()
                for variable in variables:
                    parameter_map[variable] += [(idx, gate)]

        return parameter_map

    def is_primitive(self):
        """
        Check if this is a single gate wrapped in this structure
        :return: True if the circuit is just a single gate
        """
        return len(self.gates) == 1

    def sort_gates(self):
        """
        sort self into subcircuits corresponding to all simultaneous operations, greedily; then reinitialize gates.
        Returns
        -------
        None
        """
        sl = []
        for m in self.moments:
            sd = {}
            for gate in m.gates:
                q = min(gate.qubits)
                sd[q] = gate
            sl.extend([sd[k] for k in sorted(sd.keys())])
        self._gates = sl

    def replace_gates(self, positions: list, circuits: list, replace: list = None):
        """
        Notes
        ----------
        Replace or insert gates at specific positions into the circuit
        at different positions (faster than multiple calls to replace_gate)

        Parameters
        ----------
        positions: list of int:
            the positions at which the gates should be added. Always refer to the positions in the original circuit
        circuits: list or QCircuit:
            the gates to add at the corresponding positions
        replace: list of bool: (Default value: None)
            Default is None which corresponds to all true
            decide if gates shall be replaces or if the new parts shall be inserted without replacement
            if replace[i] = true: gate at position [i] will be replaces by gates[i]
            if replace[i] = false: gates[i] circuit will be inserted at position [i] (beaming before gate previously at position [i])
        Returns
        -------
            new circuit with inserted gates
        """

        assert len(circuits) == len(positions)
        if replace is None:
            replace = [True] * len(circuits)
        else:
            assert len(circuits) == len(replace)

        dataset = zip(positions, circuits, replace)
        dataset = sorted(dataset, key=lambda x: x[0])

        offset = 0
        new_gatelist = self.gates
        for idx, circuit, do_replace in dataset:

            # failsafe
            if hasattr(circuit, "gates"):
                gatelist = circuit.gates
            elif isinstance(circuit, typing.Iterable):
                gatelist = circuit
            else:
                gatelist = [circuit]

            pos = idx + offset
            if do_replace:
                new_gatelist = new_gatelist[:pos] + gatelist + new_gatelist[pos + 1:]
                offset += len(gatelist) - 1
            else:
                new_gatelist = new_gatelist[:pos] + gatelist + new_gatelist[pos:]
                offset += len(gatelist)

        result = QCircuit(gates=new_gatelist)
        result.n_qubits = max(result.n_qubits, self.n_qubits)
        return result

    def insert_gates(self, positions, gates):
        """
        See replace_gates
        """
        return self.replace_gates(positions=positions, circuits=gates, replace=[False] * len(gates))

    def dagger(self):
        """
        Returns
        ------
        QCircuit:
            The adjoint of the circuit
        """
        result = QCircuit()
        for g in reversed(self.gates):
            result += g.dagger()
        return result

    def extract_variables(self) -> list:
        """
        return a list containing all the variable objects contained in any of the gates within the unitary
        including those nested within gates themselves.

        Returns
        -------
        list:
            the variables of the circuit
        """
        return list(self._parameter_map.keys())

    def max_qubit(self):
        """
        Returns:
        int:
             Highest index of qubits in the circuit
        """
        qmax = 0
        for g in self.gates:
            qmax = max(qmax, g.max_qubit)
        return qmax

    def is_fully_parametrized(self):
        """
        Returns
        -------
        bool:
            whether or not all gates in the circuit are paremtrized
        """
        for gate in self.gates:
            if not gate.is_parametrized():
                return False
            else:
                if hasattr(gate, 'parameter'):
                    if not hasattr(gate.parameter, 'wrap'):
                        return False
                    else:
                        continue
                else:
                    continue
        return True

    def is_fully_unparametrized(self):
        """
        Returns
        -------
        bool:
            whether or not all gates in the circuit are unparametrized
        """
        for gate in self.gates:
            if not gate.is_parametrized():
                continue
            else:
                if hasattr(gate, 'parameter'):
                    if not hasattr(gate.parameter, 'wrap'):
                        continue
                    else:
                        return False
                else:
                    return False
        return True

    def is_mixed(self):
        return not (self.is_fully_parametrized() or self.is_fully_unparametrized())

    def __iadd__(self, other):
        other = self.wrap_gate(gate=other)

        offset = len(self.gates)
        for k, v in other._parameter_map.items():
            self._parameter_map[k] += [(x[0] + offset, x[1]) for x in v]

        self._gates += other.gates
        self._min_n_qubits = max(self._min_n_qubits, other._min_n_qubits)

        return self

    def __add__(self, other):
        other = self.wrap_gate(other)
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
        self.sort_gates()
        other.sort_gates()
        for i, g in enumerate(self.gates):
            if g != other.gates[i]:
                return False
        return True

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def wrap_gate(gate: QGateImpl):
        """
        take a gate and return a qcircuit containing only that gate.
        Parameters
        ----------
        gate: QGateImpl
            the gate to wrap in a circuit.

        Returns
        -------
        QCircuit:
            a one gate circuit.
        """
        if isinstance(gate, QCircuit):
            return gate
        if isinstance(gate, list):
            return QCircuit(gates=gate)
        else:
            return QCircuit(gates=[gate])

    def to_networkx(self):
        """
        Turn a given quantum circuit from tequila into graph form via NetworkX
        :param self: tq.gates.QCircuit
        :return: G, a graph in NetworkX with qubits as nodes and gate connections as edges
        """
        # avoiding dependcies (only used here so far)
        import networkx as nx
        G = nx.Graph()
        for q in self.qubits:
            G.add_node(q)
        Gdict = {s: [] for s in self.qubits}
        for gate in self.gates:
            if gate.control:
                for s in gate.control:
                    for t in gate.target:
                        tstr = ''
                        tstr += str(t)
                        target = int(tstr)
                        Gdict[s].append(target)  # add target to key of correlated controls
                for p in gate.target:
                    for r in gate.control:
                        cstr = ''
                        cstr += str(r)
                        control = int(cstr)
                        Gdict[p].append(control)  # add control to key of correlated targets
            else:
                for s in gate.target:
                    for t in gate.target:
                        tstr2 = ''
                        tstr2 += str(t)
                        target2 = int(tstr2)
                        Gdict[s].append(target2)
        lConn = []  # List of connections between qubits
        for a, b in Gdict.items():
            for q in b:
                lConn.append((a, q))
        G.add_edges_from(lConn)
        GPaths = list(nx.connected_components(
            G))  # connections of various qubits, excluding repetitions (ex- (1,3) instead of (1,3) and (3,1))
        GIso = [g for g in GPaths if len(g) == 1]  # list of Isolated qubits
        return G

    @staticmethod
    def from_moments(moments: typing.List):
        """
        build a circuit from Moment objects.

        Parameters
        ----------
        moments: list:
            a list of Moment objects.

        Returns
        -------
        """
        c = QCircuit()
        for m in moments:
            c += m.as_circuit()
        return c

    def verify(self) -> bool:
        """
        make sure self is built properly and of the correct types.
        Returns
        -------
        bool:
            whether or not the circuit is properly constructed.

        """
        for k, v, in self._parameter_map.items():
            test = [self.gates[x[0]] == x[1] for x in v]
            test += [k in self._gates[x[0]].extract_variables() for x in v]
        return all(test)

    def map_qubits(self, qubit_map):
        """

        E.G.  Rx(1)Ry(2) --> Rx(3)Ry(1) with qubit_map = {1:3, 2:1}

        Parameters
        ----------
        qubit_map
            a dictionary which maps old to new qubits

        Returns
        -------
        A new circuit with mapped qubits
        """

        new_gates = [gate.map_qubits(qubit_map) for gate in self.gates]
        # could speed up by applying qubit_map to parameter_map here
        # currently its recreated in the init function
        return QCircuit(gates=new_gates)

    def add_controls(self, control, inpl: typing.Optional[bool] = False) \
            -> typing.Optional[QCircuit]:
        """Depending on the truth value of inpl:
            - return controlled version of self with control as the control qubits if inpl;
            - mutate self so that the qubits in control are added as the control qubits if not inpl.

        Raise ValueError if there any qubits in common between self and control.
        """
        if inpl:
            self._inpl_control_circ(control)
            return self
        else:
            # return self._return_control_circ(control)
            circ = copy.deepcopy(self)
            return circ.add_controls(control, inpl=True)

    def _return_control_circ(self, control) -> QCircuit:
        """Return controlled version of self with control as the control qubits.

        This is not an in-place method, so it DOES NOT mutates self, and only returns a circuit.

        Raise TequilaWarning if there any qubits in common between self and control.
        """
        control = list(set(list_assignment(control)))

        gates = self.gates
        cgates = []

        for gate in gates:
            cgate = copy.deepcopy(gate)

            if cgate.is_controlled():
                control_lst = list(set(list(cgate.control) + list(control)))

                if len(control_lst) < len(gate.control) + len(control):
                    # warnings.warn("Some of the controls {} were already included in the control "
                    #               "of a gate {}.".format(control, gate), TequilaWarning)
                    raise TequilaWarning(f'Some of the controls {control} were already included '
                                         f'in the control of a gate {gate}.')
            else:
                control_lst, not_control = list(control), list()

            # Raise TequilaWarning if target and control are the same qubit
            if any(qubit in control for qubit in not_control):
                # warnings.warn("The target and control {} were the same qubit for a gate {}."
                #               .format(control, gate), TequilaWarning)
                raise TequilaWarning(f'The target for a gate {gate} '
                                     f'and the control list {control_lst} had a common qubit.')

            cgate._control = tuple(control_lst)
            cgate.finalize()
            cgates.append(cgate)

        return QCircuit(gates=cgates)

    def _inpl_control_circ(self, control) -> None:
        """Mutate self such that the qubits in control are added as the control qubits

        This is an in-place method, so it mutates self and doesn't return any value.

        Raise TequilaWarning if there any qubits in common between self and control.
        """
        gates = self.gates
        control = list_assignment(control)

        for gate in gates:
            if gate.is_controlled():
                control_lst = list(set(list(gate.control) + list(control)))

                # Need to check duplicates
                not_control = list(set(qubit for qubit in list(gate.qubits)
                                       if qubit not in list(gate.control)))

                # Raise TequilaWarning if control qubit is duplicated
                if len(control_lst) < len(gate.control) + len(control):
                    # warnings.warn("Some of the controls {} were already included in the control "
                    #               "of a gate {}.".format(control, gate), TequilaWarning)
                    raise TequilaWarning(f'Some of the controls {control} were already included '
                                         f'in the control of a gate {gate}.')
            else:
                control_lst, not_control = list(control), list()

            # Raise TequilaWarning if target and control are the same qubit
            if any(qubit in control for qubit in not_control):
                # warnings.warn("The target and control {} were the same qubit for a gate {}."
                #               .format(control, gate), TequilaWarning)
                raise TequilaWarning(f'The target for a gate {gate} '
                                     f'and the control list {control} had a common qubit.')

            gate._control = tuple(control_lst)
            gate.finalize()

    def map_variables(self, variables: dict, *args, **kwargs):
        """

        Parameters
        ----------
        variables
            dictionary with old variable names as keys and new variable names or values as values
        Returns
        -------
        Circuit with changed variables

        """

        variables = {assign_variable(k): assign_variable(v) for k, v in variables.items()}

        # failsafe
        my_variables = self.extract_variables()
        for k, v in variables.items():
            if k not in my_variables:
                warnings.warn(
                    "map_variables: variable {} is not part of circuit with variables {}".format(k,
                                                                                                 my_variables),
                    TequilaWarning)

        new_gates = [copy.deepcopy(gate).map_variables(variables) for gate in self.gates]

        return QCircuit(gates=new_gates)


class Moment(QCircuit):
    """
    the class which represents a set of simultaneously applicable gates.

    Methods
    -------
    with_gate:
        attempt to add a gate to the moment, replacing any gate it may conflict with.
    with_gates:
        attempt to add multiple gates to the moment, replacing any gates they may conflict with.

    """

    @property
    def moments(self):
        return [self]

    @property
    def canonical_moments(self):
        """
        Break self up into parametrized and unparametrized layers.
        Returns
        -------
        list:
            list of 2 Moments, one of which may be empty.
        """
        mu = []
        mp = []
        for gate in self.gates:
            if not gate.is_parametrized():
                mu.append(gate)
            else:
                if hasattr(gate, 'parameter'):
                    if not hasattr(gate.parameter, 'wrap'):
                        mu.append(gate)
                    else:
                        mp.append(gate)
                else:
                    mp.append(gate)
        return [Moment(mu), Moment(mp)]

    @property
    def depth(self):
        return 1

    @property
    def canonical_depth(self):
        return 2

    def __init__(self, gates: typing.List[QGateImpl] = None, sort=False):
        """
        initialize a moment from gates.
        Parameters
        ----------
        gates: list:
            a list of gates. Can be None.
        sort:
            whether or not to sort the gates into order from lowest to highest qubit designate.
        """
        super().__init__(gates=gates)
        occ = []
        if gates is not None:
            for g in list(gates):
                for q in g.qubits:
                    if q in occ:
                        raise TequilaException(
                            'cannot have doubly occupied qubits, which occurred at qubit {}'.format(
                                str(q)))
                    else:
                        occ.append(q)
        if sort:
            self.sort_gates()

    def with_gate(self, gate: typing.Union[QCircuit, QGateImpl]):
        """
        Add a gate, or else replace some gate with it.

        Parameters
        ----------
        gate:
            the gate to insert into the moment.

        Returns
        -------
        Moment:
            a New moment with the proper gates.

        """
        prev = self.qubits
        newq = gate.qubits
        overlap = []
        for n in newq:
            if n in prev:
                overlap.append(n)

        gates = copy.deepcopy(self.gates)
        if len(overlap) == 0:
            gates.append(gate)
        else:
            for i, g in enumerate(gates):
                if any([q in overlap for q in g.qubits]):
                    del g
            gates.append(gate)

        return Moment(gates=gates)

    def with_gates(self, gates):
        """
        with gate, but on multiple gates.

        Parameters
        ----------
        gates:
            list of QGates

        Returns
        -------
        Moment:
            a new Moment, with the desired gates insert into self.

        """
        gl = list(gates)
        first_overlap = []
        for g in gl:
            for q in g.qubits:
                if q not in first_overlap:
                    first_overlap.append(q)
                else:
                    raise TequilaException(
                        'cannot have a moment with multiple operations acting on the same qubit!')

        new = self.with_gate(gl[0])
        for g in gl[1:]:
            new = new.with_gate(g)
        new.sort_gates()
        return new

    def add_gate(self, gate: typing.Union[QCircuit, QGateImpl]):
        """
        add a gate to the moment.

        Parameters
        ----------
        gate:
            the desired gate.

        Returns
        -------
        Moment
            a new moment, of self + a new gate
        """
        prev = self.qubits
        newq = gate.qubits
        for n in newq:
            if n in prev:
                raise TequilaException(
                    'cannot add gate {} to moment; qubit {} already occupied.'.format(str(gate),
                                                                                      str(n)))

        self._gates.append(gate)
        self.sort_gates()
        return self

    def replace_gate(self, position, gates):
        """
        substitute a gate at a given position with other gates.
        Parameters
        ----------
        position:
            where in the list of gates the gate to be replaced occurs.
        gates:
            the gates to replace the unwanted gate with.

        Returns
        -------
        QCircuit or Moment:
            self, with unwanted gate removed and new gates inserted. May not be a moment.
        """
        if hasattr(gates, '__iter__'):
            gs = gates
        else:
            gs = [gates]

        new = self.gates[:position]
        new.extend(gs)
        new.extend(self.gates[(position + 1):])
        try:
            return Moment(gates=new)
        except:
            return QCircuit(gates=new)

    def as_circuit(self):
        """
        convert back into the unrestricted QCircuit.
        Returns
        -------
        QCircuit:
            a circuit with the same gates as self.
        """
        return QCircuit(gates=self.gates)

    def fully_parametrized(self):
        """
        Todo: Why not just inherit from base?
        Returns
        -------
        bool:
            whether or not EVERY gate in self.gates is parameterized.
        """
        for gate in self.gates:
            if not gate.is_parametrized():
                return False
            else:
                if hasattr(gate, 'parameter'):
                    if not hasattr(gate.parameter, 'wrap'):
                        return False
                    else:
                        continue
                else:
                    continue
        return True

    def __str__(self):
        result = "Moment: "
        for g in self.gates:
            result += str(g) + ", "
        return result

    def __iadd__(self, other):

        new = self.as_circuit()
        if isinstance(other, QGateImpl):
            other = new.wrap_gate(other)

        if isinstance(other, list) and isinstance(other[0], QGateImpl):
            new._gates += other

        if isinstance(other, list) and isinstance(other[0], QCircuit):
            for o in other:
                new._gates += o.gates
        else:
            new._gates += other.gates
        new._min_n_qubits = max(self._min_n_qubits, other._min_n_qubits)
        if new.depth == 1:
            new = Moment(new.gates)
        return new

    def __add__(self, other):
        if isinstance(other, Moment):
            gates = [g.copy() for g in (self.gates + other.gates)]
            result = QCircuit(gates=gates)
            result._min_n_qubits = max(self.as_circuit()._min_n_qubits, other._min_n_qubits)
            if result.depth == 1:
                result = Moment(gates=result.gates)
                result._min_n_qubits = max(self.as_circuit()._min_n_qubits, other._min_n_qubits)
        elif isinstance(other, QCircuit) and not isinstance(other, Moment):
            if not other.is_primitive():
                gates = [g.copy() for g in (self.gates + other.gates)]
                result = QCircuit(gates=gates)
                result._min_n_qubits = max(self.as_circuit()._min_n_qubits, other._min_n_qubits)
            else:
                try:
                    result = self.add_gate(other.gates[0])
                    result._min_n_qubits += len(other.qubits)
                except:
                    result = self.as_circuit() + QCircuit.wrap_gate(other)
                    result._min_n_qubits = max(self.as_circuit()._min_n_qubits,
                                               QCircuit.wrap_gate(other)._min_n_qubits)

        else:
            if isinstance(other, QGateImpl):
                try:
                    result = self.add_gate(other)
                    result._min_n_qubits += len(other.qubits)
                except:
                    result = self.as_circuit() + QCircuit.wrap_gate(other)
                    result._min_n_qubits = max(self.as_circuit()._min_n_qubits,
                                               QCircuit.wrap_gate(other)._min_n_qubits)
            else:
                raise TequilaException(
                    'cannot add moments to types other than QCircuit,Moment,or Gate; recieved summand of type {}'.format(
                        str(type(other))))
        return result

    @staticmethod
    def wrap_gate(gate: QGateImpl):
        """
        Parameters
        ----------
        gate: QGateImpl:
            the gate, to wrap as a moment

        Returns
        -------
        Moment:
            a moment with one gate in it.
        """
        if isinstance(gate, QCircuit):
            return gate
        else:
            return Moment(gates=[gate])

    @staticmethod
    def from_moments(moments: typing.List):
        """
        Raises
        ------
        TequilaException
        """
        raise TequilaException(
            'this method should never be called from Moment. Call from the QCircuit class itself instead.')



def find_unused_qubit(U0: QCircuit = None, U1: QCircuit = None)->int:
    '''
    Function that checks which are the active qubits of two circuits and 
    provides an unused qubit that is not among them. If all qubits are used
    it adds a new one.

    Parameters
    ----------
    U0 : QCircuit, corresponding to the first state.
        
    U1 : QCircuit, corresponding to the second state.

    Returns
    -------
    control_qubit : int
        
    '''
    
    active_qubits = list(set(U0.qubits+U1.qubits))
    # default
    free_qubit = max(active_qubits) + 1
    # see if we can use another one
    for n in range(max(active_qubits)+1):
        if n not in active_qubits:
            free_qubit = n
            break
    assert free_qubit not in active_qubits
    
    return free_qubit
