import qulacs
import typing, numbers
from tequila import TequilaException
from tequila.utils.bitstrings import BitNumbering, BitString, BitStringLSB
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila.simulators.simulatorbase import BackendCircuit, BackendExpectationValue

"""
todo: overwrite simulate_objective for this simulators, might be faster

Qulacs uses different Rotational Gate conventions: Rx(angle) = exp(i angle/2 X) instead of exp(-i angle/2 X)
And the same for MultiPauli rotational gates
The angles are scaled with -1.0 to keep things consistent
"""


class TequilaQulacsException(TequilaException):
    def __str__(self):
        return "Error in qulacs backend:" + self.message


class BackendCircuitQulacs(BackendCircuit):
    recompile_swap = False
    recompile_multitarget = True
    recompile_controlled_rotation = True
    recompile_exponential_pauli = False

    numbering = BitNumbering.LSB

    def __init__(self, *args, **kwargs):
        self.variables = []  # the order of this list will be the order of variables in the qulacs circuit
        super().__init__(*args, **kwargs)

    def update_variables(self, variables):
        for k, angle in enumerate(self.variables):
            self.circuit.set_parameter(k, angle(variables))

    def do_simulate(self, variables, initial_state, *args, **kwargs):
        state = qulacs.QuantumState(self.n_qubits)
        lsb = BitStringLSB.from_int(initial_state, nbits=self.n_qubits)
        state.set_computational_basis(BitString.from_binary(lsb.binary).integer)
        self.circuit.update_quantum_state(state)

        wfn = QubitWaveFunction.from_array(arr=state.get_vector(), numbering=self.numbering)
        return wfn

    def fast_return(self, abstract_circuit):
        return False

    def initialize_circuit(self, *args, **kwargs):
        n_qubits = len(self.qubit_map)
        return qulacs.ParametricQuantumCircuit(n_qubits)

    def add_exponential_pauli_gate(self, gate, circuit, variables, *args, **kwargs):
        convert = {'x': 1, 'y': 2, 'z': 3}
        pind = [convert[x.lower()] for x in gate.paulistring.values()]
        qind = [self.qubit_map[x] for x in gate.paulistring.keys()]
        if len(gate.extract_variables()) > 0:
            self.variables.append(-gate.angle * gate.paulistring.coeff)
            circuit.add_parametric_multi_Pauli_rotation_gate(qind, pind,
                                                             -gate.angle(variables) * gate.paulistring.coeff)
        else:
            circuit.add_multi_Pauli_rotation_gate(qind, pind, -gate.angle(variables) * gate.paulistring.coeff)

    def add_gate(self, gate, circuit, *args, **kwargs):
        getattr(circuit, "add_" + gate.name.upper() + "_gate")(self.qubit_map[gate.target[0]])

    def add_controlled_gate(self, gate, circuit, *args, **kwargs):
        if len(gate.control) == 1 and gate.name.upper() == "X":
            getattr(circuit, "add_CNOT_gate")(self.qubit_map[gate.control[0]], self.qubit_map[gate.target[0]])
        elif len(gate.control) == 1 and gate.name.upper() == "Z":
            getattr(circuit, "add_CZ_gate")(self.qubit_map[gate.control[0]], self.qubit_map[gate.target[0]])
        else:
            try:
                qulacs_gate = getattr(qulacs.gate, gate.name.upper())(self.qubit_map[gate.target[0]])
                qulacs_gate = qulacs.gate.to_matrix_gate(qulacs_gate)
                for c in gate.control:
                    qulacs_gate.add_control_qubit(self.qubit_map[c], 1)
                circuit.add_gate(qulacs_gate)
            except:
                raise TequilaQulacsException("Qulacs does not know the controlled gate: " + str(gate))

    def add_rotation_gate(self, gate, variables, circuit, *args, **kwargs):
        # minus sign due to different conventions in qulacs
        if len(gate.extract_variables()) > 0:
            self.variables.append(-gate.angle)
            getattr(circuit, "add_parametric_" + gate.name.upper() + "_gate")(self.qubit_map[gate.target[0]],
                                                                              -gate.angle(variables=variables))
        else:
            getattr(circuit, "add_" + gate.name.upper() + "_gate")(self.qubit_map[gate.target[0]],
                                                                   -gate.angle(variables=variables))

    def add_controlled_rotation_gate(self, gate, variables, circuit, *args, **kwargs):
        angle = -gate.angle(variables=variables)
        qulacs_gate = getattr(qulacs.gate, gate.name.upper())(self.qubit_map[gate.target[0]], angle)
        qulacs_gate = qulacs.gate.to_matrix_gate(qulacs_gate)
        for c in gate.control:
            qulacs_gate.add_control_qubit(self.qubit_map[c], 1)

        if len(gate.extract_variables()) > 0:
            self.variables.append(-gate.angle)
            raise TequilaQulacsException("Parametric controlled-rotations are currently not possible")

        circuit.add_gate(qulacs_gate)

    def add_power_gate(self, gate, variables, circuit, *args, **kwargs):
        assert (len(gate.extract_variables()) == 0)
        power = gate.power(variables=variables)

        if power == 1:
            return self.add_gate(gate=gate, qubit_map=self.qubit_map, circuit=circuit)
        elif power == 0.5:
            getattr(circuit, "add_sqrt" + gate.name.upper() + "_gate")(self.qubit_map[gate.target[0]])
        else:
            raise TequilaQulacsException("Only sqrt gates supported as power gates")

    def add_controlled_power_gate(self, gate, qubit_map, circuit, *args, **kwargs):
        raise TequilaQulacsException("no controlled power gates")

    def add_measurement(self, gate, circuit, *args, **kwargs):
        raise TequilaQulacsException(
            "only full wavefunction simulation, no measurements. Did you forget to add the number of samples?")

    def optimize_circuit(self, circuit, max_block_size: int = None, silent: bool = True, *args, **kwargs):
        """
        Can be overwritten if the backend supports its own circuit optimization
        To be clear: Optimization means optimizing the compiled circuit w.r.t depth not
        optimizing parameters
        :return: Optimized circuit
        """

        # as far as I interpret it it makes most sense to set the block_size to the number of
        # available threads
        if max_block_size is None:
            import os
            max_block_size = int(os.environ.get('OMP_NUM_THREADS'))

        old = circuit.calculate_depth()
        opt = qulacs.circuit.QuantumCircuitOptimizer()
        opt.optimize(circuit, max_block_size)
        if not silent:
            print("qulacs: optimized circuit depth from {} to {} with max_block_size {}".format(old, circuit.calculate_depth(), max_block_size))
        return circuit


class BackendExpectationValueQulacs(BackendExpectationValue):
    BackendCircuitType = BackendCircuitQulacs
    use_mapping = True

    def simulate(self, variables, *args, **kwargs):

        # fast return if possible
        if self.H is None:
            return 0.0
        elif isinstance(self.H, numbers.Number):
            return self.H

        self.update_variables(variables)
        state = qulacs.QuantumState(self.U.n_qubits)
        self.U.circuit.update_quantum_state(state)
        return self.H.get_expectation_value(state)

    def initialize_hamiltonian(self, H):
        if self.use_mapping:
            # initialize only the active parts of the Hamiltonian and pre-evaluate the passive ones
            # passive parts are the components of each individual pauli string which act on qubits where the circuit does not act on
            # if the circuit does not act on those qubits the passive parts are always evaluating to 1 (if the pauli operator is Z) or 0 (otherwise)
            # since those qubits are always in state |0>
            non_zero_strings = []
            unit_strings = 0
            for ps in H.paulistrings:
                string = ""
                for k, v in ps.items():
                    if k in self.U.qubit_map:
                        string += v.upper() + " " + str(self.U.qubit_map[k]) + " "
                    elif v.upper() != "Z":
                        string = "ZERO"
                        break
                string = string.strip()
                if string != "ZERO":
                    non_zero_strings.append((ps.coeff, string))
                elif string == "":
                    unit_strings += 1

            if len(non_zero_strings) == 0:
                return unit_strings
            else:
                assert unit_strings == 0

            qulacs_H = qulacs.Observable(self.n_qubits)
            for coeff, string in non_zero_strings:
                qulacs_H.add_operator(coeff, string)

            return qulacs_H
        else:
            if self.U.n_qubits < H.n_qubits:
                raise TequilaQulacsException(
                    "Hamiltonian has more qubits as the Unitary. Mapped expectationvalues are not yet implemented")

            qulacs_H = qulacs.Observable(self.n_qubits)

            for ps in H.paulistrings:
                string = ""
                for k, v in ps.items():
                    string += v.upper() + " " + str(k)
                qulacs_H.add_operator(ps.coeff, string)
            return qulacs_H
