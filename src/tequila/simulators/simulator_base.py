from tequila.utils import TequilaException, to_float
from tequila.circuit.circuit import QCircuit
from tequila.utils.keymap import KeyMapSubregisterToRegister
from tequila.utils.misc import to_float
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila.circuit.compiler import change_basis
from tequila.circuit.gates import Measurement
from tequila import BitString
from tequila.objective.objective import Variable, format_variable_dictionary
from tequila.circuit import compiler

import numbers, typing, numpy

"""
TODO: Classes are now immutable: 
       - Map the hamiltonian in the very beginning
       - Add additional features from Skylars project
       - Maybe only keep paulistrings and not full hamiltonian types
"""

class BackendCircuit():
    """
    Functions in the end need to be overwritten by specific backend implementation
    Other functions can be overwritten to improve performance
    self.circuit : translated circuit
    self.abstract_circuit: compiled tequila circuit
    """

    # compiler instructions
    recompile_trotter = True
    recompile_swap = False
    recompile_multitarget = True
    recompile_controlled_rotation = False
    recompile_exponential_pauli = True
    recompile_phase = True
    recompile_power = True
    recompile_hadamard_power = True
    recompile_controlled_power = True
    recompile_controlled_phase = True
    recompile_toffoli = False
    recompile_phase_to_z = False
    cc_max = False

    @property
    def n_qubits(self) -> numbers.Integral:
        return len(self.qubit_map)

    @property
    def qubits(self) -> typing.Iterable[numbers.Integral]:
        return tuple(self._qubits)

    def __init__(self, abstract_circuit: QCircuit, variables, noise_model=None,
                 use_mapping=True, optimize_circuit=True, *args, **kwargs):
        self._variables = tuple(abstract_circuit.extract_variables())
        self.use_mapping = use_mapping
        if noise_model is not None:
            self.cc_max = True
            self.recompile_controlled_phase = True
            self.recompile_controlled_rotation = True
            self.recompile_hadamard_power = True
        # compile the abstract_circuit
        c = compiler.Compiler(multitarget=self.recompile_multitarget,
                              multicontrol=False,
                              trotterized=self.recompile_trotter,
                              exponential_pauli=self.recompile_exponential_pauli,
                              controlled_exponential_pauli=True,
                              hadamard_power=self.recompile_hadamard_power,
                              controlled_power=self.recompile_controlled_power,
                              power=self.recompile_power,
                              controlled_phase=self.recompile_controlled_phase,
                              phase=self.recompile_phase,
                              phase_to_z=self.recompile_phase_to_z,
                              toffoli=self.recompile_toffoli,
                              controlled_rotation=self.recompile_controlled_rotation,
                              cc_max=self.cc_max,
                              swap=self.recompile_swap)

        if self.use_mapping:
            qubits = abstract_circuit.qubits
        else:
            qubits = range(abstract_circuit.n_qubits)

        self._qubits = qubits
        self.abstract_qubit_map = {q: i for i, q in enumerate(qubits)}
        self.qubit_map = self.make_qubit_map(qubits)

        compiled = c(abstract_circuit)
        self.abstract_circuit = compiled
        # translate into the backend object
        self.circuit = self.create_circuit(abstract_circuit=compiled, variables=variables)

        if optimize_circuit:
            self.circuit = self.optimize_circuit(circuit=self.circuit)

        self.noise_model = noise_model

    def __call__(self,
                 variables: typing.Dict[Variable, numbers.Real] = None,
                 samples: int = None,
                 *args,
                 **kwargs):
        variables = format_variable_dictionary(variables=variables)
        if self._variables is not None and len(self._variables) > 0:
            if variables is None or set(self._variables) != set(variables.keys()):
                raise TequilaException("BackendCircuit received not all variables. Circuit depends on variables {}, you gave {}".format(self._variables, variables))
        if samples is None:
            return self.simulate(variables=variables, noise_model=self.noise_model, *args, **kwargs)
        else:
            return self.sample(variables=variables, samples=samples, noise_model=self.noise_model, *args, **kwargs)

    def create_circuit(self, abstract_circuit: QCircuit, *args, **kwargs):
        """
        Translates abstract circuits into the specific backend type
        :param abstract_circuit: Abstract circuit to be translated
        :return: translated circuit
        """

        if self.fast_return(abstract_circuit):
            return abstract_circuit

        result = self.initialize_circuit(*args,**kwargs)

        for g in abstract_circuit.gates:
            if g.is_parametrized():
                self.add_parametrized_gate(g, result, *args,**kwargs)
            else:
                if not g.name == 'Measure':
                    self.add_basic_gate(g, result, *args, **kwargs)
                else:
                    self.add_measurement(g, result, *args, **kwargs)
        return result

    def add_parametrized_gate(self, gate, circuit, *args, **kwargs):
        TequilaException("Backend Handler needs to be overwritten for supported simulators")

    def add_basic_gate(self, gate, circuit, *args, **kwargs):
        TequilaException("Backend Handler needs to be overwritten for supported simulators")

    def add_measurement(self,gate, circuit, *args, **kwargs):
        TequilaException("Backend Handler needs to be overwritten for supported simulators")

    def initialize_circuit(self, *args, **kwargs):
        TequilaException("Backend Handler needs to be overwritten for supported simulators")

    def update_variables(self, variables):
        """
        This is the default which just translates the circuit again
        Overwrite in backend if parametrized circuits are supported
        """
        self.circuit = self.create_circuit(abstract_circuit=self.abstract_circuit, variables=variables)

    def simulate(self, variables, initial_state=0, *args, **kwargs) -> QubitWaveFunction:
        """
        Simulate the wavefunction
        :param returntype: specifies how the result should be given back
        :param initial_state: The initial state of the simulation,
        if given as an integer this is interpreted as the corresponding multi-qubit basis state
        :return: The resulting state
        """
        self.update_variables(variables)
        if isinstance(initial_state, BitString):
            initial_state = initial_state.integer
        if isinstance(initial_state, QubitWaveFunction):
            if len(initial_state.keys()) != 1:
                raise TequilaException("only product states as initial states accepted")
            initial_state = list(initial_state.keys())[0].integer

        all_qubits = [i for i in range(self.abstract_circuit.n_qubits)]
        if self.use_mapping:
            active_qubits = self.abstract_circuit.qubits
            # maps from reduced register to full register
            keymap = KeyMapSubregisterToRegister(subregister=active_qubits, register=all_qubits)
        else:
            keymap = KeyMapSubregisterToRegister(subregister=all_qubits, register=all_qubits)

        result = self.do_simulate(variables=variables, initial_state=keymap.inverted(initial_state).integer)
        result.apply_keymap(keymap=keymap, initial_state=initial_state)
        return result

    def sample_paulistring(self, samples: int, paulistring, *args,
                           **kwargs) -> numbers.Real:
        # make basis change and translate to backend
        basis_change = QCircuit()
        not_in_u = []  # all indices of the paulistring which are not part of the circuit i.e. will always have the same outcome
        qubits = []
        for idx, p in paulistring.items():
            if idx not in self.abstract_qubit_map:
                not_in_u.append(idx)
            else:
                qubits.append(idx)
                basis_change += change_basis(target=idx, axis=p)

        # check the constant parts as <0|pauli|0>, can only be 0 or 1
        # so we can do a fast return of one of them is not Z
        for i in not_in_u:
            pauli = paulistring[i]
            if pauli.upper() != "Z":
                return 0.0

        # make measurement instruction
        measure = QCircuit()
        if len(qubits) == 0:
            # no measurement instructions for a constant term as paulistring
            return paulistring.coeff
        else:
            measure += Measurement(target=qubits)
            #measure += Measurement(name=str(paulistring), target=qubits)
            circuit = self.circuit + self.create_circuit(basis_change + measure)
            # run simulators
            counts = self.do_sample(samples=samples, circuit=circuit)
            # compute energy
            E = 0.0
            n_samples = 0
            for key, count in counts.items():
                parity = key.array.count(1)
                sign = (-1) ** parity
                E += sign * count
                n_samples += count
            E = E / samples * paulistring.coeff
            return E

    def sample(self, variables, samples, *args, **kwargs):
        self.update_variables(variables)
        E = 0.0
        if hasattr(self, 'H'):
            for ps in self.H.paulistrings:
                E += self.sample_paulistring(samples=samples, paulistring=ps, *args, **kwargs)
            return E
        else:
            return self.do_sample(samples=samples, circuit=self.circuit)

    def do_sample(self, samples, circuit, noise_model, *args, **kwargs) -> QubitWaveFunction:
        TequilaException("Backend Handler needs to be overwritten for supported simulators")

    # Those functions need to be overwritten:

    def do_simulate(self, variables, initial_state, *args, **kwargs) -> QubitWaveFunction:
        TequilaException("Backend Handler needs to be overwritten for supported simulators")

    def convert_measurements(self, backend_result) -> QubitWaveFunction:
        TequilaException("Backend Handler needs to be overwritten for supported simulators")

    def fast_return(self, abstract_circuit):
        return True

    def make_qubit_map(self, qubits):
        assert (len(self.abstract_qubit_map) == len(qubits))
        return self.abstract_qubit_map

    def optimize_circuit(self, circuit, *args, **kwargs):
        """
        Can be overwritten if the backend supports its own circuit optimization
        To be clear: Optimization means optimizing the compiled circuit w.r.t depth not
        optimizing parameters
        :return: Optimized circuit, if supported by backend, else no action is taken
        """
        return circuit

    def extract_variables(self) -> typing.Dict[str, numbers.Real]:
        result = self.abstract_circuit.extract_variables()
        return result

    @staticmethod
    def _name_variable_objective(objective):
        """
        Name variables in backend consistently for easier readout
        """
        variables = objective.extract_variables()
        if len(variables) == 0:
            if hasattr(variables[0], "transformation"):
                return str("f({})".format(variables[0]))
            else:
                return str(variables[0])
        else:
            variables = tuple(variables)
            return "f({})".format(variables)


class BackendExpectationValue:
    BackendCircuitType = BackendCircuit

    # map to smaller subsystem if there are qubits which are not touched by the circuits,
    # should be deactivated if expectationvalues are computed by the backend since the hamiltonians are currently not mapped
    use_mapping = True

    @property
    def n_qubits(self):
        return self.U.n_qubits

    @property
    def H(self):
        return self._H

    @property
    def U(self):
        return self._U

    def extract_variables(self) -> typing.Dict[str, numbers.Real]:
        result = []
        if self.U is not None:
            result = self.U.extract_variables()
        return result

    def __init__(self, E, variables, noise_model):
        self._U = self.initialize_unitary(E.U, variables, noise_model)
        self._H = self.initialize_hamiltonian(E.H)
        self._abstract_hamiltonians = E.H
        self._variables = E.extract_variables()
        self._contraction = E._contraction
        self._shape = E._shape

    def __call__(self, variables, samples: int = None, *args, **kwargs):

        variables = format_variable_dictionary(variables=variables)
        if self._variables is not None and len(self._variables) > 0:
            if variables is None or (not set(self._variables) <= set(variables.keys())):
                raise TequilaException(
                    "BackendExpectationValue received not all variables. Circuit depends on variables {}, you gave {}".format(
                        self._variables, variables))

        if samples is None:
            data = self.simulate(variables=variables, *args, **kwargs)
        else:
            data = self.sample(variables=variables, samples=samples, *args, **kwargs)

        if self._shape is None and self._contraction is None:
            # this is the default
            return numpy.sum(data)

        if self._shape is not None:
            data = data.reshape(self._shape)
        if self._contraction is None:
            return data
        else:
            return self._contraction(data)

    def initialize_hamiltonian(self, H):
        return tuple(H)

    def initialize_unitary(self, U, variables, noise_model):
        return self.BackendCircuitType(abstract_circuit=U, variables=variables, use_mapping=self.use_mapping,
                                       noise_model=noise_model)

    def update_variables(self, variables):
        self._U.update_variables(variables=variables)

    def sample(self, variables, samples, *args, **kwargs) -> numpy.array:
        self.update_variables(variables)

        result = []
        for H in self.H:
            E = 0.0
            for ps in H.paulistrings:
                E += self.sample_paulistring(samples=samples, paulistring=ps, *args, **kwargs)
            result.append(to_float(E))
        return numpy.asarray(result)

    def simulate(self, variables, *args, **kwargs):
        self.update_variables(variables)
        result = []
        for H in self.H:
            final_E = 0.0
            if self.use_mapping:
                # The hamiltonian can be defined on more qubits as the unitaries
                qubits_h = H.qubits
                qubits_u = self.U.qubits
                all_qubits = list(set(qubits_h) | set(qubits_u) | set(range(self.U.abstract_circuit.max_qubit() + 1)))
                keymap = KeyMapSubregisterToRegister(subregister=qubits_u, register=all_qubits)
            else:
                if H.qubits != self.U.qubits:
                    raise TequilaException(
                        "Can not compute expectation value without using qubit mappings."
                        " Your Hamiltonian and your Unitary do not act on the same set of qubits. "
                        "Hamiltonian acts on {}, Unitary acts on {}".format(
                            H.qubits, self.U.qubits))
                keymap = KeyMapSubregisterToRegister(subregister=self.U.qubits, register=self.U.qubits)
            # TODO inefficient, let the backend do it if possible or interface some library
            simresult = self.U.simulate(variables=variables, *args, **kwargs)
            wfn = simresult.apply_keymap(keymap=keymap)
            final_E += wfn.compute_expectationvalue(operator=H)

            result.append(to_float(final_E))
        return numpy.asarray(result)

    def sample_paulistring(self, samples: int,
                           paulistring,*args,**kwargs) -> numbers.Real:
        return self.U.sample_paulistring(samples=samples, paulistring=paulistring,*args,**kwargs)
