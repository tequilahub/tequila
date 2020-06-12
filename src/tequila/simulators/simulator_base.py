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
Todo: Classes are now immutable: 
       - Map the hamiltonian in the very beginning
       - Add additional features from Skylars project
       - Maybe only keep paulistrings and not full hamiltonian types
"""

class BackendCircuit():
    """
    Base class for circuits compiled to run on specific backends.

    Attributes
    ----------
    abstract_circuit:
        the tequila circuit from which the backend circuit is built.
    abstract_qubit_map:
        a dictionary mapping the tequila qubits to a consecutive set.
        eg: {0:0,3:1,53:2}, if the only qubits in the abstract circuit are 0, 3, and 53.
    circuit:
        the compiled circuit in the backend language.
    compiler_arguments:
        dictionary of arguments for compilation needed for chosen backend. Overwritten by inheritors.
    device:
        instantiated device (or None) for executing circuits.
    n_qubits:
        the number of qubits this circuit operates on.
    noise:
        the NoiseModel applied to this circuit when sampled.
    qubit_map:
        the mapping from tequila qubits to the qubits of the backend circuit.
    qubits:
        a list of the qubits operated on by the circuit.

    Methods
    -------
    create_circuit
        generate a backend circuit from an abstract tequila circuit
    check_device:
        see if a given device is valid for the backend.
    retrieve_device:
        get an instance of or necessary informaton about a device, for emulation or use.
    add_parametrized_gate
        add a parametrized gate to a backend circuit.
    add_basic_gate
        add an unparametrized gate to a backend circuit.
    add_measurement
        add a measurement gate to a backend circuit.
    initialize_circuit:
        generate an empty circuit object for the backend.
    update_variables:
        overwrite the saved values of variables for backend execution.
    simulate:
        perform simulation, simulated sampling, or execute the circuit, e.g. with some hamiltonian for measurement.
    sample_paulistring:
        sample a circuit with one paulistring of a larger hamiltonian
    sample:
        same a circuit, measuring an entire hamiltonian.
    do_sample:
        subroutine for sampling. must be overwritten by inheritors.
    do_simulate:
        subroutine for wavefunction simulation. must be overwritten by inheritors.
    convert_measurements:
        transform the result of simulation from the backend return type.
    fast_return:
        Todo: Jakob what is this?
    make_qubit_map:
        create a dictionary to map the tequila qubit ordering to the backend qubits.
    optimize_circuit:
        use backend features to improve circuit depth.
    extract_variables:
        return a list of the variables in the abstract tequila circuit this backend circuit corresponds to.
    """


    # compiler instructions, override in backends
    # try to reduce True statements as much as possible for new backends
    compiler_arguments = {
        "trotterized": True,
        "swap": True,
        "multitarget": True,
        "controlled_rotation": True,
        "gaussian": True,
        "exponential_pauli": True,
        "controlled_exponential_pauli": True,
        "phase": True,
        "power": True,
        "hadamard_power": True,
        "controlled_power": True,
        "controlled_phase": True,
        "toffoli": True,
        "phase_to_z": True,
        "cc_max": True
    }

    @property
    def n_qubits(self) -> numbers.Integral:
        return len(self.qubit_map)

    @property
    def qubits(self) -> typing.Iterable[numbers.Integral]:
        return tuple(self._qubits)

    def __init__(self, abstract_circuit: QCircuit, variables, noise=None,device=None,
                 use_mapping=True, optimize_circuit=True, *args, **kwargs):
        """

        Parameters
        ----------
        abstract_circuit: QCircuit:
            the circuit which is to be rendered in the backend language.
        variables:
            values for the variables of abstract_circuit
        noise: optional:
            noise to apply to abstract circuit.
        device: optional:
            device on which to sample (or emulate sampling) abstract circuit.
        use_mapping: bool:
            whether or not to use qubit mapping. Defaults to true.
        optimize_circuit: bool:
            whether or not to attempt backend depth optimization. Defaults to true.
        args
        kwargs
        """
        self._variables = tuple(abstract_circuit.extract_variables())
        self.use_mapping = use_mapping

        compiler_arguments = self.compiler_arguments
        if noise is not None:
            compiler_arguments["cc_max"] = True
            compiler_arguments["controlled_phase"] = True
            compiler_arguments["controlled_rotation"] = True
            compiler_arguments["hadamard_power"] = True

        # compile the abstract_circuit
        c = compiler.Compiler(**compiler_arguments)

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

        if optimize_circuit and noise ==None:
            self.circuit = self.optimize_circuit(circuit=self.circuit)

        self.noise = noise

        self.check_device(device)
        self.device = self.retrieve_device(device)

    def __call__(self,
                 variables: typing.Dict[Variable, numbers.Real] = None,
                 samples: int = None,
                 *args,
                 **kwargs):
        """
        Simulate or sample the backend circuit.

        Parameters
        ----------
        variables: dict:
            dictionary assigning values to the variables of the circuit.
        samples: int, optional:
            how many shots to sample with. If None, perform full wavefunction simulation.
        args
        kwargs

        Returns
        -------
        Float:
            the result of simulating or sampling the circuit.
        """

        variables = format_variable_dictionary(variables=variables)
        if self._variables is not None and len(self._variables) > 0:
            if variables is None or set(self._variables) != set(variables.keys()):
                raise TequilaException("BackendCircuit received not all variables. Circuit depends on variables {}, you gave {}".format(self._variables, variables))
        if samples is None:
            return self.simulate(variables=variables, noise=self.noise, *args, **kwargs)
        else:
            return self.sample(variables=variables, samples=samples, noise=self.noise, *args, **kwargs)

    def create_circuit(self, abstract_circuit: QCircuit, *args, **kwargs):
        """
        build the backend specific circuit from the abstract tequila circuit.

        Parameters
        ----------
        abstract_circuit: QCircuit:
            the circuit to build in the backend
        args
        kwargs

        Returns
        -------
        type varies
            The circuit, compiled to the backend.
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

    def check_device(self,device):
        """
        Verify if a device can be used in the selected backend. Overwritten by inheritors.
        Parameters
        ----------
        device:
            the device to verify.

        Returns
        -------

        Raises
        ------
        TequilaException
        """
        if device is not None:
            TequilaException('Devices not enabled for {}'.format(str(type(self))))

    def retrieve_device(self,device):
        """
        get the instantiated backend device object, from user provided object (e.g, a string).

        Must be overwritten by inheritors, to use devices.

        Parameters
        ----------
        device:
            object which points to the device in question, to be returned.

        Returns
        -------
        Type:
            varies by backend.
        """
        if device is None:
            return device
        else:
            TequilaException('Devices not enabled for {}'.format(str(type(self))))

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
        This is the default, which just translates the circuit again.
        Overwrite in inheritors if parametrized circuits are supported.
        """
        self.circuit = self.create_circuit(abstract_circuit=self.abstract_circuit, variables=variables)

    def simulate(self, variables, initial_state=0, *args, **kwargs) -> QubitWaveFunction:
        """
        simulate the circuit via the backend.

        Parameters
        ----------
        variables:
            the parameters with which to simulate the circuit.
        initial_state: Default = 0:
            one of several types; determines the base state onto which the circuit is applied.
            Default: the circuit is applied to the all-zero state.
        args
        kwargs

        Returns
        -------
        QubitWaveFunction:
            the wavefunction of the system produced by the action of the circuit on the initial state.

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

        result = self.do_simulate(variables=variables, initial_state=keymap.inverted(initial_state).integer, *args,
                                  **kwargs)
        result.apply_keymap(keymap=keymap, initial_state=initial_state)
        return result

    def sample(self, variables, samples, *args, **kwargs):
        """
        Sample the circuit. If circuit natively equips paulistrings, sample therefrom.
        Parameters
        ----------
        variables:
            the variables with which to sample the circuit.
        samples: int:
            the number of samples to take.
        args
        kwargs

        Returns
        -------
        QubitWaveFunction
            The result of sampling, a recreated QubitWaveFunction in the sampled basis.

        """
        self.update_variables(variables)
        return self.do_sample(samples=samples, circuit=self.circuit, *args, **kwargs)

    def sample_all_z_hamiltonian(self, samples: int, hamiltonian, *args, **kwargs):
        # make measurement instruction
        qubits = [q for q in hamiltonian.qubits if q in self.abstract_qubit_map]
        if len(qubits) == 0:
            return sum([ps.coeff for ps in hamiltonian.paulistrings])
        measure = Measurement(target=qubits)
        circuit = self.circuit + self.create_circuit(measure)
        # run simulators
        counts = self.do_sample(samples=samples, circuit=circuit, *args, **kwargs)

        # compute energy
        E = 0.0
        for paulistring in hamiltonian.paulistrings:
            n_samples = 0
            Etmp = 0.0
            for key, count in counts.items():
                parity = [k for i,k in enumerate(key.array) if i in paulistring._data].count(1)
                sign = (-1) ** parity
                Etmp += sign * count
                n_samples += count
            E += Etmp / samples * paulistring.coeff
        return E

    def sample_paulistring(self, samples: int, paulistring, *args,
                           **kwargs) -> numbers.Real:
        """
        Sample an individual pauli word (pauli string) and return the average result thereof.
        Parameters
        ----------
        samples: int:
            how many samples to evaluate.
        paulistring:
            the paulistring to be sampled.
        args
        kwargs

        Returns
        -------
        float:
            the average result of sampling the chosen paulistring
        """

        # make basis change and translate to backend
        basis_change = QCircuit()
        not_in_u = []
        # all indices of the paulistring which are not part of the circuit i.e. will always have the same outcome
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
            circuit = self.circuit + self.create_circuit(basis_change + measure)
            # run simulators
            counts = self.do_sample(samples=samples, circuit=circuit, *args, **kwargs)
            # compute energy
            E = 0.0
            n_samples = 0
            for key, count in counts.items():
                parity = key.array.count(1)
                sign = (-1) ** parity
                E += sign * count
                n_samples += count
            assert n_samples == samples
            E = E / samples * paulistring.coeff
            return E

    def do_sample(self, samples, circuit, noise, *args, **kwargs) -> QubitWaveFunction:
        """
        helper function for sampling. MUST be overwritten by inheritors.

        Parameters
        ----------
        samples: int:
            the number of samples to take
        circuit:
            the circuit to sample from.
            Note:
            Not necessarily self.circuit!
        noise:
            the noise to apply to the sampled circuit.
        args
        kwargs

        Returns
        -------
        QubitWaveFunction:
            the result of sampling.

        """
        TequilaException("Backend Handler needs to be overwritten for supported simulators")


    def do_simulate(self, variables, initial_state, *args, **kwargs) -> QubitWaveFunction:
        """
        helper for simulation. MUST be overwritten by inheritors.

        Parameters
        ----------
        variables:
            the variables with which the circuit may be simulated.
        initial_state:
            the initial state in which the system is in prior to the application of the circuit.
        args
        kwargs

        Returns
        -------
        QubitWaveFunction
            the result of simulating the circuit.

        """
        TequilaException("Backend Handler needs to be overwritten for supported simulators")

    def convert_measurements(self, backend_result) -> QubitWaveFunction:
        TequilaException("Backend Handler needs to be overwritten for supported simulators")

    def fast_return(self, abstract_circuit):
        """
        Todo: Jakob, what is this?
        Parameters
        ----------
        abstract_circuit

        Returns
        -------

        """
        return True

    def make_qubit_map(self, qubits):
        """
        Build the mapping between abstract qubits.

        Must be overwritten by inheritors to do anything other than check the validity of the map.
        Parameters
        ----------
        qubits:
            the qubits to map onto.

        Returns
        -------
        Dict
            the dictionary that maps the qubits of the abstract circuits to an ordered sequence of integers.
        """
        assert (len(self.abstract_qubit_map) == len(qubits))
        return self.abstract_qubit_map

    def optimize_circuit(self, circuit, *args, **kwargs):
        """
        Optimize a circuit using backend tools. Should be overwritten by inheritors.
        Parameters
        ----------
        circuit:
            the circuit to optimize
        args
        kwargs

        Returns
        -------
        Type
            Optimized version of the circuit.
        """
        return circuit

    def extract_variables(self) -> typing.Dict[str, numbers.Real]:
        """
        extract the tequila variables from the circuit.
        Returns
        -------
        dict:
            the variables of the circuit.
        """
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
    """
    Class representing an ExpectationValue for evaluation by some backend.

    Attributes
    ----------
    H:
        the tequila Hamiltonian of the expectationvalue
    n_qubits:
        how many qubits appear in the expectationvalue.
    U:
        the underlying BackendCircuit of the expectationvalue.

    Methods
    -------
    extract_variables:
        return the underlying tequila variables of the circuit
    initialize_hamiltonian
        prepare the hamiltonian for iteration over as a tuple
    initialize_unitary
        compile the abstract circuit to a backend circuit.
    simulate:
        simulate the unitary to measure H
    sample:
        sample the unitary to measure H
    sample_paulistring
        sample a single term from H
    update_variables
        wrapper over the update_variables of BackendCircuit.

    """
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
        """
        wrapper over circuit extract variables
        Returns
        -------
        Dict
            Todo: is it really a dict?
        """
        result = []
        if self.U is not None:
            result = self.U.extract_variables()
        return result

    def __init__(self, E, variables, noise, device):
        """

        Parameters
        ----------
        E:
            the uncompiled expectationvalue
        variables:
            variables for compilation of circuit
        noise:
            noisemodel for compilation of circuit
        device:
            device for compilation of circuit
        """
        self._U = self.initialize_unitary(E.U, variables=variables, noise=noise, device=device)
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
        """return a tuple with one member, H"""
        return tuple(H)

    def initialize_unitary(self, U, variables, noise, device):
        """return a compiled unitary"""
        return self.BackendCircuitType(abstract_circuit=U, variables=variables, device=device, use_mapping=self.use_mapping,
                                       noise=noise)

    def update_variables(self, variables):
        """wrapper over circuit update_variables"""
        self._U.update_variables(variables=variables)

    def sample(self, variables, samples, *args, **kwargs) -> numpy.array:
        """
        sample the expectationvalue.

        Parameters
        ----------
        variables: dict:
            variables to supply to the unitary.
        samples: int:
            number of samples to perform.
        args
        kwargs

        Returns
        -------
        numpy.ndarray:
            a numpy array, the result of sampling.
        """
        self.update_variables(variables)

        result = []
        for H in self._abstract_hamiltonians:
            E = 0.0
            if H.is_all_z():
                E = self.U.sample_all_z_hamiltonian(samples=samples, hamiltonian=H, *args, **kwargs)
            else:
                for ps in H.paulistrings:
                    E += self.sample_paulistring(samples=samples, paulistring=ps, *args, **kwargs)
            result.append(to_float(E))
        return numpy.asarray(result)

    def simulate(self, variables, *args, **kwargs):
        """
        Simulate the expectationvalue.

        Parameters
        ----------
        variables:
            variables to supply to the unitary.
        args
        kwargs

        Returns
        -------
        numpy array:
            the result of simulation.
        """
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
        """
        wrapper over the sample_paulistring method of BackendCircuit
        Parameters
        ----------
        samples: int:
            the number of samples to take
        paulistring:
            the paulistring to be sampled
        args
        kwargs

        Returns
        -------
        number:
            the result of simulating a single paulistring
        """

        return self.U.sample_paulistring(samples=samples, paulistring=paulistring,*args,**kwargs)
