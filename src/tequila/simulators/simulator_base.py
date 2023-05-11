from tequila.utils import TequilaException, to_float, TequilaWarning
from tequila.circuit.circuit import QCircuit
from tequila.utils.keymap import KeyMapSubregisterToRegister
from tequila.utils.misc import to_float
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila.circuit.compiler import change_basis
from tequila import BitString
from tequila.objective.objective import Variable, format_variable_dictionary
from tequila.circuit import compiler

import numbers, typing, numpy, copy, warnings

from dataclasses import dataclass

"""
Todo: Classes are now immutable: 
       - Add additional features from Skylars project
       - Maybe only keep paulistrings and not full hamiltonian types
"""


class BackendCircuit():
    """
    Base class for circuits compiled to run on specific backends.

    Attributes
    ----------
    no_translation:
        set this attribute in the derived __init__ to prevent translation of abstract_circuits
        needed for simulators that use native tequila types.
        Default is false
    abstract_circuit:
        the tequila circuit from which the backend circuit is built.
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
        Dicationary with keys being integers that enumerate the abstract qubits of the abstract_circuit
        and values being data-structures holding `number` and `instance` where number enumerates the
        backend qubits and instance is the instance of a backend qubit
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
        sample a circuit, measuring an entire hamiltonian.
    do_sample:
        subroutine for sampling. must be overwritten by inheritors.
    do_simulate:
        subroutine for wavefunction simulation. must be overwritten by inheritors.
    convert_measurements:
        transform the result of simulation from the backend return type.
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
        "generalized_rotation": True,
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
    def abstract_qubits(self) -> typing.Iterable[numbers.Integral]:
        return tuple(list(self.qubit_map.keys()))

    def qubit(self, abstract_qubit):
        """
        Convenience. Gives back a qubit instance of the corresponding backend
        Parameters
        ----------
        abstract_qubit
            the abstract tequila qubit

        Returns
        -------
            instance of backend qubit
        """
        return self.qubit_map[abstract_qubit].instance

    def __init__(self, abstract_circuit: QCircuit, variables, noise=None, device=None,
                 qubit_map=None, optimize_circuit=True, *args, **kwargs):
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
        qubit_map: dictionary:
            a qubit map which maps the abstract qubits in the abstract_circuit to the qubits on the backend
            there is no need to initialize the corresponding backend types
            the dictionary should simply be {int:int} (preferred) or {int:name}
            if None the default will map to qubits 0 ... n_qubits -1 in the backend
        optimize_circuit: bool:
            whether or not to attempt backend depth optimization. Defaults to true.
        args
        kwargs
        """

        self._input_args = {"abstract_circuit": abstract_circuit, "variables": variables, "noise": noise,
                            "qubit_map": qubit_map, "optimize_circuits": optimize_circuit, "device": device, **kwargs}

        self.no_translation = False
        self._variables = tuple(abstract_circuit.extract_variables())

        compiler_arguments = self.compiler_arguments
        if noise is not None:
            compiler_arguments["cc_max"] = True
            compiler_arguments["controlled_phase"] = True
            compiler_arguments["controlled_rotation"] = True
            compiler_arguments["hadamard_power"] = True

        # compile the abstract_circuit
        c = compiler.CircuitCompiler(**compiler_arguments)

        if qubit_map is None:
            qubit_map = {q: i for i, q in enumerate(abstract_circuit.qubits)}
        elif not qubit_map == {q: i for i, q in enumerate(abstract_circuit.qubits)}:
            warnings.warn("reveived custom qubit_map = {}\n"
                        "This is not fully integrated and might result in unexpected behaviour!"
                          .format(qubit_map), TequilaWarning)

            if len(qubit_map) > abstract_circuit.max_qubit()+1:
                raise TequilaException("Custom qubit_map has too many qubits {} vs {}".format(len(qubit_map), abstract_circuit.max_qubit()+1))
            if max(qubit_map.keys()) > abstract_circuit.max_qubit():
                raise TequilaException("Custom qubit_map tries to assign qubit {} but we only have {}".format(max(qubit_map.keys()), abstract_circuit.max_qubit()))

        # qubit map is initialized to have BackendQubits as values (they carry number and instance attributes)
        self.qubit_map = self.make_qubit_map(qubit_map)

        # pre-compilation (still an abstract ciruit, but with gates decomposed depending on backend requirements)
        compiled = c(abstract_circuit)
        self.abstract_circuit = compiled

        self.noise = noise
        self.check_device(device)
        self.device = self.retrieve_device(device)

        # translate into the backend object
        self.circuit = self.create_circuit(abstract_circuit=compiled, variables=variables)

        if optimize_circuit and noise is None:
            self.circuit = self.optimize_circuit(circuit=self.circuit)

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
            if variables is None or set(self._variables) > set(variables.keys()):
                raise TequilaException(
                    "BackendCircuit received not all variables. Circuit depends on variables {}, you gave {}".format(
                        self._variables, variables))

        self.update_variables(variables)
        if samples is None:
            return self.simulate(variables=variables, noise=self.noise, *args, **kwargs)
        else:
            return self.sample(variables=variables, samples=samples, noise=self.noise, *args, **kwargs)

    def create_circuit(self, abstract_circuit: QCircuit, circuit=None, *args, **kwargs):
        """
        build the backend specific circuit from the abstract tequila circuit.

        Parameters
        ----------
        abstract_circuit: QCircuit:
            the circuit to build in the backend
        circuit: BackendCircuitType (optional):
            Add to this already initialized circuit (not all backends support + operation)
        args
        kwargs

        Returns
        -------
        type varies
            The circuit, compiled to the backend.
        """

        # Backend uses native tequila structures
        if self.no_translation:
            return abstract_circuit

        result = circuit
        if result is None:
            result = self.initialize_circuit(*args, **kwargs)

        for g in abstract_circuit.gates:
            if g.is_parametrized():
                self.add_parametrized_gate(g, result, *args, **kwargs)
            else:
                self.add_basic_gate(g, result, *args, **kwargs)

        return result

    def check_device(self, device):
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
            raise TequilaException('Devices not enabled for {}'.format(str(type(self))))

    def retrieve_device(self, device):
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
            raise TequilaException('Devices not enabled for {}'.format(str(type(self))))

    def add_parametrized_gate(self, gate, circuit, *args, **kwargs):
        raise TequilaException("Backend Handler needs to be overwritten for supported simulators")

    def add_basic_gate(self, gate, circuit, *args, **kwargs):
        raise TequilaException("Backend Handler needs to be overwritten for supported simulators")

    def add_measurement(self, circuit, target_qubits, *args, **kwargs):
        raise TequilaException("Backend Handler needs to be overwritten for supported simulators")

    def initialize_circuit(self, *args, **kwargs):
        raise TequilaException("Backend Handler needs to be overwritten for supported simulators")

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
        active_qubits = self.qubit_map.keys()

        # maps from reduced register to full register
        keymap = KeyMapSubregisterToRegister(subregister=active_qubits, register=all_qubits)

        result = self.do_simulate(variables=variables, initial_state=keymap.inverted(initial_state).integer, *args,
                                  **kwargs)
        result.apply_keymap(keymap=keymap, initial_state=initial_state)
        return result

    def sample(self, variables, samples, read_out_qubits=None, circuit=None, *args, **kwargs):
        """
        Sample the circuit. If circuit natively equips paulistrings, sample therefrom.
        Parameters
        ----------
        variables:
            the variables with which to sample the circuit.
        samples: int:
            the number of samples to take.
        read_out_qubits: int:
            target qubits to measure (default is all)
        args
        kwargs

        Returns
        -------
        QubitWaveFunction
            The result of sampling, a recreated QubitWaveFunction in the sampled basis.

        """
        self.update_variables(variables)
        if read_out_qubits is None:
            read_out_qubits = self.abstract_qubits

        if len(read_out_qubits) == 0:
            raise Exception("read_out_qubits are empty")

        if circuit is None:
            circuit = self.add_measurement(circuit=self.circuit, target_qubits=read_out_qubits)
        else:
            circuit = self.add_measurement(circuit=circuit, target_qubits=read_out_qubits)
        return self.do_sample(samples=samples, circuit=circuit, read_out_qubits=read_out_qubits, *args, **kwargs)

    def sample_all_z_hamiltonian(self, samples: int, hamiltonian, variables, *args, **kwargs):
        """
        Sample from a Hamiltonian which only consists of Pauli-Z and unit operators
        Parameters
        ----------
        samples
            number of samples to take
        hamiltonian
            the tequila hamiltonian
        args
            arguments for do_sample
        kwargs
            keyword arguments for do_sample
        Returns
        -------
            samples, evaluated and summed Hamiltonian expectationvalue
        """
        # make measurement instruction (measure all qubits in the Hamiltonian that are also in the circuit)
        abstract_qubits_H = hamiltonian.qubits
        assert len(abstract_qubits_H) != 0  # this case should be filtered out before
        # assert that the Hamiltonian was mapped before
        if not all(q in self.qubit_map.keys() for q in abstract_qubits_H):
            raise TequilaException(
                "Qubits in {}-qubit Hamiltonian were not traced out for {}-qubit circuit".format(hamiltonian.n_qubits,
                                                                                                 self.n_qubits))

        # run simulators
        counts = self.sample(samples=samples, read_out_qubits=abstract_qubits_H, variables=variables, *args, **kwargs)
        read_out_map = {q: i for i, q in enumerate(abstract_qubits_H)}

        # compute energy
        E = 0.0
        for paulistring in hamiltonian.paulistrings:
            n_samples = 0
            Etmp = 0.0
            for key, count in counts.items():
                # get all the non-trivial qubits of the current PauliString (meaning all Z operators)
                # and mapp them to the backend qubits
                mapped_ps_support = [read_out_map[i] for i in paulistring._data.keys()]
                # count all measurements that resulted in |1> for those qubits
                parity = [k for i, k in enumerate(key.array) if i in mapped_ps_support].count(1)
                # evaluate the PauliString
                sign = (-1) ** parity
                Etmp += sign * count
                n_samples += count
            E += (Etmp / samples) * paulistring.coeff
            # small failsafe
            assert n_samples == samples
        return E

    def sample_paulistring(self, samples: int, paulistring, variables, *args,
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

        not_in_u = [q for q in paulistring.qubits if q not in self.abstract_qubits]
        reduced_ps = paulistring.trace_out_qubits(qubits=not_in_u)
        if reduced_ps.coeff == 0.0:
            return 0.0
        if len(reduced_ps._data.keys()) == 0:
            return reduced_ps.coeff

        # make basis change and translate to backend
        basis_change = QCircuit()
        qubits = []
        for idx, p in reduced_ps.items():
            qubits.append(idx)
            basis_change += change_basis(target=idx, axis=p)

        # add basis change to the circuit
        # deepcopy is necessary to avoid changing the circuits
        # can be circumvented by optimizing the measurements
        # on construction: tq.ExpectationValue(H=H, U=U, optimize_measurements=True)
        circuit = self.create_circuit(circuit=copy.deepcopy(self.circuit), abstract_circuit=basis_change)
        # run simulators
        counts = self.sample(samples=samples, circuit=circuit, read_out_qubits=qubits, variables=variables, *args,
                             **kwargs)
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

    def do_sample(self, samples, circuit, noise, abstract_qubits=None, *args, **kwargs) -> QubitWaveFunction:
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
        abstract_qubits:
            specify which qubits to measure. Default is all
        args
        kwargs

        Returns
        -------
        QubitWaveFunction:
            the result of sampling.

        """
        raise TequilaException("Backend Handler needs to be overwritten for supported simulators")

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
        raise TequilaException("Backend Handler needs to be overwritten for supported simulators")

    def convert_measurements(self, backend_result) -> QubitWaveFunction:
        raise TequilaException("Backend Handler needs to be overwritten for supported simulators")

    def initialize_qubit(self, number: int):
        """

        In case the backend has its own Qubit Types,
        this function should be overwritten by inheritors.

        Parameters
        ----------
        number
            the qubit number

        Returns
        -------
            Initialized backend qubit type

        """
        return number

    def make_qubit_map(self, qubits: dict):
        """
        Build the mapping between abstract qubits.

        Must be overwritten by inheritors to do anything other than check the validity of the map.
        Parameters
        ----------
        qubits:
            the qubits to map onto.
            If given as a dictionary, the map is already defined
            If given as a list the map will be those qubits mapped to 0 .... n_qubit-1 of the backend
        Returns
        -------
        Dict
            the dictionary that maps the qubits of the abstract circuits to an ordered sequence of integers.
            keys are the abstract qubit integers
            values are the backend qubits
            those are data structures which contain name and instance
            where number is the qubit identifier and instance the instance of the backend qubit
            if the backend does not require a special object for qubits the instance should be the same as number
        """

        @dataclass
        class BackendQubit:
            number: int = None
            instance: object = None

        if qubits is None:
            qubits = range(self.abstract_circuit.n_qubits)

        abstract_map = qubits
        if not hasattr(qubits, "keys") or not hasattr(qubits, "values"):
            abstract_map = {q: i for i, q in enumerate(qubits)}

        if all([hasattr(i, "number") and hasattr(i, "instance") for i in abstract_map.values()]):
            # qubit_map already initialized backend_types
            return qubits

        return {k: BackendQubit(number=v, instance=self.initialize_qubit(v)) for k, v in abstract_map.items()}

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

    def __copy__(self):
        return self.__deepcopy__()

    def __deepcopy__(self, memodict={}):
        return type(self)(**self._input_args)


class BackendExpectationValue:
    """
    Class representing an ExpectationValue for evaluation by some backend.

    Attributes
    ----------
    H:
        the reduced tequila Hamiltonian(s) of the expectationvalue
        reduction procedure is tracing out all qubits that are not part of the unitary U
        stored as a tuple to evaluate multiple Hamiltonians over the same circuit faster in pure simulations
    abstract_H:
        the original (non-reduced) Hamiltonian(s)
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

    def count_measurements(self):
        return self.abstract_expectationvalue.count_measurements()

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

    def __init__(self, E, variables, noise, device, *args, **kwargs):
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
        self.abstract_expectationvalue = E
        self._input_args = {"variables": variables, "device": device, "noise": noise, **kwargs}
        self._U = self.initialize_unitary(E.U, variables=variables, noise=noise, device=device, **kwargs)
        self._reduced_hamiltonians = self.reduce_hamiltonians(self.abstract_expectationvalue.H)
        self._H = self.initialize_hamiltonian(self._reduced_hamiltonians)

        self._variables = E.extract_variables()
        self._contraction = E._contraction
        self._shape = E._shape

    def __copy__(self):
        return self.__deepcopy__()

    def __deepcopy__(self, memodict={}):
        return type(self)(self.abstract_expectationvalue, **self._input_args)

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
        else:
            data = float(data)
        if self._contraction is None:
            return data
        else:
            return self._contraction(data)

    def reduce_hamiltonians(self, abstract_hamiltonians: tuple) -> tuple:
        """

        Parameters
        ----------
        abstract_hamiltonians
            tuple of abstract tequila Hamiltonians
        Returns
        -------
            reduces Hamiltonians where the qubits that are not defined in self.U are traced out
        """
        abstract_qubits_of_u = self.U.qubit_map.keys()
        reduced = []
        for H in abstract_hamiltonians:
            abstract_qubits_of_h = H.qubits
            not_in_u = [q for q in abstract_qubits_of_h if q not in abstract_qubits_of_u]
            reduced.append(H.trace_out_qubits(qubits=not_in_u))

        return tuple(reduced)

    def initialize_hamiltonian(self, hamiltonians: tuple) -> tuple:
        return hamiltonians

    def initialize_unitary(self, U, variables, noise, device, *args, **kwargs):
        """return a compiled unitary"""
        return self.BackendCircuitType(abstract_circuit=U, variables=variables, device=device,
                                       use_mapping=self.use_mapping,
                                       noise=noise, *args, **kwargs)

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

        suggested = None
        if hasattr(samples, "lower") and samples.lower()[:4] == "auto":
            if self.abstract_expectationvalue.samples is None:
                raise TequilaException("samples='auto' requested but no samples where set in individual expectation values")
            total_samples = int(samples[5:])
            samples = max(1, int(self.abstract_expectationvalue.samples * total_samples))
            suggested = samples
            # samples are not necessarily set (either the user has to set it or some functions like optimize_measurements)
 
        if suggested is not None and suggested != samples:
            warnings.warn("simulating with samples={}, but expectationvalue carries suggested samples={}\nTry calling with samples='auto-total#ofsamples'".format(samples, suggested), TequilaWarning)

        self.update_variables(variables)

        result = []
        for H in self._reduced_hamiltonians:
            E = 0.0
            if len(H.qubits) == 0:
                E = sum([ps.coeff for ps in H.paulistrings])
            elif H.is_all_z():
                E = self.U.sample_all_z_hamiltonian(samples=samples, hamiltonian=H, variables=variables, *args,
                                                    **kwargs)
            else:
                for ps in H.paulistrings:
                    E += self.U.sample_paulistring(samples=samples, paulistring=ps, variables=variables, *args,
                                                   **kwargs)
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
            # TODO inefficient,
            # Always better to overwrite this function
            wfn = self.U.simulate(variables=variables, *args, **kwargs)
            final_E += wfn.compute_expectationvalue(operator=H)
            result.append(to_float(final_E))
        return numpy.asarray(result)
