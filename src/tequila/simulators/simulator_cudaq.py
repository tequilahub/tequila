import cudaq
from cudaq import spin

import qulacs
import numbers, numpy
import warnings

from tequila import TequilaException, TequilaWarning
from tequila.utils.bitstrings import BitNumbering, BitString, BitStringLSB
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila.simulators.simulator_base import BackendCircuit, BackendExpectationValue, QCircuit, change_basis
from tequila.utils.keymap import KeyMapRegisterToSubregister






"""
Developer Note:
    Cudaq does not have objects for circuits and gates. Instead it uses quantum kernels. These only allow the input of
    arguments of certain types (and are therefore quite picky), i.e. ints, floats or a list[int], list[float].
    Due to this restriction the circuit is a list of these primitive types and the used gates are encoded.
"""

class TequilaCudaqException(TequilaException):
    def __str__(self):
        return "Error in cudaq (cuda-quantum) backend:" + self.message

class BackendCircuitCudaq(BackendCircuit):
    """
    Class representing circuits compiled to cudaq (cuda-quantum of NVIDIA).
    See BackendCircuit for documentation of features and methods inherited therefrom

    Attributes
    ----------
    counter:
        counts how many distinct sympy.Symbol objects are employed in the circuit.
    has_noise:
        whether or not the circuit is noisy. needed by the expectationvalue to do sampling properly.
    noise_lookup: dict:
        dict mapping strings to lists of constructors for cirq noise channel objects.
    op_lookup: dict:
        dictionary mapping strings (tequila gate names) to cirq.ops objects.
    variables: list:
        a list of the qulacs variables of the circuit.

    Methods
    -------
    add_noise_to_circuit:
        apply a tequila NoiseModel to a qulacs circuit, by translating the NoiseModel's instructions into noise gates.
    """

    compiler_arguments = {
        "trotterized": True,
        "swap": True,
        "multitarget": True,
        "controlled_rotation": True, # needed for gates depending on variables
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
    # set convention of numbering to LSB 
    numbering = BitNumbering.LSB

    def __init__(self, abstract_circuit, noise=None, *args, **kwargs):
        """

        Parameters
        ----------
        abstract_circuit: QCircuit:
            the circuit to compile to cudaq
        noise: optional:
            noise to apply to the circuit.
        args
        kwargs
        """


        # gates encodings 
        self.op_lookup = {
            # primitives
            'I': None,
            'X': 1,
            'Y': 2,
            'Z': 3,
            'H': 4,
            # simple 1-qubit rotations 
            'Rx': 5,
            'Ry': 6,
            'Rz': 7,
            # phase gates
            'S' : 8,
            'T' : 9,
            # controlled primitives
            'CNOT' : 10,
            'Cx' : 11,
            'Cy' : 12,
            'Cz' : 13,
            # controlled rotations 
            'CRx' : 14,
            'CRy' : 15,
            'CRz' : 16,
            # other types of gates
            'SWAP': 17,
            'Measure': 18,
            'Exp-Pauli': None
        }

        # instantiate a cudaq circuit as a list
        self.circuit = self.initialize_circuit()

        self.measurements = None
        self.variables = []
        super().__init__(abstract_circuit=abstract_circuit, noise=noise, *args, **kwargs)
        self.has_noise=False


        # noise part is still not implemented for cudaq 
        if noise is not None:

            warnings.warn("Warning: noise in qulacs module will be dropped. Currently only works for qulacs version 0.5 or lower", TequilaWarning)

            self.has_noise=True
            self.noise_lookup = {
                'bit flip': [qulacs.gate.BitFlipNoise],
                'phase flip': [lambda target, prob: qulacs.gate.Probabilistic([prob],[qulacs.gate.Z(target)])],
                'phase damp': [lambda target, prob: qulacs.gate.DephasingNoise(target,(1/2)*(1-numpy.sqrt(1-prob)))],
                'amplitude damp': [qulacs.gate.AmplitudeDampingNoise],
                'phase-amplitude damp': [qulacs.gate.AmplitudeDampingNoise,
                                         lambda target, prob: qulacs.gate.DephasingNoise(target,(1/2)*(1-numpy.sqrt(1-prob)))
                                         ],
                'depolarizing': [lambda target,prob: qulacs.gate.DepolarizingNoise(target,3*prob/4)]
            }

            self.circuit=self.add_noise_to_circuit(noise)


    



    @cudaq.kernel
    def state_modifier(             ## create a state based on an empty state like |000..00> 
        number_of_qubits : int,
        gate_encodings : list[int],
        target_qubits : list[int],
        angles : list[float],
        control_qubits : list[int],
        iteration_length : int,
    ):
        """ 
        This function applies a circuit to an EMPTY QUANTUM STATE with a given number of qubits,
        i.e. |00> for a two-qubit state.
        The function collects the gates to apply, which are stored in a list (cudaq's circuit "object")
        applies them in the given order to the state. 

        These circuits support:
            1. single-qubit gates like: X, Y, Z, H, S, T as well as controlled variantes of them with up 
            to one control-qubit
            2. parametrized single qubit gates like: Rx, Ry, Rz with a given angle  
        """
        # create an empty state with given number of qubits
        s = cudaq.qvector(number_of_qubits)


        for index in range(iteration_length):
            encoding = gate_encodings[index]
            target = target_qubits[index]
            angle = angles[index]
            control = control_qubits[index]

            # x gate 
            if encoding == 1:
                if control != -1:
                    x.ctrl(s[control], s[target])
                else:
                    x(s[target])
            # y gate        
            elif encoding == 2:
                if control != -1:
                    y.ctrl(s[control], s[target])
                else:
                    y(s[target])
            # z gate 
            elif encoding == 3: 
                if control != -1:
                    z.ctrl(s[control], s[target])
                else:
                    z(s[target])
            # h gate 
            elif encoding == 4:
                if control != -1:
                    h.ctrl(s[control], s[target])
                else:
                    h(s[target])
            # Rx gate 
            elif encoding == 5:
                if control != -1:
                    pass
                else:
                    rx(angle, s[target])
            # Rx gate 
            elif encoding == 6:
                # support only parametrized rotations but without controls 
                if control != -1:
                    pass
                else:
                    ry(angle, s[target])        
            # Rx gate 
            elif encoding == 7:
                if control != -1:
                    pass
                else:
                    rz(angle, s[target])
            # S gate 
            elif encoding == 8:
                if control != -1:
                    s.ctrl(s[control], s[target])
                else:
                    s(s[target])
            # T gate
            elif encoding == 9:
                if control != -1:
                    t.ctrl(s[control], s[target])
                else:
                    t(s[target])



                    

    @cudaq.kernel
    def state_modifier_from_initial_state(             ## create a state based on an EXISTING PREVIOUS STATE i.e. |1010010> 
        number_of_qubits : int,
        gate_encodings : list[int],
        target_qubits : list[int],
        angles : list[float],
        control_qubits : list[int],
        iteration_length : int,
        inital_state : cudaq.State
    ):
        """ 
        This function applies a circuit to an EXISTING QUANTUM STATE with a given number of qubits,
        i.e. |10110>
        - unlike "state_modifier" this function needs a special preperation of the gates to apply to the state
        and therefore works with prepare_state_from_integer

        The function collects the gates to apply, which are stored in a list (cudaq's circuit "object")
        applies them in the given order to the state. 

        These circuits support:
            1. single-qubit gates like: X, Y, Z, H, S, T as well as controlled variantes of them with up 
            to one control-qubit
            2. parametrized single qubit gates like: Rx, Ry, Rz with a given angle  
        """

        # create a quantum state based on a given initial state 
        s = cudaq.qvector(inital_state)


        for index in range(iteration_length):
            encoding = gate_encodings[index]
            target = target_qubits[index]
            angle = angles[index]
            control = control_qubits[index]

            # x gate 
            if encoding == 1:
                if control != -1:
                    x.ctrl(s[control], s[target])
                else:
                    x(s[target])
            # y gate        
            elif encoding == 2:
                if control != -1:
                    y.ctrl(s[control], s[target])
                else:
                    y(s[target])
            # z gate 
            elif encoding == 3: 
                if control != -1:
                    z.ctrl(s[control], s[target])
                else:
                    z(s[target])
            # h gate 
            elif encoding == 4:
                if control != -1:
                    h.ctrl(s[control], s[target])
                else:
                    h(s[target])
            # Rx gate 
            elif encoding == 5:
                if control != -1:
                    pass
                else:
                    rx(angle, s[target])
            # Rx gate 
            elif encoding == 6:
                if control != -1:
                    pass
                else:
                    ry(angle, s[target])        
            # Rx gate 
            elif encoding == 7:
                if control != -1:
                    pass
                else:
                    rz(angle, s[target])
            # S gate 
            elif encoding == 8:
                if control != -1:
                    s.ctrl(s[control], s[target])
                else:
                    s(s[target])
            # T gate
            elif encoding == 9:
                if control != -1:
                    t.ctrl(s[control], s[target])
                else:
                    t(s[target])

                    

        
        




    def prepare_circuit_for_state_modifier(self):
        ''' 
        this function decomposes the circuit elements for later usage in state_modifier, which uses the 
        cudaq annotation @cudaq.kernel. 
        
        Has to be done this way since other functions for expectation value for example like cudaq.observe have special syntax 
        and accept parameters and the state modifier itself 
        '''



        print('before nqb')

        # prepare parameters for usage in kernal since "self. " access doesnt work within kernels  
        number_of_qubits = self.n_qubits

        print('n qb')

        circuit = None
        if isinstance(self, BackendCircuitCudaq):
            circuit = self.circuit
        elif isinstance(self, BackendExpectationValueCudaq):
            circuit = self.U.circuit
        
        if circuit is None:
            raise ValueError("wrong attribute access in function - prepare_circuit_for_state_modifier")    
        
        print("circ prep ", circuit)

        gate_encodings = []
        target_qubits = [] 
        angles = []
        control_qubits = []

        # get single lists from dict 
        gate_encodings = circuit['gate_encodings']
        target_qubits = circuit['target_qubits']
        angles = circuit['angles']
        control_qubits = circuit['control_qubits']

        iteration_length = None

        if len(gate_encodings) == len(target_qubits) == len(angles) == len(control_qubits):
            iteration_length = len(gate_encodings)
        else:
            raise ValueError('length of params lists in prepare_circuit_for_modifier has sto match')
        
        if iteration_length == None:
            raise ValueError('iter length from prepare_circuit shall not be None')

        print('before ret prepare ')

        return (number_of_qubits, gate_encodings, target_qubits, angles, control_qubits, iteration_length)



    def prepare_state_from_integer(state_index: int, num_qubits: int):
        """Prepare gate encodings to initialize the quantum state |state_index⟩.

        Args:
            state_index (int): The integer index of the basis state (e.g. 4 for |100⟩).
            num_qubits (int): Total number of qubits in the system.

        Returns:
            tuple: (number_of_qubits, gate_encodings, target_qubits, angles, control_qubits, iteration_length)
        """
        # Binary representation of state_index, padded to match number of qubits
        binary = format(state_index, f"0{num_qubits}b")

        gate_encodings = []
        target_qubits = []
        angles = []
        control_qubits = []

        # We apply an X gate to every qubit that needs to be 1
        for i, bit in enumerate(binary):  # Qubit 0 is least significant
            if bit == '1':
                gate_encodings.append(1)        # encoding 1 = X gate
                target_qubits.append(i)
                angles.append(0.0)              # not used for X
                control_qubits.append(-1)       # no control

        iteration_length = len(gate_encodings)

        return (num_qubits, gate_encodings, target_qubits, angles, control_qubits, iteration_length)



    def do_simulate(self, variables, initial_state, *args, **kwargs):
        """
        Helper function to perform simulation.

        - performs simulation in the following order:
            1. create a quantum state based on a given input integer 
            2. 


        Parameters
        ----------
        variables: dict:
            variables to supply to the circuit.
        initial_state:
            information indicating the initial state on which the circuit should act.
        args
        kwargs

        Returns
        -------
        QubitWaveFunction:
            QubitWaveFunction representing result of the simulation.
        """
        
        

        # given an input integer get the parameters to create a quantum state from 
        params = BackendCircuitCudaq.prepare_state_from_integer(initial_state, self.n_qubits)

        # get the quantum state created based on a given initial state for applying the circuit on it 
        quantum_state_from_integer = cudaq.get_state(self.state_modifier, *params)


        # for debugging - showing the initial state the was transformed from decimal >> binary >> quantum state 
        wfn_of_from_int = QubitWaveFunction.from_array(arr=numpy.array(quantum_state_from_integer), numbering=self.numbering)
        print(f"from int {initial_state} then qubit state {wfn_of_from_int}")


        # prepare circuit components for state modifier annotation block of cudaq 
        (number_of_qubits, gate_encodings, target_qubits, angles, 
         control_qubits, iteration_length) = BackendCircuitCudaq.prepare_circuit_for_state_modifier(self)
        

        print('after param fetch')

        

        # apply state modifier (circuit) onto state and get amplitudes 
        vector = cudaq.get_state(self.state_modifier_from_initial_state, number_of_qubits, gate_encodings, target_qubits,
                                 angles, control_qubits, iteration_length, quantum_state_from_integer)

        

        print('before wfn')

        print("circuit cudaq ", self.circuit)
        for k,v in self.circuit.items():
            print(k, type(k))
            print(v, type(v))
            pass

        wfn = QubitWaveFunction.from_array(arr=numpy.array(vector), numbering=self.numbering)

        # print("state vector cudaq ", wfn)

        return wfn





    def convert_measurements(self, backend_result, target_qubits=None) -> QubitWaveFunction:
        """
        Transform backend evaluation results into QubitWaveFunction
        Parameters
        ----------
        backend_result:
            the return value of backend simulation.

        Returns
        -------
        QubitWaveFunction
            results transformed to tequila native QubitWaveFunction
        """

        result = QubitWaveFunction()
        # todo there are faster ways


        for k in backend_result:
            converted_key = BitString.from_binary(BitStringLSB.from_int(integer=k, nbits=self.n_qubits).binary)
            if converted_key in result._state:
                result._state[converted_key] += 1
            else:
                result._state[converted_key] = 1

        if target_qubits is not None:
            mapped_target = [self.qubit_map[q].number for q in target_qubits]
            mapped_full = [self.qubit_map[q].number for q in self.abstract_qubits]
            keymap = KeyMapRegisterToSubregister(subregister=mapped_target, register=mapped_full)
            result = result.apply_keymap(keymap=keymap)

        return result

    def do_sample(self, samples, circuit, noise_model=None, initial_state=0, *args, **kwargs) -> QubitWaveFunction:
        """
        Helper function for performing sampling.

        Parameters
        ----------
        samples: int:
            the number of samples to be taken.
        circuit:
            the circuit to sample from.
        noise_model: optional:
            noise model to be applied to the circuit.
        initial_state:
            sampling supports initial states for qulacs. Indicates the initial state to which circuit is applied.
        args
        kwargs

        Returns
        -------
        QubitWaveFunction:
            the results of sampling, as a Qubit Wave Function.
        """
        state = self.initialize_state(self.n_qubits)
        lsb = BitStringLSB.from_int(initial_state, nbits=self.n_qubits)
        state.set_computational_basis(BitString.from_binary(lsb.binary).integer)
        circuit.update_quantum_state(state)
        sampled = state.sampling(samples)
        return self.convert_measurements(backend_result=sampled, target_qubits=self.measurements)


    def initialize_circuit(self, *args, **kwargs):
        """
        return an empty circuit.
        Parameters
        ----------
        args
        kwargs

        Returns
        -------
        an empty list, since the circuits in cudaq has to be created 
        based on a list of gates within a kernel 
        """

        # as for now circuit supports only CNOT, single qubit gates and single rotations
        circuit = {
            "gate_encodings": [],
            "target_qubits": [],
            "angles": [],
            "control_qubits": []
        }

        return circuit


    def add_parametrized_gate(self, gate, circuit, variables, *args, **kwargs):
        """
        add a parametrized gate.
        Parameters
        ----------
        gate: QGateImpl:
            the gate to add to the circuit.
        circuit:
            the circuit to which the gate is to be added
        variables:
            dict that tells values of variables; needed IFF the gate is an ExpPauli gate.
        args
        kwargs

        Returns
        -------
        None
        """
        gate_encoding = self.op_lookup[gate.name]


        # target qubits as a list 
        target_qubits = [self.qubit(t) for t in gate.target]

        if len(target_qubits) != 1:
            raise ValueError(" at most 1 target qubits is supported, NOT MORE ")

        # save all control qubits into a list 
        control_qubits = []
        if gate.is_controlled():
            for control in gate.control:
                control_qubits.append(self.qubit(control))

        # more than one control currently not supported 
        if len(control_qubits) > 1:
            raise ValueError('at must 1 control qubit is supported, NOT MORE ')

        # extract information about angle - one per gate 
        angle = gate.parameter(variables)

        # if the gate has one control qubit append its index to the list of controls
        if len(control_qubits) == 1:
            circuit['control_qubits'].append(control_qubits[0])
        elif len(control_qubits) == 0:
            circuit['control_qubits'].append(-1)

        # append the angle of the gate 
        circuit['angles'].append(angle)
        circuit['gate_encodings'].append(gate_encoding)
        circuit['target_qubits'].append(target_qubits[0])



        

    def add_basic_gate(self, gate, circuit, *args, **kwargs):
        """
        add an unparametrized gate to the circuit.
        Parameters
        ----------
        gate: QGateImpl:
            the gate to be added to the circuit.
        circuit:
            the circuit, to which a gate is to be added.
        args
        kwargs

        Returns
        -------
        None
        """

        # fetch the gate encoding 
        gate_encoding = self.op_lookup[gate.name]

        # target qubits as a list 
        target_qubits = [self.qubit(t) for t in gate.target]

        if len(target_qubits) != 1:
            raise ValueError(" at most 1 target qubits is supported, NOT MORE ")

        # save all control qubits into a list 
        control_qubits = []
        if gate.is_controlled():
            for control in gate.control:
                control_qubits.append(self.qubit(control))

        # more than one control currently not supported 
        if len(control_qubits) > 1:
            raise ValueError('at must 1 control qubit is supported, NOT MORE ')

        print('tr ', target_qubits, type(target_qubits))
        print('con ', control_qubits)
        print('gate enc ', gate_encoding)

        # if the gate has one control qubit append its index to the list of the controls 
        if len(control_qubits) == 1:
            circuit['control_qubits'].append(control_qubits[0])
        elif len(control_qubits) == 0:
            circuit['control_qubits'].append(-1)
        
        circuit['target_qubits'].append(target_qubits[0])
        circuit['gate_encodings'].append(gate_encoding)

        # since primitive gates have to angle add zero to the list of angles
        # so all 4 lists have the same length for iteration 
        circuit['angles'].append(0.0)

        print(' finish adding basic gate ')



    def add_measurement(self, circuit, target_qubits, *args, **kwargs):
        """
        Add a measurement operation to a circuit.
        Parameters
        ----------
        circuit:
            a circuit, to which the measurement is to be added.
        target_qubits: List[int]
            abstract target qubits
        args
        kwargs

        Returns
        -------
        None
        """
        pass


class BackendExpectationValueCudaq(BackendExpectationValue):
    """
    Class representing Expectation Values compiled for Cudaq.

    Overrides some methods of BackendExpectationValue, which should be seen for details.
    """
    use_mapping = True
    BackendCircuitType = BackendCircuitCudaq








    def simulate(self, variables, *args, **kwargs) -> numpy.array:
        """
        Perform simulation of this expectationvalue.
        Parameters
        ----------
        variables:
            variables, to be supplied to the underlying circuit.
        args
        kwargs

        Returns
        -------
        numpy.array:
            the result of simulation as an array.
        """
        # fast return if possible
        if self.H is None:
            return numpy.asarray([0.0])
        elif len(self.H) == 0:
            return numpy.asarray([0.0])
        elif isinstance(self.H, numbers.Number):
            return numpy.asarray[self.H]

        # a tupel: (gate_code, target, control)
        # gate_to_apply = (gate_encoding, target_qubits, conrtol_qubits)

        # print("self cir ", self.U.circuit)

        #for g, t, c, par in self.U.circuit:
            # print(g,t,c, par)
            # continue

        # print(" circ ", self.U.circuit)

        # print("amt of qbits ", self.U.n_qubits)


        
         
        (number_of_qubits, gate_encodings, target_qubits, angles, 
         control_qubits, iteration_length) = BackendCircuitCudaq.prepare_circuit_for_state_modifier(self)

        
        

        print("right before kernel ")

        """ beginning of cudaq kernel (oracle) """

        # rotation_gates_encodings = primitive_gate_encodings

        # use state_modifier here 
            
        # print("before print of modifier ")
        # print(state_modifier)    

        print("right after kernel ")

        # array containing results of simulation 
        resulting_expectation_values = []

        # c-of prem
        # print(f"con of primitives {control_qubits_of_primitives}")

        # go over all given hamiltonians 
        for hamiltonian in self.H:
            print(hamiltonian, "hamil from for loop ")
            # compute expectation value between hamiltonian and state 


            print("before ex val")
            # expectation_value = cudaq.observe(modified_state, hamiltonian).expectation()

            expectation_value = cudaq.observe(BackendCircuitCudaq.state_modifier, hamiltonian, number_of_qubits,
                                              gate_encodings, target_qubits, angles,
                                              control_qubits, iteration_length
                                                    ).expectation()
            
            # get the amplitudes after applying simulation
            amplitudes = cudaq.get_state(BackendCircuitCudaq.state_modifier, number_of_qubits, gate_encodings,
                                         target_qubits, angles, control_qubits, iteration_length)

            # show amplitudes 
            print(f"amplis {amplitudes}")

            print("after ex val")
            # print("one before")
            # print(expectation_value)
            resulting_expectation_values.append(expectation_value)

        # print("res exvals ",resulting_expectation_values)

        return numpy.asarray(resulting_expectation_values)
    


    def initialize_hamiltonian(self, hamiltonians):
        """
        Convert reduced hamiltonians to native Qulacs types for efficient expectation value evaluation.
        Parameters
        ----------
        hamiltonians:
            an interable set of hamiltonian objects.

        Returns
        -------
        list:
            initialized hamiltonian objects.

        """


        """ 
        # get information for spins 
        print(" \n \n")
        for h in hamiltonians:
            print(h, h.paulistrings)
            print(type(h), type(h.paulistrings))
            for pauli in h.paulistrings:
                print(pauli, type(pauli))
                for a, b in pauli.items():
                    print(f"{a} bit {b} gate {pauli._coeff} coeff")
        """ 

        # TODO -> on black board an example: correct multiplication of paulis and THEN multiplying with coefficient 

        list_of_initialized_hamiltonians = []
        hamiltonian_as_spin = 1
        # assemble hamiltonian with cudaq "spin" objects                    
        for hamiltonian in hamiltonians:
            hamiltonian_as_spin = 0  # Initialize per Hamiltonian
            
            for paulistring in hamiltonian.paulistrings:
                term = 1  # Start with identity
                
                for qubit, gate in paulistring.items():
                    if gate == 'X': 
                        term *= spin.x(qubit)
                    elif gate == 'Y': 
                        term *= spin.y(qubit)
                    elif gate == 'Z': 
                        term *= spin.z(qubit)
                
                term *= paulistring._coeff  # Apply coefficient
                hamiltonian_as_spin += term  # Accumulate terms
            
            list_of_initialized_hamiltonians.append(hamiltonian_as_spin)

            

        # show hamils 
        for index, h in enumerate(list_of_initialized_hamiltonians):
            print(f"hamil number {index + 1} is {h}")
            continue

        return list_of_initialized_hamiltonians


