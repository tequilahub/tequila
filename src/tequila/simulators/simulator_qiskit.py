from tequila.simulators.simulator_base import BackendCircuit, QCircuit, BackendExpectationValue
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila import TequilaException
from tequila import BitString, BitNumbering, BitStringLSB
import qiskit
import qiskit.providers.aer.noise as qiskitnoise



def get_bit_flip(p):
    return qiskitnoise.pauli_error(noise_ops=[('X',p),('I',1-p)])

def get_phase_flip(p):
    return qiskitnoise.pauli_error(noise_ops=[('Z',p),('I',1-p)])

noise_lookup={
    'phase damp':qiskitnoise.phase_damping_error,
    'amplitude damp':qiskitnoise.amplitude_damping_error,
    'bit flip':get_bit_flip,
    'phase flip':get_phase_flip,
    'phase-amplitude damp':qiskitnoise.phase_amplitude_damping_error,
    'depolarizing':qiskitnoise.depolarizing_error
}

gate_qubit_lookup={
    'x':1,
    'y':1,
    'z':1,
    'h':1,
    'u1':1,
    'u2':1,
    'u3':1,
    'cx': 2,
    'cy': 2,
    'cz': 2,
    'ch':2,
    'cu3':2,
    'ccx':3,
    'r':1,
    'single':1,
    'control':2,
    'multicontrol':3
}

basis=['x','y','z','id','u1','u2','u3','h',
       'cx','cy','cz','cu3','ccx']
class TequilaQiskitException(TequilaException):
    def __str__(self):
        return "Error in qiskit backend:" + self.message


op_lookup={
    'I': (lambda c: c.iden),
    'X': (lambda c: c.x,lambda c: c.cx, lambda c: c.ccx),
    'Y': (lambda c: c.y,lambda c: c.cy, lambda c: c.ccy),
    'Z': (lambda c: c.z,lambda c: c.cz, lambda c: c.ccz),
    'H': (lambda c: c.h,lambda c: c.ch, lambda c: c.cch),
    'Rx': (lambda c: c.rx,lambda c: c.mcrx),
    'Ry': (lambda c: c.ry,lambda c: c.mcry),
    'Rz': (lambda c: c.rz,lambda c: c.mcrz),
    'SWAP': (lambda c: c.swap,lambda c: c.cswap),
    'Measure': (lambda c: c.measure)
}


class BackendCircuitQiskit(BackendCircuit):
    recompile_swap = True
    recompile_multitarget = True
    recompile_controlled_rotation = True
    recompile_controlled_power = True
    recompile_power = True
    recompile_hadamard_power = True
    cc_max=True
    numbering = BitNumbering.LSB

    def __init__(self, abstract_circuit: QCircuit, variables, use_mapping=True,noise_model=None, *args, **kwargs):
        if use_mapping:
            qubits = abstract_circuit.qubits
        else:
            qubits = range(abstract_circuit.n_qubits)

        nm=self.noise_model_converter(noise_model)
        self.noise_model=nm
        n_qubits = len(qubits)
        self.q = qiskit.QuantumRegister(n_qubits, "q")
        self.c = qiskit.ClassicalRegister(n_qubits, "c")
        self.classical_map = {i: self.c[j] for j, i in enumerate(qubits)}
        self.qubit_map = {i: self.q[j] for j, i in enumerate(qubits)}

        self.match_par_to_sympy={}
        self.counter = 0
        super().__init__(abstract_circuit=abstract_circuit, variables=variables,noise_model=self.noise_model, use_mapping=use_mapping, *args, **kwargs)
        if len(self.match_par_to_sympy.keys()) is None:
            self.match_sympy_to_value = None
            self.resolver=None
        else:
            self.match_sympy_to_value = {v: k for k, v in self.match_par_to_sympy.items()}
            self.resolver={k:v(variables) for k,v in self.match_sympy_to_value.items()}
        if self.noise_model is None:
            self.ol=1
        else:
            self.ol=0
    def do_simulate(self, variables, initial_state=0, *args, **kwargs) -> QubitWaveFunction:
        if self.noise_model is None:
            simulator = qiskit.Aer.get_backend("statevector_simulator")
        else:
            raise TequilaQiskitException("wave function simulation with noise cannot be performed presently")
        if initial_state != 0:
            # need something like this
            # there is a keyword for the backend for tolerance on norm
            # circuit.initialize(normed_array)
            raise TequilaQiskitException("initial state for Qiskit not yet supported here")
        backend_result = qiskit.execute(experiments=self.circuit, backend=simulator,parameter_binds=[self.resolver]).result()
        return QubitWaveFunction.from_array(arr=backend_result.get_statevector(self.circuit), numbering=self.numbering)

    def do_sample(self, circuit: qiskit.QuantumCircuit, samples: int, *args, **kwargs) -> QubitWaveFunction:
        simulator = qiskit.providers.aer.QasmSimulator()
        return self.convert_measurements(qiskit.execute(circuit,backend=simulator, shots=samples,
                                                        optimization_level=0,
                                                        noise_model=self.noise_model,parameter_binds=[self.resolver]))

    def convert_measurements(self, backend_result) -> QubitWaveFunction:
        """0.
        :param qiskit_counts: qiskit counts as dictionary, states are binary in little endian (LSB)
        :return: Counts in OpenVQE format, states are big endian (MSB)
        """
        qiskit_counts = backend_result.result().get_counts()
        result = QubitWaveFunction()
        # todo there are faster ways
        for k, v in qiskit_counts.items():
            converted_key = BitString.from_bitstring(other=BitStringLSB.from_binary(binary=k))
            result._state[converted_key] = v
        return result

    def fast_return(self, abstract_circuit):
        return isinstance(abstract_circuit, qiskit.QuantumCircuit)

    def initialize_circuit(self, *args, **kwargs):
        return qiskit.QuantumCircuit(self.q, self.c)


    def add_gate(self, gate, circuit, *args, **kwargs):
        try:
            ops=op_lookup[gate.name]
        except:
            raise TequilaQiskitException('Due to conventions around controls,we cannot look up gates not in the list of allowed gates. SORRY!')
        if gate.is_parametrized():
            try:
                par = self.match_par_to_sympy[gate.parameter]
            except:
                par = qiskit.circuit.parameter.Parameter('p_{}'.format(str(self.counter)))
                self.match_par_to_sympy[gate.parameter] = par
                self.counter += 1
            if gate.is_controlled():
                ops[1](circuit)(par,[self.qubit_map[c] for c in gate.control],self.qubit_map[gate.target[0]])
            else:
                ops[0](circuit)(par,self.qubit_map[gate.target[0]])
        else:
            if gate.name is 'Measure':
                tq = [self.qubit_map[t] for t in gate.target]
                tc = [self.classical_map[t] for t in gate.target]
                circuit.measure(tq, tc)
            else:
                if gate.is_controlled():
                    ops[len(gate.control)](circuit)(*[self.qubit_map[q] for q in gate.control+gate.target])
                else:
                    ops[0](circuit)(*[self.qubit_map[q] for q in gate.target])



    def create_circuit(self, abstract_circuit: QCircuit, *args, **kwargs):
        """
        Translates abstract circuits into the specific backend type
        :param abstract_circuit: Abstract circuit to be translated
        :return: translated circuit
        """

        if self.fast_return(abstract_circuit):
            return abstract_circuit

        result = self.initialize_circuit()

        for g in abstract_circuit.gates:
            self.add_gate(g, result)
        return result

    '''def add_gate(self, gate, circuit, *args, **kwargs):
        if len(gate.target) > 1:
            raise TequilaQiskitException("multi targets need to be explicitly recompiled for Qiskit")
        gfunc = getattr(circuit, gate.name.lower())
        gfunc(self.qubit_map[gate.target[0]])

    def add_controlled_gate(self, gate, circuit, *args, **kwargs):
        if len(gate.target) > 1:
            raise TequilaQiskitException("multi targets need to be explicitly recompiled for Qiskit")
        if len(gate.control) == 1:
            gfunc = getattr(circuit, "c" + gate.name.lower())
            gfunc(self.qubit_map[gate.control[0]], self.qubit_map[gate.target[0]])
        elif len(gate.control) == 2:
            try:
                gfunc = getattr(circuit, "cc" + gate.name.lower())
            except AttributeError:
                raise TequilaQiskitException("Double controls are currenty only supported for CCX in Qiskit")
            gfunc(self.qubit_map[gate.control[0]], self.qubit_map[gate.control[1]], self.qubit_map[gate.target[0]])
        else:
            raise TequilaQiskitException("More than two control gates currently not supported")

    def add_rotation_gate(self, gate, variables, circuit, *args, **kwargs):
        if len(gate.target) > 1:
            raise TequilaQiskitException("multi targets need to be explicitly recompiled for Qiskit")
        gfunc = getattr(circuit, gate.name.lower())
        try:
            par= self.match_par_to_sympy[gate.parameter]
        except:
            par = qiskit.circuit.parameter.Parameter('p_{}'.format(str(self.counter)))
            self.match_par_to_sympy[gate.parameter] = par
            self.counter += 1
        gfunc(par, self.qubit_map[gate.target[0]])

    def add_controlled_rotation_gate(self, gate, variables, circuit, *args, **kwargs):
        if len(gate.target) > 1:
            raise TequilaQiskitException("multi targets need to be explicitly recompiled for Qiskit")
        if len(gate.control) == 1:
            try:
                par = self.match_par_to_sympy[gate.parameter]
            except:
                par = qiskit.circuit.parameter.Parameter('p_{}'.format(str(self.counter)))
                self.match_par_to_sympy[gate.parameter] = par
                self.counter += 1
            gfunc = getattr(circuit, "c" + gate.name.lower())

            gfunc(par, self.qubit_map[gate.control[0]], self.qubit_map[gate.target[0]])
        elif len(gate.control) == 2:
            gfunc = getattr(circuit, "cc" + gate.name.lower())
            try:
                par = self.match_par_to_sympy[gate.parameter]
            except:
                par = qiskit.circuit.parameter.Parameter('p_{}'.format(str(self.counter)))
                self.match_par_to_sympy[gate.parameter] = par
                self.counter += 1
            gfunc(par, self.qubit_map[gate.control[0]], self.qubit_map[gate.control[1]],
                  self.qubit_map[gate.target[0]])
        else:
            raise TequilaQiskitException("More than two control gates currently not supported")

    def add_measurement(self, gate, circuit, *args, **kwargs):
        tq = [self.qubit_map[t] for t in gate.target]
        tc = [self.classical_map[t] for t in gate.target]
        circuit.measure(tq, tc)
    '''
    def make_map(self, qubits):
        # for qiskit this is done in init
        assert(self.q is not None)
        assert(self.c is not None)
        assert(len(self.qubit_map) == len(qubits))
        assert(len(self.abstract_qubit_map) == len(qubits))
        return self.qubit_map

    def noise_model_converter(self,nm):
        if nm is None:
            return None
        basis_gates = basis
        qnoise=qiskitnoise.NoiseModel(basis_gates)
        for noise in nm.noises:
            op=noise_lookup[noise.name]
            if op is qiskitnoise.depolarizing_error:
                active=op(noise.probs[0],noise.level)
            else:
                if noise.level==1:
                    active=op(*noise.probs)
                else:
                    active=op(*noise.probs)
                    action=op(*noise.probs)
                    for i in range(noise.level-1):
                        active=active.tensor(action)

            if noise.level is 2:
                targets=['cx',
                         'cy',
                         'cz',
                         'crz',
                         'crx',
                         'cry',
                         'cu3',
                         'ch']

            elif noise.level is 1:
                targets=['x',
                         'y',
                         'z',
                         'u3',
                         'u1',
                         'u2',
                         'h']

            elif noise.level is 3:
                targets=['ccx']
            qnoise.add_all_qubit_quantum_error(active,targets)

        return qnoise

    def update_variables(self, variables):
        """
        overriding the underlying base to make sure this stuff remains noisy
        """
        if self.match_sympy_to_value is not None:
            self.resolver={k:v(variables) for k,v in self.match_sympy_to_value.items()}
        else:
            self.resolver=None
class BackendExpectationValueQiskit(BackendExpectationValue):
    BackendCircuitType = BackendCircuitQiskit
