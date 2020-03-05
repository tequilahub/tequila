from tequila.simulators.simulator_base import QCircuit, TequilaException, BackendCircuit, BackendExpectationValue
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila import BitString, BitNumbering
import subprocess
import sys
import numpy as np
import pyquil
from pyquil import get_qc


name_dict={
    'I':'I',
    'Rx':'RX',
    'Rz':'RZ',
    'Ry':'RY',
    'X':'X',
    'x':'X',
    'Y': 'Y',
    'y': 'Y',
    'Z': 'Z',
    'z': 'Z',
    'Cz':'CZ',
    'Swap':'SWAP',
    'Cx':'CNOT',
    'cx':'CNOT',
    'ccx':'CCNOT',
    'CCx':'CCNOT',
    'H':'H',
    'h':'H',
    'Phase':'PHASE'
}

name_unitary_dict={
    'I':np.eye(2),
    'X':np.array([[0.,1.],[1.,0.]]),
    'Y':np.array([[0.,-1.j],[1.j,0.]]),
    'Z':np.array([[1.,0.],[0.,-1.]]),
    'H':np.array([[1/np.sqrt(2),1/np.sqrt(2)],[1/np.sqrt(2),-1/np.sqrt(2)]]),
    'CNOT':np.array([[1.,0.,0.,0.],
                     [0.,1.,0.,0.,],
                     [0.,0.,0.,1.],
                     [0.,0.,1.0,0.]
                     ]),
    'SWAP':np.array([[1.,0.,0.,0.],
                     [0.,0.,1.,0.],
                     [0.,1.,0.,0.],
                     [0.,0.,0.,1.]
                     ]),
    'CCNOT':np.array([[1.,0.,0.,0.,0.,0.,0.,0.],
                     [0.,1.,0.,0.,0.,0.,0.,0.],
                     [0.,0.,1.,0.,0.,0.,0.,0.],
                     [0.,0.,0.,1.,0.,0.,0.,0.],
                     [0.,0.,0.,0.,1.,0.,0.,0.],
                     [0.,0.,0.,0.,0.,1.,0.,0.],
                     [0.,0.,0.,0.,0.,0.,0.,1.],
                     [0.,0.,0.,0.,0.,0.,1.,0.]
                     ]),
}

def amp_damp_map(p):
    """
    Generate the Kraus operators corresponding to an amplitude damping
    noise channel.

    :params float p: The one-step damping probability.
    :return: A list [k1, k2] of the Kraus operators that parametrize the map.
    :rtype: list
    """
    damping_op = np.sqrt(p) * np.array([[0, 1],
                                                [0, 0]])

    residual_kraus = np.diag([1, np.sqrt(1-p)])
    return [residual_kraus, damping_op]

def phase_damp_map(p):
    mat1=np.array([[1,0],[0,np.sqrt(1-p)]])
    mat2=np.array([[0,0],[0,np.sqrt(p)]])
    return [mat1,mat2]

def bit_flip_map(p):

    mat1=np.array([[np.sqrt(1-p),0],[0,np.sqrt(1-p)]])
    mat2=np.array([[0,np.sqrt(p)],[np.sqrt(p),0]])
    return [mat1,mat2]

def phase_flip_map(p):
    mat1 = np.array([[np.sqrt(1 - p), 0], [0, np.sqrt(1 - p)]])
    mat2 = np.array([[np.sqrt(p), 0], [0, -np.sqrt(p)]])
    return [mat1, mat2]

def phase_amp_damp_map(a,b):
    A0 = [[1, 0], [0, np.sqrt(1 - a - b)]]
    A1 = [[0, np.sqrt(a)], [0, 0]]
    A2 = [[0, 0], [0, np.sqrt(b)]]
    #B0 = [[np.sqrt(1 - a - b), 0], [0, 1]]
    #B1 = [[0, 0], [np.sqrt(a), 0]]
    #B2 = [[np.sqrt(b), 0], [0, 0]]
    return [np.array(k) for k in [A0,A1,A2]]

def depolarizing_map(p):
    mat1 = np.array([[np.sqrt(1 - p), 0], [0, np.sqrt(1 - p)]])
    mat2 = np.array([[np.sqrt(p/3), 0], [0, -np.sqrt(p/3)]])
    mat3=np.array([[0,np.sqrt(p/3)],[np.sqrt(p/3),0]])
    mat4=np.array([[0.,-1.j*np.sqrt(p/3)],[1.j*np.sqrt(p/3),.0]])
    return [mat1, mat2,mat3,mat4]

noise_lookup={
    'amplitude damp': amp_damp_map,
    'phase damp': phase_damp_map,
    'bit flip': bit_flip_map,
    'phase flip':phase_flip_map,
    'phase-amplitude damp':phase_amp_damp_map,
    'depolarizing':depolarizing_map
}

def append_kraus_to_gate(kraus_ops, g):
    """
    Follow a gate `g` by a Kraus map described by `kraus_ops`.

    :param list kraus_ops: The Kraus operators.
    :param numpy.ndarray g: The unitary gate.
    :return: A list of transformed Kraus operators.
    """
    return [kj.dot(g) for kj in kraus_ops]

def unitary_maker(gate):
    try:
        return name_unitary_dict[gate.name]
    except:
        if gate.name is 'RX':
            return np.array([
                [np.cos(gate.params[0]/2),-1.j*np.sin(gate.params[0]/2)],
                [-1.j*np.sin(gate.params[0]/2),np.cos(gate.params[0]/2)]])
        if gate.name is 'RY':
            return np.array([
                [np.cos(gate.params[0]/2),-np.sin(gate.params[0]/2)],
                [np.sin(gate.params[0]/2),np.cos(gate.params[0]/2)]])

        if gate.name is 'RZ':
            return np.array([
                [np.exp(-1.j*gate.params[0] / 2), 0.],
                [0., np.exp(1.j*gate.params[0] / 2)]])


        if gate.name is 'PHASE':
            return np.array([
                [1, 0.],
                [0., np.exp(1.j*gate.params[0])]])
        else:
            raise TequilaPyquilException(' I do not know how to make the unitary for gate named {} '.format(gate.name))
class TequilaPyquilException(TequilaException):
    def __str__(self):
        return "simulator_pyquil: " + self.message


class BackendCircuitPyquil(BackendCircuit):
    recompile_swap = True
    recompile_multitarget = True
    recompile_controlled_rotation = False
    recompile_exponential_pauli = True
    recompile_trotter = True
    recompile_phase = False

    numbering = BitNumbering.LSB

    def __init__(self, abstract_circuit: QCircuit, variables, use_mapping=True,noise_model=None, *args, **kwargs):

        self.noise_model=noise_model
        super().__init__(abstract_circuit=abstract_circuit, variables=variables,noise_model=self.noise_model, use_mapping=use_mapping, *args, **kwargs)
        qvm=get_qc('{}q-qvm'.format(str(self.n_qubits)))
        self.circuit=self.get_noisy_prog(self.circuit,self.noise_model,qvm)
    def do_simulate(self, variables, initial_state, *args, **kwargs):
        simulator = pyquil.api.WavefunctionSimulator()
        n_qubits = self.n_qubits
        msb = BitString.from_int(initial_state, nbits=n_qubits)
        iprep = pyquil.Program()
        for i, val in enumerate(msb.array):
            if val > 0:
                iprep += pyquil.gates.X(i)

        with open('qvm.log', "a+") as outfile:
            sys.stdout = outfile
            sys.stderr = outfile
            outfile.write("\nSTART SIMULATION: \n")
            outfile.write(str(self.abstract_circuit))
            process = subprocess.Popen(["qvm", "-S"], stdout=outfile, stderr=outfile)
            backend_result = simulator.wavefunction(iprep + self.circuit)
            outfile.write("END SIMULATION: \n")
            process.terminate()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
        return QubitWaveFunction.from_array(arr=backend_result.amplitudes, numbering=self.numbering)

    def do_sample(self, samples, circuit,*args, **kwargs) -> QubitWaveFunction:
        n_qubits = self.n_qubits
        qc=get_qc('{}q-qvm'.format(str(n_qubits)))
        p=self.circuit
        bitstrings=qc.run_and_measure(p,trials=samples)
        return self.convert_measurements(bitstrings)

    def convert_measurements(self, backend_result) -> QubitWaveFunction:
        """0.
        :param backend_result: array from pyquil as list of lists of integers.
        :return: backend_result in Tequila format.
        """
        def string_to_array(s):
            listing=[]
            for letter in s:
                if letter not in [',',' ','[',']']:
                    listing.append(int(s))
            return listing


        result = QubitWaveFunction()
        bit_dict={}
        for b in backend_result.values():
            try:
                bit_dict[str(b)]+=1
            except:
                bit_dict[str(b)]=1

        for k,v in bit_dict.items():
            result._state[BitString.from_array(string_to_array(k))]=v
        return result


    def fast_return(self, abstract_circuit):
        return isinstance(abstract_circuit, pyquil.Program)

    def initialize_circuit(self, *args, **kwargs):
        return pyquil.Program()

    def add_gate(self, gate, circuit, *args, **kwargs):
        circuit += getattr(pyquil.gates, gate.name.upper())(self.qubit_map[gate.target[0]])

    def add_controlled_gate(self, gate, circuit, *args, **kwargs):
        pyquil_gate = getattr(pyquil.gates, gate.name.upper())(self.qubit_map[gate.target[0]])
        for c in gate.control:
            pyquil_gate = pyquil_gate.controlled(self.qubit_map[c])
        circuit += pyquil_gate

    def add_rotation_gate(self, gate, variables, circuit, *args, **kwargs):
        circuit += getattr(pyquil.gates, gate.name.upper())(gate.angle(variables), self.qubit_map[gate.target[0]])

    def add_controlled_rotation_gate(self, gate, variables, circuit, *args, **kwargs):
        pyquil_gate = getattr(pyquil.gates, gate.name.upper())(gate.angle(variables), self.qubit_map[gate.target[0]])
        for c in gate.control:
            pyquil_gate = pyquil_gate.controlled(self.qubit_map[c])
        circuit += pyquil_gate

    def add_power_gate(self, gate, circuit, *args, **kwargs):
        raise TequilaPyquilException("PowerGates are not supported")

    def add_controlled_power_gate(self, gate, circuit, *args, **kwargs):
        raise TequilaPyquilException("controlled PowerGates are not supported")

    def add_measurement(self, gate, circuit, *args, **kwargs):
        bits = len(gate.target)
        ro = circuit.declare('ro', 'BIT', bits)
        for i, t in enumerate(gate.target):
            circuit += pyquil.gates.MEASURE(self.qubit_map[t], ro[i])

    def get_noisy_prog(self,py_prog, noise_model,cxn):
        prog = py_prog
        final_prog = pyquil.Program()
        qubits=cxn.qubits()
        for l_inst in range(0, prog.__len__()):
            list_inst = str(prog.__getitem__(l_inst)).split()
            if len(list_inst) == 2:
                final_prog.inst("{0} {1}".format(list_inst[0], qubits[int(list_inst[1])]))
            elif len(list_inst) == 3:
                final_prog.inst("{0} {1} {2}".format(list_inst[0], qubits[int(list_inst[1])], qubits[int(list_inst[2])]))

        for gate in prog:
            for noise in noise_model.noises:
                if name_dict[noise.gate] is not gate.name:
                    pass
                else:
                    prog.define_noisy_gate(name_dict[noise.gate],
                                            gate.qubits,
                        append_kraus_to_gate(noise_lookup[noise.name](*noise.probs),
                                             unitary_maker(gate)))
        return prog




class BackendExpectationValuePyquil(BackendExpectationValue):
    BackendCircuitType = BackendCircuitPyquil
