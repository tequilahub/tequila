"""
Wrappers to represent photonic modes by multiple qubits
"""
from numpy import log2
from openvqe.circuit import gates

# pretty printing
from openvqe.simulator.simulator_qiskit import SimulatorQiskit

class PhotonicMode:

    def __repr__(self):
        return self.name + str(self.n_qubits) + " qubits: " + str(self.qubits)

    @property
    def n_qubits(self):
        return len(self.qubits)

    @property
    def name(self):
        if self._name is None:
            return "M"
        else:
            return self._name

    @name.setter
    def name(self, other):
        self._name = other

    def qubit_names(self):
        return [self.name+"_"+str(k) for k,v in enumerate(self.qubits)]

    def __init__(self, qubits, name=None):
        self.qubits = qubits # qubit map
        self._name = name



def prepare_332_state(nqubits=2, target_modes=None, path_names=None):
    """
    Prepare the state: |state>={|000> + |111> + |221>}_{abc}
    here the notation is |abc> where abc are paths and there is only one photon in each path
    In more notation this is:
       |000>_{abc} --> |1>_{0,a}  |1>_{0,b}  |1>_{0,c} : Mode 0 has one photon in path a as well as b and c
       |111>_{abc} --> |1>_{1,a}  |1>_{1,b}  |1>_{1,c}
       etc
    So the full state becomes with notation
       |N_0, N_1, N_2 >_a |N_0, N_1, N_2>_b |N_0, N_1, N_2>_c
       |state>=
         |100>_a |100>_b |100>_c
        +|010>_a |010>_b |010>_c
        +|001>_a |001>_b |010>_c

    :return: A circuit which prepares the photonic 332 state in qubit representation
    """

    if target_modes is None:
        target_modes = [0,1,2]

    if path_names is None:
        path_names = ["a", "b", "c"]

    Paths = {}
    for i, pkey in enumerate(path_names):
        Modes = {}
        start = i*len(target_modes)*nqubits
        for j,key in enumerate(target_modes):
            Modes[key] = PhotonicMode(qubits=[k for k in range(start+j*nqubits, start+(j+1)*nqubits, 1)], name=str(pkey)+"_M"+str(key))
        Paths[pkey] = Modes

    print("paths=\n",Paths)
    print(path_names[1])
    print(Paths[path_names[1]])

    # initialize
    circuit = gates.H(target=Paths[path_names[0]][target_modes[1]].qubits[0])
    circuit *= gates.X(target=Paths[path_names[0]][target_modes[0]].qubits[0])
    circuit *= gates.X(target=Paths[path_names[1]][target_modes[0]].qubits[0])
    circuit *= gates.X(target=Paths[path_names[2]][target_modes[0]].qubits[0])

    circuit *= gates.X(target=Paths[path_names[0]][target_modes[0]].qubits[0], control=Paths[path_names[0]][target_modes[1]].qubits[0])
    circuit *= gates.X(target=Paths[path_names[1]][target_modes[1]].qubits[0], control=Paths[path_names[0]][target_modes[1]].qubits[0])
    circuit *= gates.X(target=Paths[path_names[1]][target_modes[0]].qubits[0], control=Paths[path_names[1]][target_modes[1]].qubits[0])
    circuit *= gates.X(target=Paths[path_names[2]][target_modes[1]].qubits[0], control=Paths[path_names[1]][target_modes[1]].qubits[0])
    circuit *= gates.X(target=Paths[path_names[2]][target_modes[0]].qubits[0], control=Paths[path_names[2]][target_modes[1]].qubits[0])



    return circuit,Paths

if __name__=="__main__":

    circuit, paths = prepare_332_state(nqubits=2)

    names = []
    for k1,p in paths.items():
        for k2,m in p.items():
            names += m.qubit_names()
    print("names=", names)
    print(SimulatorQiskit().create_circuit(abstract_circuit=circuit, qname=names).draw())
