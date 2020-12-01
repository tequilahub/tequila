import qtensor, qtree
import numbers, numpy
from tequila import TequilaException
from tequila.utils.bitstrings import BitNumbering, BitString, BitStringLSB
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila.simulators.simulator_base import BackendCircuit, BackendExpectationValue, QCircuit, change_basis
from tequila.utils.keymap import KeyMapRegisterToSubregister


class TequilaQtensorException(TequilaException):
    def __str__(self):
        return "Error in qtensor backend:" + self.message

class BackendCircuitQtensor(BackendCircuit):

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

    def __init__(self, abstract_circuit, noise=None, *args, **kwargs):
        """

        Parameters
        ----------
        abstract_circuit: QCircuit:
            the circuit to compile to qulacs
        noise: optional:
            noise to apply to the circuit.
        args
        kwargs
        """
        self.op_lookup = {
            'I': None,
            'X': qtree.operator.X,
            'Y': qtree.operators.Y,
            'Z': qtree.operators.Z,
            'H': qtree.operators.H,
            'Rx': None,
            'Ry': None,
            'Rz': None,
            'SWAP': None,
            'Measure': qtree.operators.M,
            'Exp-Pauli': None
        }
        self.measurements = None
        self.variables = []
        super().__init__(abstract_circuit=abstract_circuit, noise=noise, *args, **kwargs)
        self.has_noise = False
        if noise is not None:
            raise TequilaQtensorException("Noisy simulation not supported in Qtensor backend!")


    def initialize_circuit(self, *args, **kwargs):
        return []

    def do_sample(self, samples, circuit, noise, abstract_qubits=None, *args, **kwargs) -> QubitWaveFunction:
        raise TequilaQtensorException("Sampling not supported in Qtensor backend!")

    def do_simulate(self, variables, initial_state, *args, **kwargs) -> QubitWaveFunction:
        raise TequilaQtensorException("Wavefunction simulation not supported in Qtensor interface!")

    def add_basic_gate(self, gate, circuit, *args, **kwargs):
        op = self.op_lookup[gate.name]
        qtree_gate = op(*[self.qubit(t) for t in gate.target])
        if gate.is_controlled():
            raise TequilaQtensorException("Controlled gates not yet supported in Qtensor backend! Can be circumvented by compiling")

        circuit.append(qtree_gate)


class BackendExpectationValueQtensor(BackendExpectationValue):

    use_mapping = True
    BackendCircuitType = BackendCircuitQtensor

    def sample(self, variables, samples, *args, **kwargs) -> numpy.array:
        raise TequilaQtensorException("Sampling not supported in QTensor backend")

    def simulate(self, variables, *args, **kwargs):
        

