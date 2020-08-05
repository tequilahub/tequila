import qulacs
from tequila import TequilaException
from tequila.simulators.simulator_qulacs import BackendCircuitQulacs, BackendExpectationValueQulacs

class TequilaQulacsGpuException(TequilaException):
    def __str__(self):
        return "Error in qulacs qpu backend:" + self.message

class BackendCircuitQulacsGpu(BackendCircuitQulacs):
    def initialize_state(n_qubits:int=None) -> qulacs.QuantumState:
        if n_qubits is None:
            n_qubits = self.n_qubits
        return qulacs.QuantumStateGpu(n_qubits)

class BackendExpectationValueQulacsGpu(BackendExpectationValueQulacs):
    BackendCircuitType = BackendCircuitQulacsGpu
    pass
