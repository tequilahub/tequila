import qulacs
from qulacs_core import QuantumStateGpu

from tequila import TequilaException
from tequila.simulators.simulator_qulacs import BackendCircuitQulacs, BackendExpectationValueQulacs


class TequilaQulacsGpuException(TequilaException):
    def __str__(self):
        return "Error in qulacs gpu backend:" + self.message


class BackendCircuitQulacsGpu(BackendCircuitQulacs):
    quantum_state_class = QuantumStateGpu


class BackendExpectationValueQulacsGpu(BackendExpectationValueQulacs):
    BackendCircuitType = BackendCircuitQulacsGpu
    pass
