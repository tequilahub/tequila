import qulacs
from tequila import TequilaException
from tequila.simulators.simulator_qulacs import BackendCircuitQulacs, BackendExpectationValueQulacs

class TequilaQulacsGpuException(TequilaException):
    def __str__(self):
        return "Error in qulacs qpu backend:" + self.message

class BackendCircuitQulacsGpu(BackendCircuitQulacs):

    _STATE_TYPE_ = "QuantumStateGpu"

class BackendExpectationValueQulacsGpu(BackendExpectationValueQulacs):
    # avoid namespace confusion
    pass
