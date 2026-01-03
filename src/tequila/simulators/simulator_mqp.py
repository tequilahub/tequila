from mqp.qiskit_provider import MQPProvider, MQPBackend
from qiskit.circuit import QuantumCircuit
from mqp.qiskit_provider import MQPProvider, MQPBackend
from tequila import TequilaException
from tequila.simulators.simulator_aqt import BackendCircuitAQT, BackendExpectationValueAQT


class BackendCircuitMQP(BackendCircuitAQT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # don't transpile the circuit for mqp
    def get_circuit(
        self, circuit: QuantumCircuit, qiskit_backend, initial_state=0, optimization_level=1, *args, **kwargs
    ) -> QuantumCircuit:
        circ = circuit.assign_parameters(self.resolver)  # this is necessary -- see qiskit-aer issue 1346
        circ = self.add_state_init(circ, initial_state)
        return circ

    def do_simulate(self, variables, initial_state=0, *args, **kwargs):
        raise TequilaMQPException("MQP backend does not support do_simulate")


class BackendExpectationValueMQP(BackendExpectationValueAQT):
    BackendCircuitType = BackendCircuitMQP


class TequilaMQPException(TequilaException):
    def __str__(self):
        return "Error in MQP backend:" + self.message
