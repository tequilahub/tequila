from tequila.simulators.simulator_qiskit import BackendCircuitQiskit, BackendExpectationValueQiskit


class BackendCircuitQiskitGpu(BackendCircuitQiskit):
    STATEVECTOR_DEVICE_NAME = "aer_simulator_statevector_gpu"


class BackendExpectationValueQiskitGpu(BackendExpectationValueQiskit):
    BackendCircuitType = BackendCircuitQiskitGpu
