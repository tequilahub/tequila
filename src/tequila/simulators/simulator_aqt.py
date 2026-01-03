from calendar import c
from tequila.utils.keymap import KeyMapRegisterToSubregister
from tequila import BitString
from tequila.simulators.simulator_qiskit import (
    BackendCircuitQiskit,
    BackendExpectationValueQiskit,
    TequilaQiskitException,
)
from qiskit_aqt_provider import AQTProvider
from qiskit_aqt_provider.aqt_resource import AQTResource
from qiskit.circuit import QuantumCircuit
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila import TequilaException
import qiskit
from typing import Union


class TequilaAQTException(TequilaQiskitException):
    def __str__(self):
        return "Error in AQT backend:" + self.message


class BackendCircuitAQT(BackendCircuitQiskit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "device" in kwargs.keys() and kwargs["device"] is not None:
            self.device = kwargs["device"]

    def retrieve_backend(self) -> AQTResource:
        if self.device is None:
            raise TequilaAQTException("No AQT device specified. Please provide a valid AQT device as backend.")
        return self.device

    # state measurement operation?
    def get_circuit(
        self, circuit: QuantumCircuit, qiskit_backend, initial_state=0, optimization_level=1, *args, **kwargs
    ) -> QuantumCircuit:
        circ = circuit.assign_parameters(self.resolver)  # this is necessary -- see qiskit-aer issue 1346
        circ = self.add_state_init(circ, initial_state)
        basis = qiskit_backend.target.operation_names
        circ = qiskit.transpile(circ, backend=qiskit_backend, basis_gates=basis, optimization_level=optimization_level)
        return circ

    def sample(self, variables, samples, read_out_qubits=None, circuit=None, initial_state=0, *args, **kwargs):
        if initial_state != 0 and not self.supports_sampling_initialization:
            raise TequilaException("Backend does not support initial states for sampling")

        if isinstance(initial_state, QubitWaveFunction) and not self.supports_generic_initialization:
            raise TequilaException("Backend does not support arbitrary initial states")

        self.update_variables(variables)
        if read_out_qubits is None:
            read_out_qubits = self.abstract_qubits

        if len(read_out_qubits) == 0:
            raise Exception("read_out_qubits are empty")

        if circuit is None:
            circuit = self.add_measurement(circuit=self.circuit, target_qubits=read_out_qubits)
        else:
            if isinstance(circuit, list):
                assert len(circuit) == len(read_out_qubits), "circuit and read_out_qubits have to be of the same length"
                for i, c in enumerate(circuit):
                    circuit[i] = self.add_measurement(circuit=c, target_qubits=read_out_qubits[i])
            else:
                circuit = self.add_measurement(circuit=circuit, target_qubits=read_out_qubits)
        return self.do_sample(
            samples=samples,
            circuit=circuit,
            read_out_qubits=read_out_qubits,
            initial_state=initial_state,
            *args,
            **kwargs,
        )

    def do_sample(
        self, circuit: QuantumCircuit, samples: int, read_out_qubits, initial_state=0, *args, **kwargs
    ) -> QubitWaveFunction:
        optimization_level = 1
        if "optimization_level" in kwargs:
            optimization_level = kwargs["optimization_level"]
        qiskit_backend = self.retrieve_backend()

        circuit = self.get_circuit(
            circuit=circuit,
            qiskit_backend=qiskit_backend,
            initial_state=initial_state,
            optimization_level=optimization_level,
            *args,
            **kwargs,
        )
        job = qiskit_backend.run(circuit, shots=samples)
        wfn = self.convert_measurements(job, target_qubits=read_out_qubits)

        return wfn

    def do_simulate(self, variables, initial_state=0, *args, **kwargs):
        raise TequilaAQTException("AQT backend does not support do_simulate")


class BackendExpectationValueAQT(BackendExpectationValueQiskit):
    BackendCircuitType = BackendCircuitAQT
