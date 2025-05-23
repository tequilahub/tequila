from calendar import c
from tequila.circuit.circuit import QCircuit
from tequila.utils.keymap import KeyMapRegisterToSubregister
from tequila import BitString, BitNumbering, BitStringLSB
from tequila.circuit.compiler import change_basis
import numbers, typing, numpy, copy, warnings
from tequila.objective.objective import Variable, format_variable_dictionary
import typing
from tequila.utils.misc import to_float
import numpy
from typing import Union
from tequila.simulators.simulator_qiskit import BackendCircuitQiskit, BackendExpectationValueQiskit, TequilaQiskitException
from qiskit_aqt_provider import AQTProvider
from qiskit_aqt_provider.aqt_resource import AQTResource
import qiskit
from qiskit.circuit import QuantumCircuit
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila import TequilaException, TequilaWarning, circuit
from tequila.hamiltonian.qubit_hamiltonian import QubitHamiltonian
from tequila.circuit import gates 
import numpy as np
from tequila.objective import ExpectationValue

class TequilaAQTException(TequilaQiskitException):
    def __str__(self):
        return "Error in AQT backend:" + self.message


class BackendCircuitAQT(BackendCircuitQiskit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = self.get_backend()
    
        
    def get_backend(token: str = "") -> AQTResource:
        provider = AQTProvider(token)
        # TODO: what if somebody actually passes a valid aqt cloud token?
        backend = provider.get_backend('offline_simulator_no_noise')
        return backend
    
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
        return self.do_sample(samples=samples, circuit=circuit, read_out_qubits=read_out_qubits,
                              initial_state=initial_state, *args, **kwargs)

    
    
    def do_sample(self, circuit: Union[QuantumCircuit, list[QuantumCircuit]], samples: int, read_out_qubits, initial_state=0, *args,
                  **kwargs) -> Union[QubitWaveFunction, list[QubitWaveFunction]]:
        optimization_level = 1
        if 'optimization_level' in kwargs:
            optimization_level = kwargs['optimization_level']
        qiskit_backend = self.retrieve_device(self.device)
       
        circuit = circuit.assign_parameters(self.resolver)  
        circuit = self.add_state_init(circuit, initial_state)   
        circuit = qiskit.transpile(circuit, backend=qiskit_backend, optimization_level=optimization_level)
        
        job = qiskit_backend.run(circuit, shots=samples)
        counts = job.result().get_counts()
        wfn = self.convert_measurements(counts, target_qubits=read_out_qubits)
        return wfn

    def convert_measurements(self, qiskit_counts, target_qubits=None) -> list[QubitWaveFunction]:
        result = QubitWaveFunction(self.n_qubits, self.numbering)
        # todo there are faster ways
        for k, v in qiskit_counts.items():
            # Qiskit uses LSB bitstrings, but from_binary expects MSB
            converted_key = BitString.from_binary(k[::-1])
            result[converted_key] = v
        if target_qubits is not None:
            mapped_target = [self.qubit_map[q].number for q in target_qubits]
            mapped_full = [self.qubit_map[q].number for q in self.abstract_qubits]
            keymap = KeyMapRegisterToSubregister(subregister=mapped_target, register=mapped_full)
            result = QubitWaveFunction.from_wavefunction(result, keymap, n_qubits=len(target_qubits))
   
        return result


class BackendExpectationValueAQT(BackendExpectationValueQiskit):
    BackendCircuitType = BackendCircuitAQT

