import qulacs
import numbers, numpy
from tequila import TequilaException
from tequila.utils.bitstrings import BitNumbering, BitString, BitStringLSB
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila.simulators.simulator_base import QCircuit, change_basis
from tequila.simulators.simulator_qulacs import BackendCircuitQulacs, BackendExpectationValueQulacs

"""
Developer Note:
    Qulacs uses different Rotational Gate conventions: Rx(angle) = exp(i angle/2 X) instead of exp(-i angle/2 X)
    And the same for MultiPauli rotational gates
    The angles are scaled with -1.0 to keep things consistent with the rest of tequila
"""


class TequilaQulacsGpuException(TequilaException):
    def __str__(self):
        return "Error in qulacs qpu backend:" + self.message

class BackendCircuitQulacsGpu(BackendCircuitQulacs):

    compiler_arguments = {
        "trotterized": True,
        "swap": False,
        "multitarget": True,
        "controlled_rotation": True, # needed for gates depending on variables
        "gaussian": True,
        "exponential_pauli": False,
        "controlled_exponential_pauli": True,
        "phase": True,
        "power": True,
        "hadamard_power": True,
        "controlled_power": True,
        "controlled_phase": True,
        "toffoli": False,
        "phase_to_z": True,
        "cc_max": False
    }

    numbering = BitNumbering.LSB

    def do_simulate(self, variables, initial_state, *args, **kwargs):
        state = qulacs.QuantumStateGpu(self.n_qubits)
        lsb = BitStringLSB.from_int(initial_state, nbits=self.n_qubits)
        state.set_computational_basis(BitString.from_binary(lsb.binary).integer)
        self.circuit.update_quantum_state(state)

        wfn = QubitWaveFunction.from_array(arr=state.get_vector(), numbering=self.numbering)
        return wfn

class BackendExpectationValueQulacsGpu(BackendExpectationValueQulacs):
    BackendCircuitType = BackendCircuitQulacsGpu
    use_mapping = True

    def simulate(self, variables, *args, **kwargs) -> numpy.array:
        # fast return if possible
        if self.H is None:
            return numpy.asarray([0.0])
        elif len(self.H) == 0:
            return numpy.asarray([0.0])
        elif isinstance(self.H, numbers.Number):
            return numpy.asarray[self.H]

        self.U.update_variables(variables)
        state = qulacs.QuantumStateGpu(self.U.n_qubits)
        self.U.circuit.update_quantum_state(state)
        result = []
        for H in self.H:
            if isinstance(H, numbers.Number):
                result.append(H) # those are accumulated unit strings, e.g 0.1*X(3) in wfn on qubits 0,1
            else:
                result.append(H.get_expectation_value(state))

        return numpy.asarray(result)

    def sample(self, variables, samples, *args, **kwargs) -> numpy.array:
        # todo: generalize in baseclass. Do Hamiltonian mapping on initialization
        self.update_variables(variables)
        state = qulacs.QuantumStateGpu(self.U.n_qubits)
        self.U.circuit.update_quantum_state(state)
        result = []
        for H in self._abstract_hamiltonians:
            E = 0.0
            for ps in H.paulistrings:
                # change basis, measurement is destructive so copy the state
                # to avoid recomputation
                bc = QCircuit()
                zero_string = False
                for idx, p in ps.items():
                    if idx not in self.U.qubit_map:
                        # circuit does not act on the qubit
                        # case1: paulimatrix is 'Z' -> unit factor: ignore that part
                        # case2: zero factor -> continue with next ps
                        if p.upper() != "Z":
                            zero_string = True
                    else:
                        bc += change_basis(target=idx, axis=p)

                if zero_string:
                    continue

                qbc = self.U.create_circuit(abstract_circuit=bc, variables=None)
                Esamples = []
                for sample in range(samples):
                    if self.U.has_noise:
                        state = qulacs.QuantumStateGpu(self.U.n_qubits)
                        self.U.circuit.update_quantum_state(state)
                        state_tmp = state
                    else:
                        state_tmp = state.copy()
                    if len(bc.gates) > 0:  # otherwise there is no basis change (empty qulacs circuit does not work out)
                        qbc.update_quantum_state(state_tmp)
                    ps_measure = 1.0
                    for idx in ps.keys():
                        if idx not in self.U.qubit_map:
                            continue  # means its 1 or Z and <0|Z|0> = 1 anyway
                        else:
                            M = qulacs.gate.Measurement(self.U.qubit_map[idx], self.U.qubit_map[idx])
                            M.update_quantum_state(state_tmp)
                            measured = state_tmp.get_classical_value(self.U.qubit_map[idx])
                            ps_measure *= (-2.0 * measured + 1.0)  # 0 becomes 1 and 1 becomes -1
                    Esamples.append(ps_measure)
                E += ps.coeff * sum(Esamples) / len(Esamples)
            result.append(E)
        return numpy.asarray(result)
