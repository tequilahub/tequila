from openvqe import OpenVQEModule, OpenVQEException, BitNumbering
from openvqe.circuit.circuit import QCircuit
from openvqe.keymap import KeyMapSubregisterToRegister
from openvqe.qubit_wavefunction import QubitWaveFunction
from openvqe.hamiltonian import PauliString
from openvqe.circuit.compiler import change_basis
from openvqe.circuit.gates import Measurement
from openvqe import BitString
from openvqe import copy, dataclass, typing
from openvqe.objective import Objective
from openvqe.simulator.heralding import HeraldingABC


@dataclass
class SimulatorReturnType:
    abstract_circuit: QCircuit = None
    circuit: int = None
    wavefunction: QubitWaveFunction = None
    measurements: typing.Dict[str, QubitWaveFunction] = None
    backend_result: int = None

    @property
    def counts(self, key: str = None):
        if key is None:
            keys = [k for k in self.measurements.keys()]
            return self.measurements[keys[0]]
        else:
            return self.measurements[key]


class Simulator(OpenVQEModule):
    """
    Abstract Base Class for OpenVQE interfaces to simulators
    """

    numbering: BitNumbering = BitNumbering.MSB

    def __init__(self, heralding: HeraldingABC = None):
        self._heralding = heralding

    def run(self, abstract_circuit: QCircuit, samples: int = 1) -> SimulatorReturnType:
        circuit = self.create_circuit(abstract_circuit=abstract_circuit)
        backend_result = self.do_run(circuit=circuit, samples=samples)
        return SimulatorReturnType(circuit=circuit,
                                   abstract_circuit=abstract_circuit,
                                   backend_result=backend_result,
                                   measurements=self.postprocessing(self.convert_measurements(backend_result)))

    def do_run(self, circuit, samples: int = 1):
        raise OpenVQEException("do_run needs to be overwritten by corresponding backend")

    def simulate_wavefunction(self, abstract_circuit: QCircuit, returntype=None,
                              initial_state: int = 0) -> SimulatorReturnType:
        """
        Simulates an abstract circuit with the backend specified by specializations of this class
        :param abstract_circuit: The abstract circuit
        :param returntype: specifies how the result should be given back
        :param initial_state: The initial state of the simulation,
        if given as an integer this is interpreted as the corresponding multi-qubit basis state
        :return: The resulting state
        """

        if isinstance(initial_state, BitString):
            initial_state = initial_state.integer
        if isinstance(initial_state, QubitWaveFunction):
            if len(initial_state.keys()) != 1:
                raise OpenVQEException("only product states as initial states accepted")
            initial_state = list(initial_state.keys())[0].integer

        active_qubits = abstract_circuit.qubits
        all_qubits = [i for i in range(abstract_circuit.n_qubits)]

        # maps from reduced register to full register
        keymap = KeyMapSubregisterToRegister(subregister=active_qubits, register=all_qubits)

        result = self.do_simulate_wavefunction(abstract_circuit=abstract_circuit,
                                               initial_state=keymap.inverted(initial_state).integer)
        result.wavefunction.apply_keymap(keymap=keymap, initial_state=initial_state)
        return result

    def do_simulate_wavefunction(self, circuit, initial_state=0) -> SimulatorReturnType:
        raise OpenVQEException(
            "called from base class of simulator, or non-supported operation for this backend")

    def create_circuit(self, abstract_circuit: QCircuit):
        """
        If the backend has its own circuit objects this can be created here
        :param abstract_circuit:
        :return: circuit object of the backend
        """
        raise OpenVQEException(
            "called from base class of simulator, or non-supported operation for this backend")

    def convert_measurements(self, backend_result) -> typing.Dict[str, QubitWaveFunction]:
        raise OpenVQEException(
            "called from base class of simulator, or non-supported operation for this backend")

    def measure_objective(self, objective: Objective, samples: int = 1, return_simulation_data: bool = False) -> float:
        final_E = 0.0
        data = []
        for U in objective.unitaries:
            weight = U.weight
            E = 0.0
            result_data = {}
            for ps in objective.observable.paulistrings:
                Etmp, tmp = self.measure_paulistring(abstract_circuit=U, paulistring=ps, samples=samples)
                E += Etmp
                result_data[str(ps)] = tmp
            final_E += weight * E
            if return_simulation_data:
                data.append(tmp)
        if return_simulation_data:
            return final_E, data
        else:
            return final_E

    def simulate_objective(self, objective: Objective, return_simulation_data: bool = False) -> float:
        final_E = 0.0
        data = []
        H = objective.observable
        for U in objective.unitaries:
            simresult = self.simulate_wavefunction(abstract_circuit=U)
            wfn = simresult.wavefunction
            final_E += U.weight * wfn.compute_expectationvalue(operator=H)
            if return_simulation_data:
                data.append(simresult)
        if return_simulation_data:
            return final_E, data
        else:
            return final_E

    def measure_paulistring(self, abstract_circuit: QCircuit, paulistring, samples: int = 1):
        # make basis change
        basis_change = QCircuit()
        for idx, p in paulistring.items():
            basis_change += change_basis(target=idx, axis=p)
        # make measurment instruction
        measure = QCircuit()
        qubits = [idx[0] for idx in paulistring.items()]
        if len(qubits) == 0:
            # no measurement instructions for a constant term as paulistring
            return (paulistring.coeff, SimulatorReturnType())
        else:
            measure += Measurement(name=str(paulistring), target=qubits)
            circuit = abstract_circuit + basis_change + measure
            # run simulator
            sim_result = self.run(abstract_circuit=circuit, samples=samples)

            # compute energy
            counts = sim_result.counts
            E = 0.0
            n_samples = 0
            for key, count in counts.items():
                parity = key.array.count(1)
                sign = (-1) ** parity
                E += sign * count
                n_samples += count
            assert (n_samples == samples)  # failsafe
            E = E / samples * paulistring.coeff
            return (E, sim_result)

    def postprocessing(self, measurements: typing.Dict[str, QubitWaveFunction]) -> typing.Dict[str, QubitWaveFunction]:
        # fast return
        if self._heralding is None:
            return measurements

        result = dict()
        for k, v in measurements.items():
            result[k] = self._heralding(input=v)
        return result
