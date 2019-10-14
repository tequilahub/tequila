from openvqe import OpenVQEModule, OpenVQEException, BitNumbering
from openvqe.circuit.circuit import QCircuit
from openvqe.tools.convenience import number_to_string
from openvqe.hamiltonian import PauliString
from numpy import isclose, ndarray, abs, sign
from openvqe.circuit.compiler import change_basis
from openvqe.circuit.gates import Measurement
from openvqe import BitString, BitStringLSB, initialize_bitstring
import copy
from dataclasses import dataclass
from openvqe.objective import Objective
from openvqe import typing

class KeyMapABC:

    @property
    def n_qubits(self):
        return None

    @property
    def numbering(self):
        return BitNumbering.MSB

    def __call__(self, input_state: BitString, initial_state: BitString = 0):
        return input_state


class KeyMapLSB2MSB(KeyMapABC):

    def __call__(self, input_state:BitStringLSB, initial_state: int=None ) -> BitString:
        return BitString.from_int(integer=input_state)

class KeyMapMSB2LSB(KeyMapABC):

    @property
    def numbering(self) -> BitNumbering:
        return BitNumbering.LSB

    def __call__(self, input_state: BitString, initial_state: int=None ) -> BitStringLSB:
        return BitStringLSB.from_int(integer=input_state)


class KeyMapQubitSubregister(KeyMapABC):

    @property
    def n_qubits(self):
        return len(self.register)

    @property
    def register(self):
        return self._register

    @property
    def subregister(self):
        return self._subregister

    @property
    def complement(self):
        return self.make_complement()

    def __init__(self, subregister: typing.List[int], register: typing.List[int]):
        self._subregister = subregister
        self._register = register

    def make_complement(self):
        return [i for i in self._register if i not in self._subregister]

    def __call__(self, input_state: BitString, initial_state: BitString = 0):

        input_state = BitString.from_int(integer=input_state, nbits=len(self._subregister))
        initial_state = BitString.from_int(integer=initial_state, nbits=len(self._subregister))

        output_state = BitString.from_int(integer=initial_state.integer, nbits=len(self._register))
        for k, v in enumerate((self._subregister)):
            output_state[v] = input_state[k]

        return output_state

    def inverted(self, input_state: int):
        """
        Map from register to subregister
        :param input_state:
        :return: input_state only on subregister
        """
        input_state = BitString.from_int(integer=input_state, nbits=len(self._register))
        output_state = BitString.from_int(integer=0, nbits=len(self._subregister))
        for k, v in enumerate(self._subregister):
            output_state[k] = input_state[v]
        return output_state


    def __repr__(self):
        return "keymap:\n" + "register    = " + str(self.register) + "\n" + "subregister = " + str(self.subregister)


class QubitWaveFunction:
    """
    Store Wavefunction as dictionary of comp. basis state and complex numbers
    Use the same structure for Measurments results with int instead of complex numbers (counts)
    """

    numbering = BitNumbering.MSB

    def apply_keymap(self, keymap, initial_state: BitString=None):
        self.n_qubits = keymap.n_qubits
        mapped_state = dict()
        for k, v in self.state.items():
            mapped_state[keymap(input_state=k, initial_state=initial_state)] = v

        self.state = mapped_state
        return self

    @property
    def n_qubits(self) -> int:
        if self._n_qubits is None:
            return self.min_qubits()
        else:
            return max(self._n_qubits,self.min_qubits())

    def min_qubits(self) -> int:
        if len(self.state) > 0:
            maxk = max(self.state.keys())
            return maxk.nbits
        else:
            return 0

    @n_qubits.setter
    def n_qubits(self, n_qubits):
        if n_qubits is not None:
            self._n_qubits = max(n_qubits, self.min_qubits())
        return self

    @property
    def state(self):
        if self._state is None:
            return dict()
        else:
            return self._state

    @state.setter
    def state(self, other: typing.Dict[BitString, complex]):
        assert (isinstance(other, dict))
        self._state = other

    def __init__(self, state: typing.Dict[int, complex] = None):
        if state is None:
            self._state = dict()
        else:
            self._state = state
        self._n_qubits = None

    def items(self):
        return self.state.items()

    def keys(self):
        return self.state.keys()

    def values(self):
        return self.state.values()

    def __getitem__(self, item):
        return self.state[item]

    def __setitem__(self, key, value):
        self._state[key]=value
        return self

    def __len__(self):
        return len(self.state)

    @classmethod
    def from_array(cls, arr: ndarray, keymap=None, threshold: float = 1.e-6, numbering: BitNumbering = BitNumbering.MSB):
        assert (len(arr.shape) == 1)
        state = dict()
        maxkey = len(arr)-1
        maxbit = initialize_bitstring(integer=maxkey, numbering_in=numbering, numbering_out=cls.numbering).nbits
        for ii, v in enumerate(arr):
            i = initialize_bitstring(integer=ii, nbits=maxbit, numbering_in=numbering, numbering_out=cls.numbering)
            if not isclose(abs(v), 0, atol=threshold):
                key = i if keymap is None else keymap(i)
                state[key] = v
        result =QubitWaveFunction(state)

        if cls.numbering != numbering:
            if cls.numbering == BitNumbering.MSB:
                result.apply_keymap(keymap=KeyMapLSB2MSB())
            else:
                result.apply_keymap(keymap=KeyMapMSB2LSB())

        return result

    @classmethod
    def from_int(cls, i: int, coeff=1):
        if isinstance(i, BitString):
            return QubitWaveFunction(state={i:coeff})
        else:
            return QubitWaveFunction(state={BitString.from_int(integer=i): coeff})

    def __repr__(self):
        result = str()
        for k, v in self.items():
            result += number_to_string(number=v) + "|" + str(k.binary) + "> "
        return result

    def __eq__(self, other):
        if len(self.state) != len(other.state):
            return False
        for k, v in self.state.items():
            if k not in other.state:
                return False
            elif not isclose(complex(v), complex(other.state[k]), atol=1.e-6):
                return False

        return True

    def __add__(self, other):
        result = QubitWaveFunction(state=copy.deepcopy(self._state))
        for k, v in other.items():
            if k in result._state:
                result._state[k] += v
            else:
                result._state[k] = v
        return result

    def __iadd__(self, other):
        for k, v in other.items():
            if k in self._state:
                self._state[k] += v
            else:
                self._state[k] = v
        return self

    def __rmul__(self, other):
        result = QubitWaveFunction(state=copy.deepcopy(self._state))
        for k, v in result._state.items():
            result._state[k] *= other
        return result

    def inner(self, other):
        # currently very slow and not optimized in any way
        result = 0.0
        for k, v in self.items():
            if k in other._state:
                result += v.conjugate()*other._state[k]
        return result


@dataclass
class SimulatorReturnType:

    abstract_circuit: QCircuit = None
    circuit: int = None
    wavefunction: QubitWaveFunction = None
    measurements: typing.Dict[str,QubitWaveFunction] = None
    backend_result: int = None

    @property
    def counts(self, key:str=None):
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

    def run(self, abstract_circuit: QCircuit, samples: int = 1) -> SimulatorReturnType:
        circuit = self.create_circuit(abstract_circuit=abstract_circuit)
        backend_result = self.do_run(circuit=circuit, samples=samples)
        return SimulatorReturnType(circuit=circuit,
                                   abstract_circuit=abstract_circuit,
                                   backend_result=backend_result,
                                   measurements=self.convert_measurements(backend_result))

    def do_run(self, circuit, samples: int = 1):
        raise OpenVQEException("run needs to be overwritten")

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
            if len(initial_state.keys())!=1:
                raise OpenVQEException("only product states as initial states accepted")
            initial_state = list(initial_state.keys())[0].integer

        active_qubits = abstract_circuit.qubits
        all_qubits = [i for i in range(abstract_circuit.n_qubits)]

        # maps from reduced register to full register
        keymap = KeyMapQubitSubregister(subregister=active_qubits, register=all_qubits)

        result = self.do_simulate_wavefunction(abstract_circuit=abstract_circuit,initial_state=keymap.inverted(initial_state).integer)
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

    def measure_objective(self, objective: Objective, samples:int = 1, return_simulation_data: bool=False) -> float:
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
            final_E += weight*E
            if return_simulation_data:
                data.append(tmp)
        if return_simulation_data:
            return final_E, data
        else:
            return final_E

    def measure_paulistring(self, abstract_circuit: QCircuit, paulistring, samples: int=1):
        # make basis change
        basis_change = QCircuit()
        for idx, p in paulistring.items():
            basis_change += change_basis(target=idx, axis=p)
        # make measurment instruction
        measure = QCircuit()
        qubits = [idx[0] for idx in paulistring.items()]
        measure *= Measurement(name=str(paulistring), target=qubits)
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
        assert (n_samples == samples) # failsafe
        E = E / samples * paulistring.coeff
        return (E, sim_result)

    def measure_paulistrings(self, abstract_circuit: QCircuit, paulistrings: list, samples: int = 1):
        """
        Simulate Circuit and measure PauliString
        All measurments performed in Z basis
        Basis changes are applied automatically
        :param abstract_circuit: The circuit
        :param paulistring: The PauliString in OVQE dataformat
        :return: Measurment
        """

        if isinstance(paulistrings, PauliString):
            paulistrings = [paulistrings]

        assembled = copy.deepcopy(abstract_circuit)
        for paulistring in paulistrings:
            # make basis change
            U_front = QCircuit()
            U_back = QCircuit()
            for idx, p in paulistring.items():
                U_front *= change_basis(target=idx, axis=p)
                U_back *= change_basis(target=idx, axis=p, daggered=True)

            # make measurment instruction
            measure = QCircuit()
            qubits = [idx[0] for idx in paulistring.items()]
            measure *= Measurement(name=str(paulistring), target=qubits)
            assembled *= U_front * measure * U_back

        sim_result = self.run(abstract_circuit=assembled, samples=samples)

        # post processing
        result = []
        for paulistring in paulistrings:
            measurements = sim_result.measurements[str(paulistring)] # TODO will work only for cirq -> Change
            E = 0.0
            n_samples = 0
            for key, count in measurements.items():
                parity = key.array.count(1)
                sign = (-1) ** parity
                E += sign * count
                n_samples += count
            assert (n_samples == samples)
            E = E / samples * paulistring.coeff
            result.append(E)

        return result
