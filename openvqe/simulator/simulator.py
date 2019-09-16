from openvqe import OpenVQEModule, OpenVQEException
from openvqe.circuit.circuit import QCircuit
from openvqe.tools.convenience import number_to_binary
from openvqe.hamiltonian import PauliString
from numpy import isclose
from openvqe.circuit.compiler import change_basis
from openvqe.circuit.gates import Measurement
import copy


class MeasurementResultType:
    """
    Measurement result container
    Holds measurement as dictionary (name of measurement and corresponding results)
    Results are stored as dictionary of computational basis states and their count as value
    """

    def __getitem__(self, item):
        return self._result[item]

    def __init__(self):
        self._result = {}

    def __setitem__(self, key, value):
        self._result[key] = value

    def __repr__(self):
        return str(self._result)


class SimulatorReturnType:

    def __init__(self, result=None, circuit=None, abstract_circuit: QCircuit = None, *args, **kwargs):
        self.circuit = circuit
        self.abstract_circuit = abstract_circuit
        self.wavefunction = None
        self.density_matrix = None
        self.measurements = None
        self.backend_result = None
        self.__post_init__(result)

    def __post_init__(self, result):
        """
        Overwrite this function in the specific backends to get wavefunctions, density matrices and whatever you want
        """
        self.result = result

    def __repr__(self):
        qubits = 0
        result = ""
        if self.abstract_circuit is not None:
            qubits = self.abstract_circuit.n_qubits

        if self.wavefunction is not None:
            for i, v in enumerate(self.wavefunction):
                if not isclose(abs(v), 0, atol=1.e-5):
                    if isclose(v.imag, 0, atol=1.e-5):
                        result += '+({0.real:.4f})'.format(v) + "|" + str(
                            number_to_binary(number=i, bits=qubits)) + ">"
                    elif isclose(v.real, 0, atol=1.e-5):
                        result += '+({0.imag:.4f}i)'.format(v) + "|" + str(
                            number_to_binary(number=i, bits=qubits)) + ">"
                    else:
                        result += '+({0.real:.4f} + {0.imag:.4f}i)'.format(v) + "|" + str(
                            number_to_binary(number=i, bits=qubits)) + ">"
            return result
        elif self.density_matrix is not None:
            return str(self.density_matrix)
        else:
            return "None"


class Simulator(OpenVQEModule):
    """
    Base Class for OpenVQE interfaces to simulators
    """

    def run(self, abstract_circuit: QCircuit, samples: int = 1):
        circuit = self.create_circuit(abstract_circuit=abstract_circuit)
        result = self.do_run(circuit=circuit, samples=samples)
        result.abstract_circuit = abstract_circuit
        return result

    def do_run(self, circuit, samples: int = 1):
        raise OpenVQEException("run needs to be overwritten")

    def simulate_wavefunction(self, abstract_circuit: QCircuit, returntype=None,
                              initial_state: int = 0):
        """
        Simulates an abstract circuit with the backend specified by specializations of this class
        :param abstract_circuit: The abstract circuit
        :param returntype: specifies how the result should be given back
        :param initial_state: The initial state of the simulation,
        if given as an integer this is interpreted as the corresponding multi-qubit basis state
        :return: The resulting state
        """

        circuit = self.create_circuit(abstract_circuit=abstract_circuit)
        result = self.do_simulate_wavefunction(circuit=circuit, initial_state=initial_state)
        result.abstract_circuit = abstract_circuit

        return result

    def do_simulate_wavefunction(self, circuit, initial_state=0):
        raise OpenVQEException(
            "called from base class of simulator, or non-supported operation for this backend")

    def simulate_density_matrix(self, abstract_circuit: QCircuit, returntype=None, initial_state=0):
        """
        Translates the abstract circuit to the backend which is specified by specializations of this class
        :param abstract_circuit: the abstract circuit
        :param returntype: if callable a class which wraps the return values otherwise the plain result of the backend simulator is given back
        see the SimulatorReturnType baseclass
        :return: The density matrix of the simulation
        """

        circuit = self.create_circuit(abstract_circuit=abstract_circuit)
        result = self.do_simulate_density_matrix(circuit=circuit, initial_state=initial_state)
        result.abstract_circuit = abstract_circuit

        if callable(returntype):
            return returntype(result=result, circuit=circuit, abstract_circuit=abstract_circuit)
        else:
            return result

    def do_simulate_density_matrix(self, circuit, initial_state=0):
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
            qubits=[idx[0] for idx in paulistring.items()]
            measure *= Measurement(name=str(paulistring), target=qubits)
            assembled *= U_front * measure * U_back

        sim_result = self.run(abstract_circuit=assembled, samples=samples)

        # post processing
        result = []
        print("circuit=", sim_result.circuit)
        for paulistring in paulistrings:
            measurements = sim_result.measurements[str(paulistring)]
            E = 0.0
            n_samples = 0
            for key, count in measurements.items():
                parity = number_to_binary(key).count(1)
                sign = (-1)**parity
                E += sign*count
                n_samples += count
            assert(n_samples==samples)
            E = E/samples*paulistring.coeff
            result.append(E)

        return result




