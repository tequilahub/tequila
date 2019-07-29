from openvqe import OpenVQEModule, OpenVQEException
from openvqe.circuit.circuit import QCircuit
from openvqe.tools.convenience import number_to_binary
from numpy import isclose

class SimulatorReturnType:

    def __init__(self, result=None, circuit=None, abstract_circuit: QCircuit=None, *args, **kwargs):
        self.circuit = circuit
        self.abstract_circuit = abstract_circuit
        self.wavefunction = None
        self.density_matrix = None
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
            qubits = self.abstract_circuit.max_qubit()
        if self.wavefunction is not None:
            for i,v in enumerate(self.wavefunction):
                if not isclose(abs(v),0):
                    result += str(v) + "|" + str(number_to_binary(number=i, bits=qubits)) + ">"
            return result
        elif self.density_matrix is not None:
            return str(self.density_matrix)
        else:
            return str(self)

class Simulator(OpenVQEModule):
    """
    Base Class for OpenVQE interfaces to simulators
    """

    def simulate_wavefunction(self, abstract_circuit: QCircuit, returntype=SimulatorReturnType,
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
        result.abstract_circuit=abstract_circuit

        if callable(returntype):
            return returntype(result=result, circuit=circuit, abstract_circuit=abstract_circuit)
        else:
            return result


    def do_simulate_wavefunction(self, circuit, initial_state=0):
        raise OpenVQEException(
            "called from base class of simulator, or non-supported operation for this backend")

    def simulate_density_matrix(self, abstract_circuit: QCircuit, returntype=SimulatorReturnType, initial_state=0):
        """
        Translates the abstract circuit to the backend which is specified by specializations of this class
        :param abstract_circuit: the abstract circuit
        :param returntype: if callable a class which wraps the return values otherwise the plain result of the backend simulator is given back
        see the SimulatorReturnType baseclass
        :return: The density matrix of the simulation
        """

        circuit = self.create_circuit(abstract_circuit=abstract_circuit)
        result = self.do_simulate_density_matrix(circuit=circuit, initial_state=initial_state)
        result.abstract_circuit=abstract_circuit

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
