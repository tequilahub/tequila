from openvqe.abc import  parametrized
from .ansatz_base import AnsatzBase
from openvqe.exceptions import OpenVQEParameterError
from openvqe import HamiltonianQC
import numpy
import openfermion
from dataclasses import dataclass
from openvqe.ansatz.ansatz_base import ParametersAnsatz


@dataclass
class ParametersUCC(ParametersAnsatz):
    # UCC specific parameters
    # have to be assigned
    decomposition: str = "trotter"
    trotter_steps: int = 1


class ManyBodyAmplitudes:
    """
    Class which stores ManyBodyAmplitudes
    """

    def __init__(self, one_body: numpy.ndarray = None, two_body: numpy.ndarray = None):
        self.one_body = one_body
        self.two_body = two_body

    def __str__(self):
        rep = type(self).__name__
        rep += "\n One-Body-Terms:\n"
        rep += str(self.one_body)
        rep += "\n Two-Body-Terms:\n"
        rep += str(self.two_body)
        return rep

    def __repr__(self):
        return self.__str__()

@parametrized(ParametersAnsatz)
class AnsatzUCC(AnsatzBase):
    """
    Class for UCC ansatz
    """

    def __call__(self, angles):
        circuit = self.backend_handler.init_circuit()
        circuit += self.prepare_reference()
        circuit += self.prepare_ucc(angles=angles)
        return circuit

    def prepare_reference(self):
        """
        Apply an X gate to all occupied reference functions
        which are given by the first self.parameters.n_electrons qubits
        so right now we are assuming HF as reference
        :return: A circuit which prepares the reference function (right now only closed shell HF)
        """
        circuit = self.backend_handler.init_circuit()
        for i in range(self.parameters.n_qubits):
            if i < self.hamiltonian.n_electrons():
                circuit += self.backend_handler(name="X", targets=[i])
            else:
                circuit += self.backend_handler(name="I", targets=[i])

        return circuit

    def prepare_ucc(self, angles):
        if self.parameters.decomposition == "trotter":
            return self.prepare_ucc_trotter(angles=angles)
        else:
            raise OpenVQEParameterError(parameter_name="decomposition", parameter_value=self.parameters.decomposition, parameter_class=type(self.parameters).__name__, called_from=type(self).__name__)

    def prepare_ucc_trotter(self, angles):
        cluster_operator = self.make_cluster_operator(angles=angles)
        circuit = self.backend_handler.init_circuit()
        for index in range(int(self.parameters.trotter_steps)):
            factor = 1j / self.parameters.trotter_steps
            for key, value in cluster_operator.terms.items():
                if key == ():
                    # dont implement the constant part
                    continue
                elif not numpy.isclose(value, 0.0, rtol=1.e-8, atol=1.e-8):
                    # don;t make circuit for too small values
                    # @todo include ampltidude_neglect_threshold into parameters
                    circuit += self.exponential_pauli_gate(paulistring=key, angle=value * factor)
        return circuit

    # @todo: considering moving functions like those into a 'compiler' module
    def exponential_pauli_gate(self, paulistring, angle):
        """
        Returns the circuit: exp(i*angle*paulistring)
        where
        :param paulistring: The paulistring in given as tuple of tuples (openfermion format)
        like e.g  ( (0, 'Y'), (1, 'X'), (5, 'Z') )
        :param angle: The angle which parametrizes the gate -> should be real
        :returns: the above mentioned circuit in pyquil or cirq format depending
        on the backend choosen in self.parameters.backend
        """

        if not numpy.isclose(numpy.imag(angle), 0.0):
            raise Warning("angle is not real, angle=" + str(angle))

        circuit = self.backend_handler.init_circuit()

        # the general circuit will look like:
        # series which changes the basis if necessary
        # series of CNOTS associated with basis changes
        # Rz gate parametrized on the angle
        # series of CNOT (inverted direction compared to before)
        # series which changes the basis back
        change_basis = self.backend_handler.init_circuit()
        change_basis_back = self.backend_handler.init_circuit()
        cnot_cascade = self.backend_handler.init_circuit()
        rz_operations = self.backend_handler.init_circuit()
        reversed_cnot = self.backend_handler.init_circuit()

        last_qubit = None
        previous_qubit = None
        for pq in paulistring:
            pauli = pq[1]
            qubit = [pq[0]] # wrap in list for targets= ...

            # see if we need to change the basis
            if pauli.upper() == "X":
                change_basis += self.backend_handler(name="H", targets=qubit)
                change_basis_back += self.backend_handler(name="H", targets=qubit)
            elif pauli.upper() == "Y":
                change_basis += self.backend_handler(name="Rx", targets=qubit, angle=numpy.pi / 2)
                change_basis_back += self.backend_handler(name="Rx", targets=qubit, angle=-numpy.pi / 2)

            if previous_qubit is not None:
                cnot_cascade += self.backend_handler(name="CNOT", targets=qubit, controls=previous_qubit)

            previous_qubit = qubit
            last_qubit = qubit

        for cn in reversed(cnot_cascade):
            reversed_cnot += self.backend_handler.wrap_gate(cn)

        # assemble the circuit
        circuit += change_basis
        circuit += cnot_cascade
        # factor 2 is since gates are defined with angle/2
        circuit += self.backend_handler(name="Rz", angle=2.0*angle, targets=last_qubit)
        circuit += reversed_cnot
        circuit += change_basis_back

        return circuit

    def make_cluster_operator(self, angles: ManyBodyAmplitudes) -> openfermion.QubitOperator:
        """
        Creates the clusteroperator
        :param angles: CCSD amplitudes
        :return: UCCSD Cluster Operator as QubitOperator
        """
        nq = self.hamiltonian.n_qubits()
        # double angles are expected in iajb form
        single_amplitudes = numpy.zeros([nq, nq])
        double_amplitudes = numpy.zeros([nq, nq, nq, nq])
        if angles.one_body is not None:
            single_amplitudes = angles.one_body
        if angles.two_body is not None:
            double_amplitudes = angles.two_body  # (psi4 parser from opernfermion includes factor 1/2, but the uccsd_singlet_generator excpets the amplitudes only

        # #openfermions uccsd_singlet_generator does not generate singlets but broken symmetries
        # op = openfermion.utils.uccsd_singlet_generator(
        #     packed_amplitudes=openfermion.utils.uccsd_singlet_get_packed_amplitudes(
        #         single_amplitudes=single_amplitudes,
        #         double_amplitudes=double_amplitudes,
        #         n_qubits=nq,
        #         n_electrons=self.hamiltonian.n_electrons()
        #     ),
        #     n_qubits=nq, n_electrons=self.hamiltonian.n_electrons())

        op = openfermion.utils.uccsd_generator(
                single_amplitudes=single_amplitudes,
                double_amplitudes=double_amplitudes
            )

        if self.hamiltonian.parameters.transformation.upper() == "JW":
            return openfermion.jordan_wigner(op)
        elif self.hamiltonian.parameters.transformation.lower() == "BK":
            # @todo opernfermion has some problems with bravyi_kitaev and interactionoperators
            return openfermion.bravyi_kitaev(op)
        else :
            raise OpenVQEParameterError(parameter_name="transformation", parameter_class=type(self.hamiltonian.parameters).__name__, parameter_value=self.hamiltonian.parameters.transformation)

    def verify(self) -> bool:
        from openvqe import OpenVQETypeError
        """
        Overwritten verify function to check specificly for ParametersQC type
        :return:
        """
        # do some verification specifically for this class

        # check if the hamiltonian is the right type and call its own verify function
        if not isinstance(self.hamiltonian, HamiltonianQC):
            raise OpenVQETypeError(attr=type(self).__name__ + ".hamiltonian", expected=type(HamiltonianQC).__name__, type=type(self.hamiltonian).__name__)

        # do the standard checks for the baseclass
        return self._verify()