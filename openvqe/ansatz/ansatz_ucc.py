from openvqe.abc import parametrized
from .ansatz_base import AnsatzBase
from openvqe.exceptions import OpenVQEParameterError
from openvqe import HamiltonianQC
import numpy
import openfermion
from dataclasses import dataclass
from openvqe.ansatz.ansatz_base import ParametersAnsatz
from openvqe.tools.convenience import binary_to_number, number_to_binary


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
        return self.make_cluster_operator(angles=angles)

    def initial_state(self) -> int:
        """
        :return: Hatree-Fock Reference as binary-number
        """
        l = [0]*self.hamiltonian.n_qubits()
        for i in range(self.hamiltonian.n_electrons()):
            l[i] = 1

        return binary_to_number(l=l)

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
            double_amplitudes = angles.two_body

        op = openfermion.utils.uccsd_generator(
            single_amplitudes=single_amplitudes,
            double_amplitudes=double_amplitudes
        )

        if self.hamiltonian.parameters.transformation.upper() == "JW":
            return openfermion.jordan_wigner(op)
        elif self.hamiltonian.parameters.transformation.lower() == "BK":
            # @todo opernfermion has some problems with bravyi_kitaev and interactionoperators
            return openfermion.bravyi_kitaev(op)
        else:
            raise OpenVQEParameterError(parameter_name="transformation",
                                        parameter_class=type(self.hamiltonian.parameters).__name__,
                                        parameter_value=self.hamiltonian.parameters.transformation)

    def verify(self) -> bool:
        from openvqe import OpenVQETypeError
        """
        Overwritten verify function to check specificly for ParametersQC type
        :return:
        """
        # do some verification specifically for this class

        # check if the hamiltonian is the right type and call its own verify function
        if not isinstance(self.hamiltonian, HamiltonianQC):
            raise OpenVQETypeError(attr=type(self).__name__ + ".hamiltonian", expected=type(HamiltonianQC).__name__,
                                   type=type(self.hamiltonian).__name__)

        # do the standard checks for the baseclass
        return self._verify()
