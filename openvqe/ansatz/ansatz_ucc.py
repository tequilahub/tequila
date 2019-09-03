from openvqe.openvqe_abc import parametrized
from .ansatz_base import AnsatzBase
from openvqe.openvqe_exceptions import OpenVQEParameterError
import numpy
import openfermion
from dataclasses import dataclass
from openvqe.ansatz.ansatz_base import ParametersAnsatz
from openvqe.tools.convenience import binary_to_number, number_to_binary


@dataclass
class ParametersUCC(ParametersAnsatz):
    # UCC specific parameters
    # have to be assigned
    transformation = "JW"


class ManyBodyAmplitudes:
    """
    Class which stores ManyBodyAmplitudes
    """

    @staticmethod
    def convert_array(asd):
        # dummy for now
        return asd

    def __init__(self, one_body: numpy.ndarray = None, two_body: numpy.ndarray = None):
        self.one_body = self.convert_array(one_body)
        self.two_body = self.convert_array(two_body)

    def __str__(self):
        rep = type(self).__name__
        rep += "\n One-Body-Terms:\n"
        rep += str(self.one_body)
        rep += "\n Two-Body-Terms:\n"
        rep += str(self.two_body)
        return rep

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.one_body) + len(self.two_body)


@parametrized(ParametersUCC)
class AnsatzUCC(AnsatzBase):
    """
    Class for UCC ansatz
    """

    def __call__(self, angles):
        return self.make_cluster_operator(angles=angles)

    def initial_state(self, hamiltonian) -> int:
        """
        :return: Hatree-Fock Reference as binary-number
        """
        l = [0]*hamiltonian.n_qubits()
        for i in range(hamiltonian.n_electrons()):
            l[i] = 1

        l = [i for i in reversed(l)]
        return binary_to_number(l=l)

    def make_cluster_operator(self, angles: ManyBodyAmplitudes) -> openfermion.QubitOperator:
        """
        Creates the clusteroperator
        :param angles: CCSD amplitudes
        :return: UCCSD Cluster Operator as QubitOperator
        """
        #nq = self.hamiltonian.n_qubits()
        # double angles are expected in iajb form
        #single_amplitudes = numpy.zeros([nq, nq])
        #double_amplitudes = numpy.zeros([nq, nq, nq, nq])
        if angles.one_body is not None:
            single_amplitudes = angles.one_body
        if angles.two_body is not None:
            double_amplitudes = angles.two_body

        op = openfermion.utils.uccsd_generator(
            single_amplitudes=single_amplitudes,
            double_amplitudes=double_amplitudes
        )

        if self.parameters.transformation.upper() == "JW":
            return openfermion.jordan_wigner(op)
        elif self.parameters.transformation.lower() == "BK":
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

        # do the standard checks for the baseclass
        return self._verify()
