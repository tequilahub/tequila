import numpy
import openfermion
from openvqe.hamiltonian import QubitHamiltonian
from openvqe.circuit.exponential_gate import QCircuit
from openvqe import typing


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

    def __rmul__(self, other):
        return ManyBodyAmplitudes(one_body=other * self.one_body, two_body=other * self.two_body)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.one_body) + len(self.two_body)


class AnsatzUCC:
    """
    Class for UCC ansatz
    """

    def __init__(self, decomposition: typing.Callable = None, transformation: typing.Callable = None):
        self._decomposition = decomposition
        if transformation is None or transformation in ["JW", "Jordan-Wigner", "jordan-wigner", "jw"]:
            self._transformation = openfermion.jordan_wigner
        else:
            self._transformation = transformation

    def __call__(self, angles) -> QCircuit:
        generator = 1.0j * QubitHamiltonian(hamiltonian=self.make_cluster_operator(angles=2.0*angles))
        return self._decomposition(generators=[generator])

    def make_cluster_operator(self, angles: ManyBodyAmplitudes) -> openfermion.QubitOperator:
        """
        Creates the clusteroperator
        :param angles: CCSD amplitudes
        :return: UCCSD Cluster Operator as QubitOperator
        """

        if angles.one_body is not None:
            single_amplitudes = angles.one_body
        if angles.two_body is not None:
            double_amplitudes = angles.two_body

        op = openfermion.utils.uccsd_generator(
            single_amplitudes=single_amplitudes,
            double_amplitudes=double_amplitudes
        )

        # @todo opernfermion has some problems with bravyi_kitaev and interactionoperators
        return self._transformation(op)

    def verify(self) -> bool:
        from openvqe import OpenVQETypeError
        """
        Overwritten verify function to check specificly for ParametersQC type
        :return:
        """

        # do the standard checks for the baseclass
        return self._verify()
