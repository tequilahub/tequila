import numpy
import openfermion
from openvqe.hamiltonian import QubitHamiltonian
from openvqe.circuit.exponential_gate import QCircuit
from openvqe import typing


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

    def __call__(self, i, a, j=None, b=None, *args, **kwargs):
        """
        :param i: in absolute numbers (as spin-orbital index)
        :param a: in absolute numbers (as spin-orbital index)
        :param j: in absolute numbers (as spin-orbital index)
        :param b: in absolute numbers (as spin-orbital index)
        :return: amplitude t_aijb
        """
        if j is None:
            assert (b is None)
            return self.one_body[a, i]
        else:
            return self.two_body[a, i, b, j]

    def __getitem__(self, item: tuple):
        return self.__call__(*item)

    def __setitem__(self, key: tuple, value):
        if len(key) == 2:
            self.one_body[key[0], key[1]] = value
        else:
            self.two_body[key[0], key[1], key[2], key[3]] = value
        return self

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
        generator = 1.0j * QubitHamiltonian(hamiltonian=self.make_cluster_operator(angles=2.0 * angles))
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
