from .ansatz_base import AnsatzBase
from openvqe.parameters import ParametersUCC


class AnsatzUCC(AnsatzBase):
    """
    Class for UCC ansatz
    """

    def __call__(self):
        raise NotImplementedError()