from openvqe import OpenVQEParameters, OpenVQEModule
from openvqe.openvqe_abc import parametrized
from dataclasses import dataclass
from openvqe.circuit.compiler import compile_controlled_rotation_gate

@dataclass
class ParametersHamiltonian(OpenVQEParameters):
    pass

@parametrized(ParametersHamiltonian)
class Observable(OpenVQEModule):

    unitary: list = None # list of QCircuit objects defining the state preparation
    observable: list = None # will be list of pauli strings


def make_gradient(observable: Observable) -> list:
    """
    Returns the Gradient of an observable as a list of observables
    :param observable: The original observable
    :return: The list of observables corresponding to the gradient of the unfrozen parameters
    """
    raise NotImplementedError("not implemented yet")
    pass


