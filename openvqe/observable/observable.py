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


def make_gradient(observable: Observable, mode: Bool = 0) -> list:
    """
    Returns the Gradient of an observable as a list of (lists of tuples of) observables.
    :param observable: The original observable
    :return: The list of observables corresponding to the gradient of the unfrozen parameters
    """
    circuit_grads=[]
    gradient=[]
    for circuit in observable.unitary:
        variables=circuit.extract_angles()
        indices=[v[0] for v in variables]
        grad_list=[circuit.get_gradient(i) for i in indices]
        circuit_grads.append[grad_list]

        #### need to do something here about the operation which concatenates the gradients of different circuits.
        #### arbitrary choice: modes. Mode 0: prepared to take the mean. Mode 1: do nothing
        #### a little bit difficult to just reverse from shape (#circuits,#variables) to (#variables,#circuits) due to the fact that this isn't a numpy array.

    if mode == 0:
        for j in range(len(circuit_grads[0])):
            new_list=[]
            for grad in grad_circuits:
                new_list.append(grad[j])
            gradient.append(new_list)


    if mode == 1:
        gradient = circuit_grads
        
    return gradient


