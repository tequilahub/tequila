from openvqe.objective import Objective
from openvqe.openvqe_abc import parametrized
from openvqe.hamiltonian import HamiltonianBase
from openfermion import QubitOperator
from openvqe.circuit.gates import Ry, X, Y
from numpy import pi, sqrt, asarray, exp
import numpy
from openvqe.simulator.simulator_cirq import SimulatorCirq


# make an easy Hamiltonian without invoking to much stuff
# it will just be H = sigma_x(0)
# just ignore this whole class for this example and a bit of a Hack right now .... will make that easier in the future
# The whole Hamiltonian structure is too much tailored towards fermionic Hamiltonians
# Will change soon
class MyQubitHamiltonian(HamiltonianBase):

    # Hamiltonian needs to be aware of this
    def n_qubits(self):
        return 1

    def my_trafo(self, H):
        """
        Here we define the hamiltonian to be already in qubit form, so no transformation will be needed
        """
        return H

    def get_fermionic_hamiltonian(self):
        H = QubitOperator()
        H.terms[((0, 'X'),)] = 1.0
        return H


class SqrtObjective(Objective):
    """
    A very simple example on how we could change the objective class
    to return the sqrt of the default
    """

    def objective_function(self, values, weights=None):
        if weights is None:
            weights = [1] * len(values)

        values = asarray(values)
        weights = asarray(weights)

        return sqrt(values.dot(weights))

class DiffObjective(Objective):
    """
    Workaround for this example since the weights are not yet available in the main branch
    ... just ignore
    """

    def objective_function(self, values, weights=None):
        if weights is None:
            weights = [1] * len(values)

        assert (len(values)==2)
        weights[1]=-1

        values = asarray(values)
        weights = asarray(weights)

        return sqrt(values.dot(weights))


class NumpyObjective(Objective):
    """
    More advanced example on customly designed objective classes
    """

    functionname: str

    def objective_function(self, values, weights=None):
        if weights is None:
            weights = [1] * len(values)

        values = asarray(values)
        weights = asarray(weights)

        f = getattr(numpy, self.functionname)
        return f(values.dot(weights))


if __name__ == "__main__":
    # first initialize a Hamiltonian
    hamiltonian = MyQubitHamiltonian()
    hamiltonian.parameters.transformation = "my_trafo"

    print("This is the Hamiltonian")
    print(hamiltonian())

    # Initialize a one-Qubit unitary
    U = Ry(target=0, angle=pi / 4)

    # Initialize the simulator (using cirq here)
    # Changing the simulator backend can be done be changing only this line here
    # (does not work for pyquil right now since I hacked the expectation_value function in for cirq, not generalized yet)
    simulator = SimulatorCirq()

    # lets simulate and print the state the unitary gives us
    # just to print out how the state looks
    state = simulator.simulate_wavefunction(abstract_circuit=U)
    print("State is: ", state)

    # Initialize an objective
    O = Objective(observable=hamiltonian, unitaries=U)

    # Now pass the objective to the simulator
    E = simulator.expectation_value(objective=O)
    print("Expectation Value is = ", E, " it should be ", 1.0 / sqrt(2))

    # Same thing with The SqrtObjective function
    O = SqrtObjective(observable=hamiltonian, unitaries=U)
    E = simulator.expectation_value(objective=O)
    print("Expectation Value is = ", E, " it should be ", sqrt(1.0 / sqrt(2)))

    # Same thing with The NumpyObjective function
    O = NumpyObjective(observable=hamiltonian, unitaries=U)
    O.functionname = "sqrt"
    E = simulator.expectation_value(objective=O)
    print("Expectation Value is = ", E, " it should be ", sqrt(1.0 / sqrt(2)))
    O.functionname = "exp"
    E = simulator.expectation_value(objective=O)
    print("Expectation Value is = ", E, " it should be ", exp(1.0 / sqrt(2)))

    # Now we are preparing two different unitaries and summing the result up
    # i.e. we are computing: <0|XXX|0> + <0|YXY|0> = <1|x|1> + (-i)^2<1|x|1> = 1 -1 = 0
    # (middle X is the hamiltonian in this simple example)
    U1 = X(0)
    U2 = Y(0)
    O = Objective(observable=hamiltonian, unitaries=[U1, U2])
    E = simulator.expectation_value(objective=O)
    print("Expectation Value is = ", E, " it should be  1-1 = 0")

    # Now the gradient for the first example
    # Here we have only one parameter so the gradient is a scalar
    # but this is how the individual components of the gradient can be evaluated
    dU1 = Ry(target=0, angle=pi / 4 + pi)
    dU2 = Ry(target=0, angle=pi / 4 - pi)
    # DiffObjective is just a workaround in this example since the weights in the unitaries are not yet merged
    O = DiffObjective(observable=hamiltonian, unitaries=[dU1, dU2])
    dE = simulator.expectation_value(objective=O)
    print("gradient is = ", dE, " it should be 0")

