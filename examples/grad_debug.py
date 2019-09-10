from openvqe.circuit import *
from openvqe.objective import Objective
from openvqe.circuit.gradient import grad
from openvqe.circuit.gates import X, Ry, Rx, Rz
from openvqe.circuit._gates_impl import RotationGateImpl
from openvqe.hamiltonian import HamiltonianBase
from openfermion import QubitOperator
from openvqe.simulator.simulator_cirq import SimulatorCirq

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

from numpy import pi

if __name__ == "__main__":

    H = MyQubitHamiltonian()
    H.parameters.transformation = "my_trafo"
    theta =pi

    U = X(target=1)*Ry(target=0, control=1, angle=theta)
    # if you wanna see the states
    wfn = SimulatorCirq().simulate_wavefunction(abstract_circuit=U)
    # leaving this here to demonstrate how to use cirqs ascii print since we don't have our own right now
    print(SimulatorCirq().create_circuit(abstract_circuit=U))
    O = Objective(observable=H, unitaries=U)
    E = SimulatorCirq().expectation_value(objective=O)
    dE = SimulatorCirq().expectation_value(objective=grad(O)[0])
    for i in grad(O)[0].unitaries:
        print(SimulatorCirq().create_circuit(abstract_circuit=i)," ",i.weight)
    print("Gradient fails!")
    print(theta, " ", E, " ", dE)



    from matplotlib import pyplot as plt


    print("Demonstrating correct behaviour here:")
    from openvqe.circuit.compiler import compile_controlled_rotation_gate
    energies = []
    gradients = []
    angles = []
    for n in range(2):
        theta = n*pi/4
        # multiplication syntax still follows circuits ... should probably change that at some point
        U = X(target=1)*Ry(target=0, control=1, angle=theta)
        U = compile_controlled_rotation_gate(gate=U) # that actually works
        # leaving this here to demonstrate how to use cirqs ascii print since we don't have our own right now
        print(SimulatorCirq().create_circuit(abstract_circuit=U))
        O = Objective(observable=H, unitaries=U)
        E = SimulatorCirq().expectation_value(objective=O)
        energies.append(E)
        dE0 = SimulatorCirq().expectation_value(objective=grad(O)[0])
        dE1 = SimulatorCirq().expectation_value(objective=grad(O)[1])
        gradients.append(0.5*(dE0-dE1)) # with this it works
        angles.append(theta)

    print("Gradient works again")
    for i,a in enumerate(angles):
        print(angles[i], " ", energies[i], " ", gradients[i])

    from matplotlib import pyplot as plt
    plt.title("Gradient Test for controlled Rotation -- manually")
    plt.xlabel("t")
    plt.plot(angles, energies, label="E")
    plt.plot(angles, gradients, label="dE/dt")
    plt.legend()
    plt.show()
