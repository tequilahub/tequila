
from openvqe.circuit import *
from openvqe.objective import Objective
from openvqe.circuit.gradient import grad
from openvqe.circuit.gates import X, Ry, Rx, Rz
from openvqe.circuit._gates_impl import RotationGateImpl
from openvqe.hamiltonian import PX, PY, PZ
from openvqe.simulator.simulator_cirq import SimulatorCirq

from numpy import pi

if __name__ == "__main__":
    gate = Ry(target=1, control=3, angle=pi / 3, phase=1.0, frozen=False)

    gradient = grad(gate)
    print("gradient at of Ry at 'gate' level", gradient)

    #######
    # the following should not be done .... but would work anyway
    #######
    gate = RotationGateImpl(axis=1, angle=pi / 3, target=1, control=3)
    gradientx = grad(gate)
    print("gradient at of Ry at gate level", gradientx, " test:", gradient[0] == gradientx[0])

    ac = QCircuit()
    ac *= X(target=0, control=None)
    ac *= Ry(target=1, control=None, angle=pi / 2)

    obj = Objective(unitaries=ac, observable=None)
    gradient = grad(obj.unitaries[0])

    print("gradient of X, Ry at objective level", gradient)

    ac = QCircuit()
    ac *= X(target=0, power=2.3, phase=-1.0)
    ac *= Ry(target=1, control=0, angle=pi / 2)

    print('gradient of Xpower, controlled Ry at circuit level:', grad(ac))
    obj = Objective(unitaries=ac, observable=None)
    gradient = grad(obj.unitaries[0])

    print("gradient of Xpower controlled Ry at objective level", gradient)

    ac = QCircuit()
    ac *= X(0)
    ac *= Rx(target=2, angle=0.5, frozen=True)
    ac *= Ry(target=1, control=0, angle=pi / 2)
    ac *= Rz(target=1, control=[0, 2], angle=pi / 2)

    obj = Objective(unitaries=ac, observable=None)
    gradient = grad(obj.unitaries[0])

    print("gradient at objective level", gradient)

    from openvqe.circuit.gradient import grad

    print("new impl", grad(ac))

    """
    Doing <0|Ry(theta)_dag Sigma_Y Ry(theta)|0>
    and testing gradients
    """

    #H = PY(qubit=0) # with this Hamiltonian the example works fine
    H = 1j*PX(qubit=0)*PZ(qubit=0) # should give the same results

    print("The Hamiltonian is:\n")
    print(H)

    print("\n\nGradients of non-controlled gates work:\n")

    energies = []
    gradients = []
    angles = []
    for n in range(5):
        theta = n*pi/4
        U = Rx(target=0, angle=theta)
        print(U)
        result = SimulatorCirq().simulate_wavefunction(abstract_circuit=U)
        print(result)
        print(SimulatorCirq().create_circuit(abstract_circuit=U))
        O = Objective(observable=H, unitaries=U)
        E = SimulatorCirq().expectation_value(objective=O)
        energies.append(E)
        dE = SimulatorCirq().expectation_value(objective=grad(O)[0])
        print(grad(O)[0])
        gradients.append(dE)
        angles.append(theta)

    print("angles energies gradients")
    for i,a in enumerate(angles):
        print(angles[i], " ", energies[i], " ", gradients[i])

    from matplotlib import pyplot as plt
    plt.title("Gradient Test")
    plt.xlabel("t")
    plt.plot(angles, energies, label="E")
    plt.plot(angles, gradients, label="dE/dt")
    plt.legend()
    plt.show()

    print("\n\nSomething bizarre is happening in simulation:")
    energies = []
    gradients = []
    angles = []
    for n in range(5):
        theta = n*pi/4
        # multiplication syntax still follows circuits ... should probably change that at some point
        U = X(target=1)*Rx(target=0, control=1, angle=theta)
        # if you wanna see the states
        wfn = SimulatorCirq().simulate_wavefunction(abstract_circuit=U)
        # leaving this here to demonstrate how to use cirqs ascii print since we don't have our own right now
        #print(SimulatorCirq().create_circuit(abstract_circuit=U))
        O = Objective(observable=H, unitaries=U)
        E = SimulatorCirq().expectation_value(objective=O)
        energies.append(E)
        dE = SimulatorCirq().expectation_value(objective=grad(O)[0])
        print("gradient:\n", grad(O)[0])
        gradients.append(dE)
        angles.append(theta)

    print("what the hell!")
    for i,a in enumerate(angles):
        print(angles[i], " ", energies[i], " ", gradients[i])

    from matplotlib import pyplot as plt
    plt.title("Gradient Test for controlled Rotation")
    plt.xlabel("t")
    plt.plot(angles, energies, label="E")
    plt.plot(angles, gradients, label="dE/dt")
    plt.legend()
    plt.show()


    print("\n\nDemonstrating correct behaviour here:")
    from openvqe.circuit.compiler import compile_controlled_rotation_gate
    man_energies = []
    man_gradients = []
    man_angles = []
    for n in range(5):
        theta = n*pi/4
        print(theta)
        # multiplication syntax still follows circuits ... should probably change that at some point
        U = X(target=1)*Rx(target=0, control=1, angle=theta)
        U = compile_controlled_rotation_gate(gate=U) # that actually works
        # leaving this here to demonstrate how to use cirqs ascii print since we don't have our own right now
        print(SimulatorCirq().create_circuit(abstract_circuit=U))
        O = Objective(observable=H, unitaries=U)
        E = SimulatorCirq().expectation_value(objective=O)
        man_energies.append(E)

        dE0 = SimulatorCirq().expectation_value(objective=grad(O)[0])
        dE1 = SimulatorCirq().expectation_value(objective=grad(O)[1])
        print("gradient part 0\n", grad(O)[0])
        print("gradient part 1\n", grad(O)[1])
        man_gradients.append(-0.5*(dE0-dE1)) # with this it works
        man_angles.append(theta)

    print("Manual gradient is behaving as expected")
    for i,a in enumerate(angles):
        print(angles[i], " ", man_energies[i], " ", man_gradients[i])

    from matplotlib import pyplot as plt
    plt.title("Gradient Test for controlled Rotation -- manually")
    plt.xlabel("t")
    plt.plot(angles, man_energies, label="E")
    plt.plot(angles, man_gradients, label="dE/dt")
    plt.legend()
    plt.show()

    plt.title('Automatic and Manual gradients on the same plot')
    plt.xlabel("t")
    plt.plot(angles, gradients, label="Automatic")
    plt.plot(angles, man_gradients, label="Manual")
    plt.legend()
    plt.show()

    plt.title('Automatic and Manual energies on the same plot')
    plt.xlabel("t")
    plt.plot(angles, energies, label="Automatic")
    plt.plot(angles, man_energies, label="Manual")
    plt.legend()
    plt.show()