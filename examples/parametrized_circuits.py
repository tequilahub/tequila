from openvqe.circuit import gates
from openvqe.circuit import Variable
from openvqe.hamiltonian import paulis
from openvqe.simulator.simulator_qiskit import SimulatorQiskit
from openvqe.simulator.simulator_qulacs import SimulatorQulacs
from openvqe.simulator.simulator_cirq import SimulatorCirq
from openvqe.objective import Objective
from openvqe.circuit.gradient import grad
from openvqe.optimizers import GradientDescent

if __name__ == "__main__":

    optimizer = GradientDescent()
    a = Variable(name="a", value=3.0)
    b = Variable(name="b", value=2.0)

    H = paulis.X(1)
    U = gates.Ry(target=0, angle=a)
    U += gates.Ry(target=1, control=0, angle=b)

    simulator = SimulatorCirq()

    angles = U.extract_parameters()

    for iter in range(100):

        O = Objective(unitaries=U, observable=H)
        E = simulator.simulate_objective(objective=O)

        dO = grad(O)

        dE = dict()
        for k, dOi in dO.items():
            dE[k] = simulator.simulate_objective(objective=dOi)

        print("E     =",E)
        print("dE    =",dE)
        print("angles=", angles)
        angles = optimizer(angles=angles, energy=E, gradient=dE)
        U.update_parameters(parameters=angles)

    optimizer.plot(plot_energies=True, plot_gradients=["a", "b"])



