from openvqe.circuit import gates
from openvqe.circuit import Variable
from openvqe.hamiltonian import paulis
from openvqe.optimizers import GradientDescent
from openvqe.simulator.simulator_qulacs import SimulatorQulacs
from openvqe.simulator.simulator_cirq import SimulatorCirq
from openvqe.simulator.simulator_symbolic import SimulatorSymbolic
from openvqe.simulator.simulator_qiskit import SimulatorQiskit
from openvqe.simulator.simulator_pyquil import SimulatorPyquil
if __name__ == "__main__":

    # parameters with explanation:
    samples = None # number of samples for each run, None means full wavefunction simulation
    simulator = SimulatorQulacs # pick the simulator, None means it is automatically picked
    stepsize = 0.1 # stepsize for each update step in gradient descent
    maxiter = 100

    optimizer = GradientDescent(samples=samples, simulator=simulator, stepsize=stepsize, maxiter=maxiter)

    a = Variable(name="a", value=1.0)
    b = Variable(name="b", value=2.0)

    H = paulis.X(0)
    U = gates.Ry(target=0, angle=a)
    U += gates.Ry(target=1, control=0,  angle=b)
   # U += gates.X(target=1, control=0)

    angles = U.extract_parameters()

    optimizer.solve(U,H, angles=angles)

    optimizer.plot(plot_energies=True, plot_gradients=["a", "b"])



