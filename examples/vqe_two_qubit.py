from openvqe.circuit import gates
from openvqe.circuit import Variable
from openvqe.hamiltonian import paulis
from openvqe.objective import Objective
from openvqe.optimizers import GradientDescent

"""
A very simple example for a two qubit gradient descent optimization

Keynotes:
- Initialization of variables
- Rough demonstration of how the optimizers might work

Play around with stepsize, iterations and initial values
- The true minimum is at -1
- The true maximum is at +1
- there is a stationary point at a=0 and b=0 and others

See vqe_two_qubit_alternativ.py for an alternative string based initialization

"""

# uncomment if you want to use a specific simulator
# from openvqe.simulator.simulator_cirq import SimulatorCirq
# from openvqe.simulator.simulator_qiskit import SimulatorQiskit
# from openvqe.simulator.simulator_qulacs import SimulatorQulacs
# from openvqe.simulator.simulator_pyquil import SimulatorPyquil

# parameters with explanation:
samples = None      # number of samples for each run, None means full wavefunction simulation
simulator = None    # pick the simulator, None means it is automatically picked. Does not need to be initialized
stepsize = 0.1      # stepsize for each update step in gradient descent
maxiter = 200       # max number of iterations

if __name__ == "__main__":

    optimizer = GradientDescent(samples=samples, simulator=simulator, stepsize=stepsize, maxiter=maxiter, minimize=True)

    # initialize Variables with initial values
    a = Variable(name="a", value=4.0)
    b = Variable(name="b", value=2.0)

    # initialize the Hamiltonian
    H = paulis.X(1)

    # initialize the parametrized Circuit
    U = gates.Ry(target=0, angle=-a/2, frozen=False) # frozen=true: this variable will not be optimized
    U += gates.Ry(target=0, angle=-a/2) # will behave the same as only one time Ry with angle=-a, this is just to demonstrate that it works. This is not possible in the string based initialization
    U += gates.Ry(target=1, control=0,  angle=b, frozen=False) # frozen=true: this variable will not be optimized
    U += gates.Rx(target=0, angle=1.234) # this gate will not be recognized as parametrized (it also has no effect on the energy in this example)

    # initialize the objective
    O = Objective(unitaries=U, observable=H)

    # call the optimizer
    angles = optimizer(objective=O, initial_values=None) # take the current values of the circuit as the initial ones, alternativ use initial_values={"a": 2.0, "b": 2.0}

    print("final angles are:\n", angles)

    # plot results
    optimizer.plot(plot_energies=True, plot_gradients=True) # plot only a specific gradient with plot_gradients=["a"]



