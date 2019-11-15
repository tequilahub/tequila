"""
Example of a simple one Qubit VQE optimized with simple GradientDescent

Try to use the OptimizerSciPy instead as an exercise (see vqe_two_qubit.py example)

"""

from openvqe.circuit import gates
from openvqe.hamiltonian import paulis
from openvqe.objective import Objective
from openvqe import numpy
from openvqe.circuit import Variable
from openvqe.optimizers import GradientDescent

# uncomment if you want to use a specific simulators
# from openvqe.simulators.simulator_cirq import SimulatorCirq
# from openvqe.simulators.simulator_qiskit import SimulatorQiskit
# from openvqe.simulators.simulator_qulacs import SimulatorQulacs
# from openvqe.simulators.simulator_pyquil import SimulatorPyquil

# some variables to play around with for this example
stepsize = 0.1
initial_angle = 0.0
maxiter = 100
optimal_angle = numpy.pi / 2
samples = None # none means infinite i.e. full wavefunction simulation
simulator = None # none means it is automatically picked, uncomment above if you want to try specific ones

if __name__ == "__main__":

    # initialize a Hamiltonian
    H = paulis.X(qubit=0)

    # initialize the initial angles
    angle = Variable(name="angle", value=initial_angle)

    # initialize an Ansatz for the Wavefunction
    U = gates.Ry(target=0, angle=angle)
    # alternativ without use of Variable (value is then set to 0.0):
    # U = gates.Ry(target=0, angle="angle")

    # initialize the objective
    O = Objective(unitaries=U, observable=H)

    # do the optimization
    optimizer = GradientDescent(stepsize=stepsize, maxiter=maxiter, samples=samples, simulator=simulator)
    E, angles = optimizer(objective=O, initial_values={"angle": initial_angle}) # don't need to set initial_values

    print("optimal energy = ", -1.0)
    print("optimal angle  = ", optimal_angle)
    print("found energy   = ", E)
    print("found angle    = ", angles)

    optimizer.history.plot('energies')
    optimizer.history.plot('gradients')


