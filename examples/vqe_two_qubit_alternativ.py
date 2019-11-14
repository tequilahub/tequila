from openvqe.circuit import gates
from openvqe.circuit import Variable
from openvqe.hamiltonian import paulis
from openvqe.objective import Objective
from openvqe.optimizers.scipy_optimizers import OptimizerSciPy

"""
Same example as vqe_two_qubit.py with lazy variable initialization (less expressive)

The lazy initialization will set both variables to 0
In this example, this is stationary point leading to the optimizer being stuck
But we can change the initial values

Note how the 'a' parameter differs from the vqe_two_qubit.py example
since the signs are now part of the variable


"""

# uncomment if you want to use a specific simulator
# from openvqe.simulator.simulator_cirq import SimulatorCirq
# from openvqe.simulator.simulator_qiskit import SimulatorQiskit
# from openvqe.simulator.simulator_qulacs import SimulatorQulacs
# from openvqe.simulator.simulator_pyquil import SimulatorPyquil

# parameters with explanation:
samples = None      # number of samples for each run, None means full wavefunction simulation
simulator = None    # pick the simulator, None means it is automatically picked. Does not need to be initialized

# Sympy specific variables which you can set in 'minimize'
method = 'BFGS'
tol = 1.e-3
# see the minimize function signature for more

if __name__ == "__main__":
    # initialize Variables with initial values
    a = Variable(name="a", value=4.0)
    b = Variable(name="b", value=2.0)

    # initialize the Hamiltonian
    H = paulis.X(1)

    # initialize the parametrized Circuit
    U = gates.Ry(target=0, angle="a")
    U += gates.Ry(target=1, control=0,  angle="b", frozen=False) # frozen=true: this variable will not be optimized
    U += gates.Rx(target=0, angle=1.234) # this gate will not be recognized as parametrized (it also has no effect on the energy in this example)

    # initialize the objective
    O = Objective(unitaries=U, observable=H)

    # extract parameters from circuit
    angles = U.extract_parameters()

    # define initial values for the optimizer
    angles['a'] = -4.0
    angles['b'] = 2.0

    # Optimize
    optimizer = OptimizerSciPy(simulator=simulator, samples=samples, method=method)
    E, angles = optimizer(objective=O, initial_values=angles)

    # plot the history
    optimizer.history.plot()
    optimizer.history.plot(property='gradients', key='a') # if no key is given it will plot all of them
    optimizer.history.plot(property='gradients', key='b')
    optimizer.history.plot(property='angles', key='a')
    optimizer.history.plot(property='angles', key='b')
    # combine plots
    optimizer.history.plot(property='angles', key=['a','b'])
    optimizer.history.plot(property=['angles', 'energies'], key=['a', 'b'])

    # plot other results
    print("final angles are:\n", angles)
    print("final energy is :\n", E)
    print("iterations      :", optimizer.history.iterations)

    # some inntuitive ways to deal with the history
    history = optimizer.history
    all_energies = history.energies
    all_angles = history.angles
    # evolution of angle 'a'
    all_angles_a = history.extract_angles(key='a')






