from openvqe.circuit import gates
from openvqe.circuit import Variable
from openvqe.hamiltonian import paulis
from openvqe.objective import Objective
from openvqe.optimizers.scipy_optimizers import OptimizerSciPy
import numpy

"""
A very simple example for a two qubit VQE optimized with scipy
This is the string based initialization which might be more convenient but initialization with Variables
Is way more flexible (see the original vqe_two_qubit.py)

Keynotes:
- Initialization of variables
- Rough demonstration of how the optimizers work
- Usage of optimizer History
- Usage of convenience plot function in history

Play around with stepsize, iterations and initial values
- The true minimum is at -1
- The true maximum is at +1
- there is a stationary point at a=0 and b=0 and others where the Energy is also 0.0
- For some methods bounds can be defined which allow you to avoid hitting this stationary point

"""

# uncomment if you want to use a specific simulator
# from openvqe.simulator.simulator_cirq import SimulatorCirq
# from openvqe.simulator.simulator_qiskit import SimulatorQiskit
# from openvqe.simulator.simulator_qulacs import SimulatorQulacs
# from openvqe.simulator.simulator_pyquil import SimulatorPyquil

# parameters with explanation:
samples = None  # number of samples for each run, None means full wavefunction simulation
simulator = None  # pick the simulator, None means it is automatically picked. Does not need to be initialized

# Sympy specific variables which you can set in 'minimize'
# check other optimizations at
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
# note that you can not use methods which use hessians
# You should get a meaningful scipy error if you chose them

# Gradient based methods
method = 'BFGS'
# method = 'L-BFGS-B'
# method = 'CG'
# method = 'TNC'

# Gradient Free Methods
# method = 'Powell'
# method = 'COBYLA'
# method = 'Nelder-Mead'

tol = 1.e-3
# see the minimize function signature for more

if __name__ == "__main__":
    # initialize Variables with initial values
    a = Variable(name="a", value=4.0)
    b = Variable(name="b", value=2.0)

    # initialize the Hamiltonian
    H = paulis.X(1)

    # initialize the parametrized Circuit
    U = gates.Ry(target=0, angle=-a / 2, frozen=False)  # frozen=true: this variable will not be optimized
    U += gates.Ry(target=0,
                  angle=-a / 2)  # will behave the same as only one time Ry with angle=-a, this is just to demonstrate that it works. This is not possible in the string based initialization
    U += gates.Ry(target=1, control=0, angle=b, frozen=False)  # frozen=true: this variable will not be optimized
    U += gates.Rx(target=0,
                  angle=1.234)  # this gate will not be recognized as parametrized (it also has no effect on the energy in this example)

    # initialize the objective
    O = Objective(unitaries=U, observable=H)

    # Extract parameters from circuit and set initial values
    # not necessary if already initialized above
    angles = U.extract_parameters()
    angles['a'] = 4.0
    angles['b'] = 2.0

    # some of the SciPy optimizers support bounds on the variables
    #bounds = {'a': (0.0, 4.0 * numpy.pi), 'b': (0.0, 4.0 * numpy.pi)}
    # avoid the stationary point with E=0.0
    bounds = {'a': (0.1, 1.9 * numpy.pi), 'b': (0.1, 1.9 * numpy.pi)}

    # Optimize
    optimizer = OptimizerSciPy(simulator=simulator, samples=samples, method=method, method_bounds=bounds)
    E, angles = optimizer(objective=O)

    # plot the history
    optimizer.history.plot()
    optimizer.history.plot(property='gradients', key='a')  # if no key is given it will plot all of them
    optimizer.history.plot(property='gradients', key='b')
    optimizer.history.plot(property='angles', key='a')
    optimizer.history.plot(property='angles', key='b')
    # combine plots
    optimizer.history.plot(property='angles', key=['a', 'b'])
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
