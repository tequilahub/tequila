from tequila.circuit import gates
from tequila.circuit import Variable
from tequila.hamiltonian import paulis
from tequila.objective import Objective
from tequila.optimizers.optimizer_scipy import minimize
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

# uncomment if you want to use a specific simulators
# from tequila.simulators.simulator_cirq import SimulatorCirq
# from tequila.simulators.simulator_qiskit import SimulatorQiskit
# from tequila.simulators.simulator_qulacs import SimulatorQulacs
# from tequila.simulators.simulator_pyquil import SimulatorPyquil

# parameters with explanation:
samples = None  # number of samples for each run, None means full wavefunction simulation
simulator = None  # pick the simulators, None means it is automatically picked. Does not need to be initialized

# Sympy specific variables which you can set in 'minimize'
# check other optimizations at
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
# note that you can not use methods which use hessians
# You should get a meaningful scipy error if you chose them

# Gradient based methods

method = 'L-BFGS-B'
#method = 'BFGS'
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
    H = paulis.X(1) + 0.001*paulis.Z(1)

    # initialize the parametrized Circuit
    U = gates.Ry(target=0, angle=-a / 2, frozen=False)  # frozen=true: this variable will not be optimized
    U += gates.Ry(target=0, angle=-a / 2)  # will behave the same as only one time Ry with angle=-a, this is just to demonstrate that it works. This is not possible in the string based initialization
    U += gates.Ry(target=1, control=0, angle=b, frozen=False)  # frozen=true: this variable will not be optimized
    U += gates.Rx(target=0,angle=1.234)  # this gate will not be recognized as parametrized (it also has no effect on the energy in this example)

    # initialize the objective
    O = Objective.ExpectationValue(U=U, H=H)

    # Extract parameters from circuit and set initial values
    # not necessary if already initialized above
    angles = U.extract_variables()
    angles['a'] = 4.0
    angles['b'] = 2.0

    # some of the SciPy optimizers support bounds on the variables
    #bounds = {'a': (0.0, 4.0 * numpy.pi), 'b': (0.0, 4.0 * numpy.pi)}
    # avoid the stationary point with E=0.0
    bounds = {'a': (0.1, 1.9 * numpy.pi), 'b': (0.1, 1.9 * numpy.pi)}

    # Optimize
    result = minimize(objective=O, simulator=simulator, samples=samples, method=method, method_bounds=bounds)

    # plot the history, default are energies
    result.history.plot()
    result.history.plot(property='gradients', key='a')  # if no key is given it will plot all of them
    result.history.plot(property='gradients', key='b')
    result.history.plot(property='angles', key='a')
    result.history.plot(property='angles', key='b')
    # combine plots
    result.history.plot(property='angles', key=['a', 'b'])
    result.history.plot(property=['angles', 'energies'], key=['a', 'b'])

    # plot other results
    print("final angles are:\n", result.angles)
    print("final energy is :\n", result.energy)
    print("iterations      :", result.history.iterations)

    # some inntuitive ways to deal with the history
    all_energies = result.history.energies
    all_angles = result.history.angles
    # evolution of angle 'a'
    all_angles_a = result.history.extract_angles(key='a')
