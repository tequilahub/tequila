from openvqe.circuit import gates
from openvqe.circuit import Variable
from openvqe.hamiltonian import paulis
from openvqe.objective import Objective
from openvqe.optimizers import minimize

"""
A simple example for a two qubit optimization using scipy.optimize
"""

# uncomment if you want to use a specific simulator
# from openvqe.simulator.simulator_cirq import SimulatorCirq
# from openvqe.simulator.simulator_qiskit import SimulatorQiskit
# from openvqe.simulator.simulator_qulacs import SimulatorQulacs
# from openvqe.simulator.simulator_pyquil import SimulatorPyquil

# parameters with explanation:
samples = None      # number of samples for each run, None means full wavefunction simulation
simulator = None    # pick the simulator, None means it is automatically picked. Does not need to be initialized

if __name__ == "__main__":
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

    # Optimize
    E, angles, res = minimize(O, return_all=True)

    print("final angles are:\n", angles)
    print("final energy is:\n", E)
    print("total number of evaluations of O:\n", res.nfev)





