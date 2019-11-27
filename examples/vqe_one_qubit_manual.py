"""
Example of a simple one Qubit VQE optimized with simple GradientDescent

This does the same as vqe_one_qubit.py just performs the steps manual

"""

from tequila.circuit import gates
from tequila.hamiltonian import paulis
from tequila.objective import Objective
from tequila.circuit.gradient import grad
import numpy
from tequila.circuit import Variable
from tequila.optimizers import GradientDescent #Doesn't do much, but is convenient for the plots in this example

# uncomment if you want to try a different simulators (not all combinations of samples and simulators are possible)
from tequila.simulators.simulator_cirq import SimulatorCirq
# from tequila.simulators.simulator_qiskit import SimulatorQiskit
# from tequila.simulators.simulator_qulacs import SimulatorQulacs
# from tequila.simulators.simulator_pyquil import SimulatorPyquil

# some variables to play around with for this example
stepsize = 0.1
initial_angle = 0.0
max_iter = 100
optimal_angle = numpy.pi / 2
samples = 1000  # number of samples/measurements for the simulators, will have no effect if full wavefunction simulation is used
use_full_wavefunction_simulation = True # if true: you can use cirq, pyquil and qulacs. If false you can use qiskit and cirq

if __name__ == "__main__":

    # initialize a Hamiltonian
    H = paulis.X(qubit=0)
    # initialize the initial angles
    angle = Variable(name="angle", value=initial_angle)

    simulator = SimulatorCirq()
    if use_full_wavefunction_simulation:
        simulator = SimulatorCirq()

    # initialize the optimizer
    optimizer = GradientDescent(stepsize=stepsize)

    # initialize an Ansatz for the Wavefunction
    U = gates.Ry(target=0, angle=angle)

    # do the optimization
    angles = U.extract_variables()
    E = 100.0
    for iter in range(max_iter):
        print(angles)
        # initialize an objective and compute Energy and gradient
        O = Objective(unitaries=U, observable=H)

        # compute the energy
        if use_full_wavefunction_simulation:
            Enew = simulator.simulate_objective(objective=O)
        else:
            Enew = simulator.measure_objective(objective=O, samples=samples)


        diff = Enew - E
        E = Enew
        print("E   =", E)
        print("diff=", diff)

        # compute the gradient, more sensitive to number of samples than energy
        dO = grad(O)
        dE = dict()
        for k in dO.keys():
            if use_full_wavefunction_simulation:
                dE[k] = simulator.simulate_objective(objective=dO[k])
            else:
                dE[k] = simulator.measure_objective(objective=dO[k], samples=samples)
        print("dE  =", dE)


        # update the angle
        angles['angle'] = angles['angle'] - stepsize*dE['angle']
        U.update_variables(angles)

    print("angle after ", max_iter, "iterations : ", angles)
    print("optimal angle : ", optimal_angle)



