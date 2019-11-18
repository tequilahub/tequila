from openvqe.circuit import gates
from openvqe.simulators.simulator_qiskit import SimulatorQiskit # full wavefunction simulation currently not supported
from openvqe.simulators.simulator_cirq import SimulatorCirq
from openvqe.simulators.simulator_pyquil import SimulatorPyquil # currently only full wavefunction simulation
from openvqe.tools import plotters # convenience
from openvqe import numpy

"""
Basic examples on how to simulate a GHZ State as full wavefunction or with individual measurements
"""

if __name__ == "__main__":

    # Create the circuit
    # change the angle to change the resulting wavenfunction/count distribution
    U = gates.Ry(target=0, angle=numpy.pi/2) # alternatively use gates.H(target=0)
    U+=gates.X(target=1, control=0)
    U+= gates.X(target=2, control=0)

    # Simulate the full wavefunction (use cirq or pyquil)
    simulator = SimulatorCirq()
    result = simulator.simulate_wavefunction(abstract_circuit=U)

    # the simulators gives back a lot of information, we only need the computed wavefunction
    wfn = result.wavefunction
    print("|GHZ>=", wfn)

    # the result stores for example the circuit object created by the backend
    backend_circuit = result.circuit
    our_original_circuit = result.abstract_circuit
    print("this was created by the backend:\n", backend_circuit)

    # count based simulation
    # changing simulators (not necessary)
    simulator = SimulatorQiskit()

    # we need to add measurement instructions for that
    U += gates.Measurement(target=[0,1,2])

    result = simulator.run(abstract_circuit=U, samples=1000)

    counts = result.counts

    # counts uses currently the same format as the wavefunction, therefore the printout
    print("counts:\n", counts)

    # give a filename to plot to a file
    plotters.plot_counts(counts=counts, filename=None)
