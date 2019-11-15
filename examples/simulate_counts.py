"""
Small example on how to construct a circuit and simulate its measurements in the computational standardbasis (counts)
"""

from openvqe.circuit import gates
from numpy import pi
from openvqe.simulators import pick_simulator
from openvqe.tools import plotters
"""
Same example as 'simulate_wavefunctions' but here we simulate counts

If you use Pyquil, make sure the QVM runs in the background
 -> open terminal, type 'qvm -S'
And make sure the rigetti qvm is actually installed 
"""

if __name__ == "__main__":

    """
    Here comes the actual example
    """

    # create a simple abstract circuit
    ac = gates.X(0)+gates.Ry(target=1, control=0, angle=pi / 2)

    # we need to add the measurements, here we measure both qubits
    ac += gates.Measurement(target=[0,1,2,3], name="a") #naming is possible but is not supported by every backend

    # choose a backend
    # 'symbolic' can only simulate wavefunctions, counts are not supported here. Could be implemented quickly
    # 'pyquil' has currently no support for individual measurements ... on the todo list
    Simulator = pick_simulator(samples=10000, demand_full_wfn=False)
    simulator = Simulator()
    simulator_results = simulator.run(abstract_circuit=ac, samples=10000)
    counts = simulator_results.counts

    print("circuit:\n", simulator_results.circuit, "\n")
    print("counts:\n", counts)

    plotters.plot_counts(counts, filename=None, label_with_integers=False)


