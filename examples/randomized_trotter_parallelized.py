"""
Some illustrations on randomized Trotter Decompositions
And how to combine the OpenVQE primitives with multiprocessing
The example is meant to be executed on clusters or workstations

For a more simple introduction see the randomized_trotter.py example
This an easier example without multiprocessing


Here we simulate the process

|0101> --> 1.0/sqrt(2)*(|1000> + |0010>)

through the unitary operator

U = exp(i(c_0*a_1 - c_1*a_0))
where c and a are creation/annihilation operator acting on modes
encoded into 2-qubits each

This means:
|0101> : mode a is occupied by one quanta as well as mode b -> |1>_a|1>_b
|1000> : mode a is occupied by two quanta, mode b is unoccupied
|0001> : mode b is occupied by two quanta, mode a is unoccupied

Using randomization the balance between the two states is significantly improved

Since for the randomization we need to take single samples each time, as we would on real hardware,
this example is especially slow. For that reason the evaluation of the circuits are parallelized here.


"""

from openvqe.simulators.simulator_qiskit import SimulatorQiskit, QubitWaveFunction
from openvqe.circuit import gates
from openvqe.hamiltonian import QubitHamiltonian, paulis
from openvqe.tools import plotters
import typing
import numpy
from openvqe import dataclass

import multiprocessing as mp


# need this definition for this example
def anihilation(qubits: typing.List[int]) -> QubitHamiltonian:
    max_occ = 2 ** len(qubits) - 1
    result = QubitHamiltonian.init_zero()
    for occ in range(max_occ):
        c = numpy.sqrt(occ + 1)
        result += c * paulis.decompose_transfer_operator(ket=occ + 1, bra=occ, qubits=qubits)
    return result


# need this definition for this example
def creation(qubits: typing.List[int]) -> QubitHamiltonian:
    max_occ = 2 ** len(qubits) - 1
    result = QubitHamiltonian.init_zero()
    for occ in range(max_occ):
        c = numpy.sqrt(occ + 1)
        result += c * paulis.decompose_transfer_operator(ket=occ, bra=occ + 1, qubits=qubits)
    return result


@dataclass
class DoIt:
    """
    This does the actual computation, it's wrapped into this DoIt class
    To simplify parallelization
    """

    steps: int
    trotter_parameters: gates.TrotterParameters

    def __call__(self, *args, **kwargs) -> QubitWaveFunction:
        U = gates.X(1) + gates.X(3)
        U += gates.Trotterized(steps=self.steps, generators=generators, parameters=self.trotter_parameters)
        U += gates.Measurement(target=qubits_a + qubits_b)
        result = SimulatorQiskit().run(abstract_circuit=U, samples=1)
        return result.counts


if __name__ == "__main__":

    # set up parallel environment
    samples = 100
    nproc = 4  # set this to None to use all available processes (better not do it on a laptop)
    result_file = "randomized_trotter_results.pdf"  # if set to None, the results are displayed directly. Won't work on a cluster

    print("CPU Count is: ", mp.cpu_count())
    if nproc is None:
        nproc = mp.cpu_count()
    print("will use ", nproc, " processes")
    nproc = min(nproc, mp.cpu_count())
    pool = mp.Pool(processes=nproc)

    # basc definitions
    qubits_a = [0, 1]
    qubits_b = [2, 3]
    H_0 = 1.0j * numpy.pi / 2 * (
            creation(qubits=qubits_a) * anihilation(qubits=qubits_b) - anihilation(qubits=qubits_a) * creation(
        qubits=qubits_b))
    H_1 = -1.0j * numpy.pi / 2 * (
            creation(qubits=qubits_b) * anihilation(qubits=qubits_a) - anihilation(qubits=qubits_b) * creation(
        qubits=qubits_a))
    print("H_0       =", H_0)  # biases 0010
    print("H_1       =", H_1)  # biases 1000
    print("commutator=", H_0 * H_1 - H_1 * H_0)

    # different ways to initialize which generate formally the same operator
    # pick one by commenting out the rest
    generators = [H_0]  # strong bias on 0010
    # generators = [H_1]  # strong bias on 1000
    # generators = [0.5 * H_0, 0.5 * H_1]  # bias on 1000
    # generators = [0.5 * H_1, 0.5 * H_0]  # bias on 0010

    # Parameters
    parameters = {
        'steps': 3,
        'trotter_parameters' : gates.TrotterParameters(randomize=True, randomize_component_order=True, join_components=False)
    }

    # run
    all_counts = pool.map(func=DoIt(**parameters), iterable=range(0, samples))

    # accumulate results
    counts = QubitWaveFunction()
    for c in all_counts:
        counts += c

    # plot results
    # if no filename is given the results are displayed
    # on a cluster: give a filename!
    plotters.plot_counts(counts=counts, label_with_integers=False, filename=result_file)
