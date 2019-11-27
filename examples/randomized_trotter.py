from tequila.simulators.simulator_qiskit import SimulatorQiskit
from tequila.circuit import gates
from tequila.hamiltonian import paulis
from tequila.tools import plotters
import numpy
from tequila import BitString

"""
In this example we implement a unitary which acts only on the two-qubit basis states
|00>, |01>, |10> 
in the following way (leaving out normalization):
    U|00> = |01> + |10>
We do that by creating an Hamiltonian of the form: H = |psi><phi| + |phi><psi|
and exponentiate it. The mapping to PauliStrings is in binary (one could use other mappings)
I chosse binary since its simple to follow

The exponential of the Hamiltonian is Trotterized
Play with the parameters to witness different results:
    randomize : the order of the paulistrings in the trotter decomposition is randomized
    samples
    steps: the number of trotter steps

The true result is an equal distribution between 01 and 10

Try for yourself: make multiple runs with samples=1 and randomize=True and accumulate the counts (make shure to reconstruct the circuit everytime in to create new random instances)

See randomized_trotter_parallelized for a similar example which uses multiprocessing to efficiently simulate the individual runs

"""

if __name__ == "__main__":

        steps=5
        samples=100000
        trotter_parameters = gates.TrotterParameters(randomize=True)

        a = BitString.from_binary(binary="01") # you can also just type a = 1
        b = BitString.from_binary(binary="10") # you can also just type b = 2
        c = BitString.from_binary(binary="00") # you can also just type c = 0
        fac = numpy.pi/numpy.sqrt(2)
        H = fac*paulis.decompose_transfer_operator(ket=a.integer, bra=c.integer, qubits=[0,1])
        H+= fac*paulis.decompose_transfer_operator(ket=b.integer, bra=c.integer, qubits=[0,1])
        H+= fac*paulis.decompose_transfer_operator(ket=c.integer, bra=a.integer, qubits=[0,1])
        H+= fac*paulis.decompose_transfer_operator(ket=c.integer, bra=b.integer, qubits=[0,1])
        print(H)
        # this is the Hamiltonian we created here
        #+1.1107X(1)+1.1107Z(0)X(1)+1.1107X(0)+1.1107X(0)Z(1)

        U = gates.Trotterized(generators=[H], steps=steps, parameters=trotter_parameters)
        U += gates.Measurement(target=[0,1])
        result = SimulatorQiskit().run(abstract_circuit=U, samples=samples)

        plotters.plot_counts(counts=result.counts, title="random="+str(trotter_parameters.randomize)+", steps="+str(steps)+", samples="+str(samples))
