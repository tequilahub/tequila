"""
In this example a compressed version of the CIS states of 2-electron in 8 Qubits systems is constructed
The 2-electron system can be H2, He or for example LiH with frozen core approximation

The example uses the Jordan-Wigner transformation to not overcomplicate things
Other mappings are also possible

Note that you can not print the Cirq object after simulation since there is currently a bug in cirq
which messes with the printout for multi-control gates (at least if they are initialized in the way tequila
does it

If you want pretty printout: Use Qiskit for printing out the circuit
"""

from tequila.apps import UnaryStatePrep
from tequila.simulators.simulator_cirq import SimulatorCirq
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila.circuit import gates


def unary_compressor(q: list):
    """
    Compresses unary states from 4 to 2 qubits (generalizable to N -> log(N))
    but manually here
    compressed qubits are: q[2] and q[3]
    :param q: the list of qubits
    :return: the encoder as circuit
    """
    assert(len(q)==4)
    result = gates.X(target=q[3])
    for i in range(0, 3):
        result += gates.X(target=q[3], control=q[i])

    result += gates.X(control=q[2], target=q[1])
    result += gates.X(control=q[2], target=q[0])
    result += gates.X(control=[q[1], q[0]], target=q[2])

    return result



def make_encoder(spin_sorted: bool = False):
    """
    Can be generalized for 2-electron systems, but not yet integrated in OpenVQE
    Therefore included here manually
    :return: Compression circuit from 8 to 3 qubits
    """
    qubits_up = [0, 2, 4, 6]
    qubits_down = [1, 3, 5, 7]
    result = unary_compressor(q=qubits_up)
    result += unary_compressor(q=qubits_down)
    return result


if __name__ == "__main__":
    # initialize the simulators (can currently be pyquil, cirq or symbolic)
    # the example uses full wavefunction simulation, so Quiskit won't work
    simulator = SimulatorCirq()
    #simulators = SimulatorSymbolic()

    # this is the wavefunction we want to initialize with UnaryStatePrep:
    target_wfn = QubitWaveFunction.from_string(
        "0.3|01100000> -0.3|10010000>+0.2|01001000> -0.2|10000100> + 0.1|01000010> -0.1|10000001> + 1.0|11000000>").normalize()

    # CIS prep circuit:
    USP = UnaryStatePrep(target_space=target_wfn)
    circuit = USP(wfn=target_wfn)

    # add the encoder
    circuit += make_encoder()

    wfn = simulator.simulate_wavefunction(abstract_circuit=circuit).wavefunction

    # this should be the compressed CIS wavefunction where the last 4 Qubits are all 0 and can be re-used for other tasks
    print("compressed wavefunction:", wfn)

