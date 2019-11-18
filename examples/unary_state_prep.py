"""
Example file on how to get simple circuits to construct
States which are similar to unary states or CIS states
i.e. number_of_qubits == number_of_contributing_basis_states

In this example we are constructing CIS excitation states for a 2-electron system in an 8 Qubit basis in Jordan-Wigner representation

Note: The UnaryStatePrep code is currently unstable and needs improvement
"""

from openvqe.apps import UnaryStatePrep
from openvqe.simulators.simulator_cirq import SimulatorCirq
from openvqe.qubit_wavefunction import QubitWaveFunction

if __name__ == "__main__":

    # initialize the simulators (can currently be pyquil, cirq or symbolic)
    simulator = SimulatorCirq()

    # this is the wavefunction we want to initialize with UnaryStatePrep:
    target_wfn = QubitWaveFunction.from_string("0.3|01100000> -0.3|10010000>+0.2|01001000> -0.2|10000100> + 0.1|01000010> -0.1|10000001> + 1.0|11000000>")
    target_wfn = target_wfn.normalize()

    print("target wavefunction: ", target_wfn)

    USP = UnaryStatePrep(target_space=target_wfn)
    print(USP._abstract_circuit)

    circuit = USP(wfn=target_wfn)

    result = simulator.simulate_wavefunction(abstract_circuit=circuit)

    wfn = result.wavefunction
    print("crated circuit:\n", result.circuit)
    print("prepared wavefunction: ", wfn)
    print("targeted wavefunction: ", target_wfn)
    print("difference           : ", target_wfn - wfn)



