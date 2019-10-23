"""
Example file on how to get simple circuits to construct
States which are similar to unary states or CIS states
i.e. number_of_qubits == number_of_contributing_basis_states

In this example we are constructing CIS excitation states for the H2 molecule

Note: The Code is currently unstable and needs improvement
"""

from openvqe.apps import UnaryStatePrep
from openvqe.simulator.simulator_cirq import SimulatorCirq

if __name__ == "__main__":

    # initialize the simulator (can currently be pyquil, cirq or symbolic)
    simulator = SimulatorCirq()

    singles_space =[
    '01100000',
    '01001000',
    '01000010',
    '10010000',
    '10000100',
    '10000001',
    '11000000'
    ]

    # the value makes no sense
    # note that they do not need to be normalized (happens in the object)
    singles_coefficients = [
        1.0,
        2.0,
        4.0,
        1.0,
        2.0,
        4.0,
        8.0,
    ]

    USP = UnaryStatePrep(target_space=singles_space)

    circuit = USP(coeffs=singles_coefficients)

    result = simulator.simulate_wavefunction(abstract_circuit=circuit)

    wfn = result.wavefunction
    print("crated circuit:\n", result.circuit)
    print("prepared wavefunction: ", wfn)


