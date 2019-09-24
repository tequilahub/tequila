from openvqe.hamiltonian import HamiltonianQC, ParametersQC
from openvqe.ansatz import AnsatzUCC
from openvqe.circuit.compiler import compile_trotter_evolution
from openvqe.simulator.simulator_cirq import SimulatorCirq
from openvqe.tools.expectation_value_cirq import expectation_value_cirq
from openvqe.objective import Objective
from openvqe.circuit.gradient import grad
from numpy import isclose

import unittest


class TestParameters(unittest.TestCase):

    def test_h2_energy_cirq(self):
        # check examples for comments
        parameters_qc = ParametersQC(geometry="data/h2.xyz", basis_set="sto-3g")
        parameters_qc.transformation = "JW"
        parameters_qc.psi4.run_ccsd = True
        parameters_qc.filename = "psi4"
        hqc = HamiltonianQC(parameters_qc)

        filename = parameters_qc.filename

        amplitudes = hqc.parse_ccsd_amplitudes()

        ucc = AnsatzUCC()
        ucc_operator = ucc(angles=amplitudes)

        abstract_circuit = compile_trotter_evolution(cluster_operator=ucc_operator, steps=1, anti_hermitian=True)

        simulator = SimulatorCirq()

        result = simulator.simulate_wavefunction(abstract_circuit=abstract_circuit, returntype=None,
                                                 initial_state=ucc.initial_state(hqc))

        assert (ucc.initial_state(hqc) == 12)

        energy = expectation_value_cirq(final_state=result.backend_result.final_state,
                                        hamiltonian=hqc(),
                                        n_qubits=hqc.n_qubits)

        assert (isclose(energy, -1.1368354639104123))

        O = Objective(observable=hqc, unitaries=abstract_circuit)
        energy2 = SimulatorCirq().expectation_value(objective=O, initial_state=ucc.initial_state(hqc))
        assert (isclose(energy, energy2))

        dO = grad(O)
        gradient = 0.0
        for dOi in dO:
            value = SimulatorCirq().expectation_value(objective=dOi, initial_state=ucc.initial_state(hqc))
            gradient += value
        assert (isclose(gradient, 0.0, atol=1.e-4, rtol=1.e-4))
