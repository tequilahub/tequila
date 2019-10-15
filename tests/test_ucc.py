from openvqe.hamiltonian import HamiltonianPsi4, ParametersQC
from openvqe.ansatz import AnsatzUCC
from openvqe.simulator.simulator_cirq import SimulatorCirq
from openvqe.tools.expectation_value_cirq import expectation_value_cirq
from openvqe.objective import Objective
from openvqe.circuit.gradient import grad
from numpy import isclose
from openvqe.circuit.exponential_gate import DecompositionFirstOrderTrotter
from openvqe.ansatz import prepare_product_state

import unittest


class TestParameters(unittest.TestCase):

    def test_h2_energy_cirq(self):
        # check examples for comments
        parameters_qc = ParametersQC(geometry="data/h2.xyz", basis_set="sto-3g")
        parameters_qc.transformation = "JW"
        parameters_qc.psi4.run_ccsd = True
        parameters_qc.filename = "psi4"
        hqc = HamiltonianPsi4(parameters_qc)

        filename = parameters_qc.filename

        amplitudes = hqc.parse_ccsd_amplitudes()

        ucc = AnsatzUCC(decomposition=DecompositionFirstOrderTrotter(steps=1, threshold=0.0))
        abstract_circuit = ucc(angles=amplitudes)

        simulator = SimulatorCirq()

        result = simulator.simulate_wavefunction(abstract_circuit=abstract_circuit, returntype=None,
                                                 initial_state=hqc.reference_state())

        assert (hqc.reference_state() == 12)
        prep_ref = prepare_product_state(state=hqc.reference_state())
        abstract_circuit = prep_ref + abstract_circuit

        O = Objective(observable=hqc, unitaries=abstract_circuit)
        energy = SimulatorCirq().simulate_objective(objective=O)
        assert (isclose(energy, -1.1368354639104123))

        dO = grad(O)
        gradient = 0.0
        for dOi in dO:
            value = SimulatorCirq().simulate_objective(objective=dOi)
            gradient += value
        assert (isclose(gradient, 0.0, atol=1.e-4, rtol=1.e-4))
