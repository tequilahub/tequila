from openvqe.hamiltonian import HamiltonianQC
from openvqe.quantumchemistry.qc_base import ParametersQC
from openvqe.ansatz import AnsatzUCC
from openvqe.simulator.simulator_cirq import SimulatorCirq
from openvqe.objective import Objective
from openvqe.circuit.gradient import grad
from numpy import isclose
from openvqe.circuit.exponential_gate import DecompositionFirstOrderTrotter
from openvqe.ansatz import prepare_product_state

import pytest
import openfermion
from shutil import which

system_has_psi4 = which("psi4") is not None

@pytest.mark.skipif(condition=not system_has_psi4, reason="you don't have psi4")
@pytest.mark.parametrize("geom", [" H 0.0 0.0 1.0\n H 0.0 0.0 -1.0", " he 0.0 0.0 0.0"])
@pytest.mark.parametrize("basis", ["sto-3g"])
@pytest.mark.parametrize("trafo", ["JW", "BK"])
def test_hamiltonian(geom: str, basis: str, trafo: str):
    parameters_qc = ParametersQC(geometry=geom, basis_set=basis, transformation=trafo, outfile="asd")
    hqc = HamiltonianQC(parameters_qc)
    Hmol = hqc.make_fermionic_hamiltonian()
    H = hqc.hamiltonian
    if trafo == 'JW':
        assert(parameters_qc.jordan_wigner())
        assert(H == openfermion.jordan_wigner(Hmol))
    else:
        assert(trafo == "BK")
        assert(parameters_qc.bravyi_kitaev())
        assert(H == openfermion.bravyi_kitaev(Hmol))

@pytest.mark.skipif(condition=not system_has_psi4, reason="you don't have psi4")
def test_ucc():
        # check examples for comments
        parameters_qc = ParametersQC(geometry="data/h2.xyz", basis_set="sto-3g")
        parameters_qc.transformation = "JW"
        parameters_qc.psi4.run_ccsd = True
        parameters_qc.filename = "psi4"
        hqc = HamiltonianQC(parameters_qc)

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
        for dOi in dO.values():
            value = SimulatorCirq().simulate_objective(objective=dOi)
            gradient += value
        assert (isclose(gradient, 0.0, atol=1.e-4, rtol=1.e-4))

