"""
Will replace test_psi4.py

Todo: write real test class with tear_down procedure to get rid of psi4 output files
"""

import pytest
import openvqe.quantumchemistry as qc
from openvqe import numpy
from openvqe.simulator import pick_simulator
from openvqe.objective import Objective


@pytest.mark.skipif(condition=not (qc.has_pyscf and qc.has_psi4),
                    reason="you don't have a quantum chemistry backend installed")
@pytest.mark.parametrize("geom", [" H 0.0 0.0 1.0\n H 0.0 0.0 -1.0", " he 0.0 0.0 0.0", " be 0.0 0.0 0.0"])
@pytest.mark.parametrize("basis", ["sto-3g"])
@pytest.mark.parametrize("trafo", ["JW", "BK"])
def test_hamiltonian_consistency(geom: str, basis: str, trafo: str):
    parameters_qc = qc.ParametersQC(geometry=geom, basis_set=basis, outfile="asd")
    hqc1 = qc.QuantumChemistryPsi4(parameters=parameters_qc).make_hamiltonian(transformation=trafo)
    hqc2 = qc.QuantumChemistryPySCF(parameters=parameters_qc).make_hamiltonian(transformation=trafo)
    assert (hqc1.hamiltonian == hqc2.hamiltonian)


@pytest.mark.skipif(condition=not qc.has_psi4, reason="you don't have psi4")
def test_h2_hamiltonian_psi4():
    do_test_h2_hamiltonian(qc_interface=qc.QuantumChemistryPsi4)


@pytest.mark.skipif(condition=not qc.has_pyscf, reason="you don't have pyscf")
def test_h2_hamiltonian_pysf():
    do_test_h2_hamiltonian(qc_interface=qc.QuantumChemistryPySCF)


def do_test_h2_hamiltonian(qc_interface):
    parameters = qc.ParametersQC(geometry="data/h2.xyz", basis_set="sto-3g")
    H = qc_interface(parameters=parameters).make_hamiltonian().tomatrix()
    vals = numpy.linalg.eigvalsh(H)
    assert (numpy.isclose(vals[0], -1.1368354639104123, atol=1.e-4))
    assert (numpy.isclose(vals[1], -0.52718972, atol=1.e-4))
    assert (numpy.isclose(vals[2], -0.52718972, atol=1.e-4))
    assert (numpy.isclose(vals[-1], 0.9871391, atol=1.e-4))


@pytest.mark.skipif(condition=not qc.has_psi4, reason="you don't have psi4")
@pytest.mark.parametrize("trafo", ["JW", "BK", "BKT"])
def test_ucc_psi4(trafo):
    parameters_qc = qc.ParametersQC(geometry="data/h2.xyz", basis_set="sto-3g")
    do_test_ucc(qc_interface=qc.QuantumChemistryPsi4, parameters=parameters_qc, result=-1.1368354639104123, trafo=trafo)


@pytest.mark.skipif(condition=not qc.has_pyscf, reason="you don't have pyscf")
@pytest.mark.parametrize("trafo", ["JW", "BK"])
def test_ucc_pyscf(trafo):
    parameters_qc = qc.ParametersQC(geometry="data/h2.xyz", basis_set="sto-3g")
    do_test_ucc(qc_interface=qc.QuantumChemistryPySCF, parameters=parameters_qc, result=-1.1368354639104123, trafo=trafo)


def do_test_ucc(qc_interface, parameters, result, trafo):
    # check examples for comments
    psi4_interface = qc_interface(parameters=parameters, transformation=trafo)

    hqc = psi4_interface.make_hamiltonian()

    # called twice on purpose (see if reloading works)
    amplitudes = psi4_interface.compute_ccsd_amplitudes()
    amplitudes = psi4_interface.compute_ccsd_amplitudes()

    U = psi4_interface.make_uccsd_ansatz(trotter_steps=1, initial_amplitudes="ccsd",
                                         include_reference_ansatz=True)
    H = psi4_interface.make_hamiltonian()
    O = Objective(observable=H, unitaries=U)
    Simulator = pick_simulator(samples=None)
    energy = Simulator().simulate_objective(objective=O)
    assert (numpy.isclose(energy, result))


@pytest.mark.skipif(condition=not qc.has_psi4, reason="you don't have psi4")
def test_mp2_psi4():
    # the number might be wrong ... its definetely not what psi4 produces
    # however, no reason to expect projected MP2 is the same as UCC with MP2 amplitudes
    parameters_qc = qc.ParametersQC(geometry="data/h2.xyz", basis_set="sto-3g")
    do_test_mp2(qc_interface=qc.QuantumChemistryPsi4, parameters=parameters_qc, result=-1.1279946983462537)

@pytest.mark.skipif(condition=not qc.has_pyscf, reason="you don't have pyscf")
def test_mp2_pyscf():
    # the number might be wrong ... its definetely not what psi4 produces
    # however, no reason to expect projected MP2 is the same as UCC with MP2 amplitudes
    parameters_qc = qc.ParametersQC(geometry="data/h2.xyz", basis_set="sto-3g")
    do_test_mp2(qc_interface=qc.QuantumChemistryPySCF, parameters=parameters_qc, result=-1.1279946983462537)


def do_test_mp2(qc_interface, parameters, result):
    # check examples for comments
    psi4_interface = qc_interface(parameters=parameters)
    hqc = psi4_interface.make_hamiltonian()

    # called twice on purpose (see if reloading works)
    amplitudes = psi4_interface.compute_ccsd_amplitudes()
    amplitudes = psi4_interface.compute_ccsd_amplitudes()

    U = psi4_interface.make_uccsd_ansatz(trotter_steps=1, initial_amplitudes="mp2",
                                         include_reference_ansatz=True)
    H = psi4_interface.make_hamiltonian()
    O = Objective(observable=H, unitaries=U)
    Simulator = pick_simulator(samples=None)

    energy = Simulator().simulate_objective(objective=O)
    assert (numpy.isclose(energy, result))
