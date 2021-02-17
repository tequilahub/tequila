import pytest
import numpy
import os
import tequila as tq

root=os.environ.get("MAD_ROOT_DIR")
executable = tq.quantumchemistry.madness_interface.QuantumChemistryMadness.find_executable(root)
print("root = ", root)
print("executable = ", executable)

def test_executable():
    if root is not None and executable is None:
        raise Exception("Found no pno_integrals executable but found MAD_ROOT_DIR={}\n"
                        "Seems like you wanted that tested".format(root))

@pytest.mark.skipif(not(os.path.isfile('he_gtensor.npy') and os.path.isfile('be_htensor.npy')), reason="data not there")
def test_madness_he_data():
    # relies that he_xtensor.npy are present (x=g,h)
    geomstring="He 0.0 0.0 0.0"
    molecule = tq.Molecule(name="he", geometry=geomstring)
    H = molecule.make_hamiltonian()
    UHF = molecule.prepare_reference()
    EHF = tq.simulate(tq.ExpectationValue(H=H, U=UHF))
    assert(numpy.isclose(-2.861522e+00, EHF, atol=1.e-5))
    U = molecule.make_upccgsd_ansatz()
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(method="bfgs", objective=E, initial_values=0.0, silent=True)
    print(result.energy)
    assert (numpy.isclose(-2.87761809, result.energy, atol=1.e-5))

@pytest.mark.skipif(executable is None, reason="madness was not found")
def test_madness_full_he():
    # relies on madness being compiled and MAD_ROOT_DIR exported
    # or pno_integrals in the path
    geomstring="He 0.0 0.0 0.0"
    molecule = tq.Molecule(geometry=geomstring, n_pno=1)
    H = molecule.make_hamiltonian()
    UHF = molecule.prepare_reference()
    EHF = tq.simulate(tq.ExpectationValue(H=H, U=UHF))
    assert(numpy.isclose(-2.861522e+00, EHF, atol=1.e-5))
    U = molecule.make_upccgsd_ansatz()
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(method="bfgs", objective=E, initial_values=0.0, silent=True)
    assert (numpy.isclose(-2.87761809, result.energy, atol=1.e-5))

@pytest.mark.skipif(executable is None, reason="madness was not found")
def test_madness_full_li_plus():
    # relies on madness being compiled and MAD_ROOT_DIR exported
    # or pno_integrals in the path
    geomstring="Li 0.0 0.0 0.0"
    molecule = tq.Molecule(name="li+", geometry=geomstring, n_pno=1, charge=1)
    H = molecule.make_hamiltonian()
    UHF = molecule.prepare_reference()
    EHF = tq.simulate(tq.ExpectationValue(H=H, U=UHF))
    assert(numpy.isclose(-7.236247e+00, EHF, atol=1.e-5))
    U = molecule.make_upccgsd_ansatz()
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(method="bfgs", objective=E, initial_values=0.0, silent=True)
    assert (numpy.isclose(-7.251177798, result.energy, atol=1.e-5))

@pytest.mark.skipif(executable is None, reason="madness was not found")
def test_madness_full_be():
    # relies on madness being compiled and MAD_ROOT_DIR exported
    # or pno_integrals in the path
    geomstring="Be 0.0 0.0 0.0"
    molecule = tq.Molecule(name="be", geometry=geomstring, n_pno=3, frozen_core=True)
    H = molecule.make_hamiltonian()
    UHF = molecule.prepare_reference()
    EHF = tq.simulate(tq.ExpectationValue(H=H, U=UHF))
    assert(numpy.isclose(-14.57269300, EHF, atol=1.e-5))
    U = molecule.make_upccgsd_ansatz()
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(method="bfgs", objective=E, initial_values=0.0, silent=True)
    assert (numpy.isclose(-14.614662051580321, result.energy, atol=1.e-5))
