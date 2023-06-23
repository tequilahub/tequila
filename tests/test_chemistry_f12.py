import pytest
import tequila as tq
import numpy
import os, glob

HAS_PSI4 = "psi4" in tq.quantumchemistry.INSTALLED_QCHEMISTRY_BACKENDS

@pytest.mark.skipif(condition=not HAS_PSI4, reason="you don't have psi4")
def test_correction_psi4_active():
    geomstring = "He 0.0 0.0 0.0"
    # This active space does not need to make sense
    mol = tq.Molecule(geometry=geomstring, basis_set='cc-pvqz', active_orbitals={'AG': [0,1], 'B1U': [0,1]})
    rdminfo = {'rdm__psi4_method': 'detci'}
    dE = mol.perturbative_f12_correction(cabs_type='active', **rdminfo)

    assert numpy.isclose(dE, -0.023784170991254745, atol=1.e-7)

cabs_ready_psi4 = False 
if HAS_PSI4:
    import psi4
    #!! The function "cheap_ri_space"  was added on top of psi4, but is unused at the moment. 
    #!! This means it might be removed again when a PR into psi4 is issued, 
    #!! but if successful the test does not need to be skipped anymore :)
    if hasattr(psi4.core.OrbitalSpace, "cheap_ri_space"): 
        cabs_ready_psi4 = True

@pytest.mark.skipif(condition=not cabs_ready_psi4, reason="You don't have the correct version of psi4 with CABS-hack.")
def test_correction_psi4_cabsplus():
    # Try to run CABS+, which as of now is only available via direct installation of personal fork
    # Choice of CABS also not sensible here
    geomstring = "He 0.0 0.0 0.0"
    mol = tq.Molecule(geometry=geomstring, basis_set='cc-pvdz', active_orbitals=[i for i in range(2)], point_group='c1')
    rdminfo = {'rdm__psi4_method': 'detci'}
    cabs_options = {'cabs_name': 'cc-pvqz'}
    dE = mol.perturbative_f12_correction(cabs_type='cabs+', cabs_options=cabs_options, **rdminfo)

    assert numpy.isclose(dE, -0.02285941739816253, atol=1.e-7)

data_path = 'data/f12/'
@pytest.mark.skipif(not(    os.path.isfile(str(data_path)+'he-f12_gtensor.npy')
                        and os.path.isfile(str(data_path)+'he-f12_htensor.npy')
                        and os.path.isfile(str(data_path)+'he-f12_f12tensor.npy')), reason="data not there")
@pytest.mark.parametrize("trafo", tq.quantumchemistry.encodings.known_encodings())
def test_correction_madness(trafo):
    geomstring = "He 0.0 0.0 0.0"
    mol = tq.Molecule(name='data/f12/he-f12',geometry=geomstring, basis_set='madness', n_pno="read",
                      active_orbitals=[0, 1], madness_root_dir=None, keep_mad_files=trafo)
    if not mol.supports_ucc():
        return
    U = mol.make_upccgsd_ansatz(name='UpCCD')
    angles = {(((0, 1),), 'D', (None, 0)): -0.1278860185100716} 
    rdminfo = {"U": U, "variables": angles}
    dE = mol.perturbative_f12_correction(f12_filename=str(data_path)+'he-f12_f12tensor.npy', **rdminfo)

    assert numpy.isclose(dE, -0.020945873534783065, atol=1.e-7)

