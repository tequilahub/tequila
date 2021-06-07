import pytest
import tequila as tq
import numpy
import os, glob

HAS_PSI4 = "psi4" in tq.quantumchemistry.INSTALLED_QCHEMISTRY_BACKENDS

## Get QC backends for parametrized testing
#import select_backends
#backends = select_backends.get()

@pytest.mark.skipif(condition=not HAS_PSI4, reason="you don't have psi4")
def test_psi4_active():
    geomstring = "He 0.0 0.0 0.0"
    # This active space does not need to make sense
    mol = tq.Molecule(geometry=geomstring, basis_set='cc-pvqz', active_orbitals={'AG': [0,1], 'B1U': [0,1]})
    rdminfo = {'rdm__psi4_method': 'detci'}
    dE = mol.perturbative_f12_correction(cabs_type='active', **rdminfo)
    
    assert numpy.isclose(dE, -0.023784170991254745, atol=1.e-7)


@pytest.mark.skipif(condition=not HAS_PSI4, reason="you don't have psi4")
def test_psi4_cabsplus():
    # Try to run CABS+, which as of now is only available via direct installation of personal fork
    # Choice of CABS also not sensible here
    geomstring = "He 0.0 0.0 0.0"
    mol = tq.Molecule(geometry=geomstring, basis_set='cc-pvdz', active_orbitals=[i for i in range(2)], point_group='c1')
    rdminfo = {'rdm__psi4_method': 'detci'}
    cabs_options = {'cabs_name': 'cc-pvqz'}
    try:
        dE = mol.perturbative_f12_correction(cabs_type='cabs+', cabs_options=cabs_options, **rdminfo)
        has_correct_psi4 = True 
    except:
        has_correct_psi4 = False 
        print("It seems you haven't installed the python version that goes with the CABS+ implementation.") 

    if has_correct_psi4:    
        assert numpy.isclose(dE, -0.02285941739816253, atol=1.e-7)

data_path = 'data/f12/'
@pytest.mark.skipif(not(os.path.isfile(data_path+'he-f12_gtensor.npy') and os.path.isfile(data_path+'he-f12_htensor.npy') and os.path.isfile(data_path+'he-f12_f12tensor.npy')), reason="data not there")
def test_madness():
    geomstring = "He 0.0 0.0 0.0"
    mol = tq.Molecule(name=(data_path+'he-f12'),geometry=geomstring, basis_set='madness', n_pno=39, active_orbitals=[0, 1], madness_root_dir=None)
    U = mol.make_upccgsd_ansatz(name='UpCCD')
    angles = {(0, ((0, 2), (1, 3)), None): 0.12788620002422374, (0, (0, 2), None): -2.5466955700597297e-05, (0, (1, 3), None): -2.5466955700597297e-05}
    rdminfo = {"U": U, "variables": angles}
    dE = mol.perturbative_f12_correction(f12_filename='he_f12tensor.npy', **rdminfo)

    assert numpy.isclose(dE, -0.020945464502545082, atol=1.e-7)
