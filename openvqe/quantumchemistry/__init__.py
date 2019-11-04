import importlib
has_pyscf = False
try:
    importlib.util.find_spec('pyscf')
    has_pyscf = True
except ImportError:
    has_pyscf = False

from shutil import which
has_psi4 = which("psi4") is not None

if has_psi4:
    from openvqe.quantumchemistry.psi4_interface import QuantumChemistryPsi4


