

from shutil import which
has_psi4 = which("psi4") is not None

from .qc_base import ParametersQC, Amplitudes

if has_psi4:
    from .psi4_interface import QuantumChemistryPsi4

# import importlib
# has_pyscf = False
# try:
#     importlib.util.find_spec('pyscf')
#     has_pyscf = True
# except ImportError:
#     has_pyscf = False
#
# if has_pyscf:
#     from .pyscf_interface import QuantumChemistryPySCF

def show_available_modules() -> str:
    return "Available QuantumChemistry Modules:\n" + \
           "psi4 = " + str(has_psi4) + "\n" + \
           "pyscf = " + str(has_pyscf) + "\n"


