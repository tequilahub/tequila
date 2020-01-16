from shutil import which

SUPPORTED_QCHEMISTRY_BACKENDS = ["psi4", "pyscf"]
INSTALLED_QCHEMISTRY_BACKENDS = {}

from .qc_base import ParametersQC, Amplitudes

has_psi4 = which("psi4") is not None
if has_psi4:
    from .psi4_interface import QuantumChemistryPsi4
    INSTALLED_QCHEMISTRY_BACKENDS["psi4"] = QuantumChemistryPsi4

# import importlib
has_pyscf = False
# try:
#     importlib.util.find_spec('pyscf')
#     has_pyscf = True
# except ImportError:
#     has_pyscf = False
#
# if has_pyscf:
#     from .pyscf_interface import QuantumChemistryPySCF
#     INSTALLED_QCHEMISTRY_BACKENDS["pyscf"] = QuantumChemistryPySCF

def show_available_modules():
    print ("Available QuantumChemistry Modules:")
    for k in INSTALLED_QCHEMISTRY_BACKENDS.keys():
        print(k)

def show_supported_modules():
    print(SUPPORTED_QCHEMISTRY_BACKENDS)

def pick_backend(backend: str):

    if backend not in INSTALLED_QCHEMISTRY_BACKENDS:
        raise Exception(str(backend) + " was not found on your system")

    if backend not in SUPPORTED_QCHEMISTRY_BACKENDS:
        raise Exception(str(backend) + " is not (yet) supported by tequila")

    return INSTALLED_QCHEMISTRY_BACKENDS[backend]