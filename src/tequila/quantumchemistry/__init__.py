from shutil import which
import typing

SUPPORTED_QCHEMISTRY_BACKENDS = ["psi4", "pyscf"]
INSTALLED_QCHEMISTRY_BACKENDS = {}

from .qc_base import ParametersQC

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

def Molecule(geometry: str, basis_set: str, transformation: typing.Union[str, typing.Callable] = None, backend: str = None, *args, **kwargs):
    """
    :param geometry: filename to an xyz file
    :param basis_set: basis set in standard notation
    :param transformation: Jordan-Wigner, Bravyi-Kitaev, and whatever OpenFermion supports
    :param backend: psi4 or pyscf
    :param kwargs: further parameters defined in ParametersQC
    :return:
    """

    keyvals = {}
    for k,v in kwargs.items():
        if k in ParametersQC.__dict__.keys():
            keyvals[k] = v

    parameters = ParametersQC(geometry=geometry, basis_set=basis_set, multiplicity=1, **keyvals)

    if backend is None:
        if "psi4" in INSTALLED_QCHEMISTRY_BACKENDS:
            backend = "psi4"
        elif "pyscf" in INSTALLED_QCHEMISTRY_BACKENDS:
            backend = "pyscf"
        else:
            raise Exception("No quantum chemistry backends installed on your syste,")

    if backend not in SUPPORTED_QCHEMISTRY_BACKENDS:
        raise Exception(str(backend) + " is not (yet) supported by tequila")

    if backend not in INSTALLED_QCHEMISTRY_BACKENDS:
        raise Exception(str(backend) + " was not found on your system")

    return INSTALLED_QCHEMISTRY_BACKENDS[backend](parameters=parameters, transformation=transformation, *args, **kwargs)