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

def Molecule(geometry: str,
             basis_set: str,
             transformation: typing.Union[str, typing.Callable] = None,
             backend: str = None,
             guess_wfn = None,
             filename = None,
             *args,
             **kwargs) -> qc_base.QuantumChemistryBase:
    """

    Parameters
    ----------
    geometry
        molecular geometry as string or as filename (needs to be in xyz format with .xyz ending)
    basis_set
        quantum chemistry basis set (sto-3g, cc-pvdz, etc)
    transformation
        The Fermion to Qubit Transformation (jordan-wigner, bravyi-kitaev, bravyi-kitaev-tree and whatever OpenFermion supports)
    backend
        quantum chemistry backend (psi4, pyscf)
    guess_wfn
        pass down a psi4 guess wavefunction to start the scf cycle from
        can also be a filename leading to a stored wavefunction
    filename
        Filename root for the backend calculations
    args
    kwargs

    Returns
    -------
        The Fermion to Qubit Transformation (jordan-wigner, bravyi-kitaev, bravyi-kitaev-tree and whatever OpenFermion supports)
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
            raise Exception("No quantum chemistry backends installed on your system")

    if backend not in SUPPORTED_QCHEMISTRY_BACKENDS:
        raise Exception(str(backend) + " is not (yet) supported by tequila")

    if backend not in INSTALLED_QCHEMISTRY_BACKENDS:
        raise Exception(str(backend) + " was not found on your system")

    if guess_wfn is not None and backend != 'psi4':
        raise Exception("guess_wfn only works for psi4")
    return INSTALLED_QCHEMISTRY_BACKENDS[backend](parameters=parameters, transformation=transformation, guess_wfn=guess_wfn, *args, **kwargs)

def MoleculeOpenFermion(molecule,
                        transformation: typing.Union[str, typing.Callable] = None,
                        backend:str=None,
                        *args,
                        **kwargs) -> qc_base.QuantumChemistryBase:
    """
    Initialize a tequila Molecule directly from an openfermion molecule object
    Parameters
    ----------
    molecule
        The openfermion molecule
    transformation
        The Fermion to Qubit Transformation (jordan-wigner, bravyi-kitaev, bravyi-kitaev-tree and whatever OpenFermion supports)
    backend
        The quantum chemistry backend, can be None in this case
    Returns
    -------
        The tequila molecule
    """
    if backend is None:
        return qc_base.QuantumChemistryBase.from_openfermion(molecule=molecule, transformation=transformation, *args, **kwargs)
    else:
        INSTALLED_QCHEMISTRY_BACKENDS[backend].from_openfermion(molecule=molecule, transformation=transformation, *args, **kwargs)
