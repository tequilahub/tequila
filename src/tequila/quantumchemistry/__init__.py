import typing
from .qc_base import ParametersQC, QuantumChemistryBase

SUPPORTED_QCHEMISTRY_BACKENDS = ["base", "psi4"]
INSTALLED_QCHEMISTRY_BACKENDS = {"base": QuantumChemistryBase}



try:
    from .psi4_interface import QuantumChemistryPsi4
    INSTALLED_QCHEMISTRY_BACKENDS["psi4"] = QuantumChemistryPsi4
except ImportError:
    pass

def show_available_modules():
    print("Available QuantumChemistry Modules:")
    for k in INSTALLED_QCHEMISTRY_BACKENDS.keys():
        print(k)

def show_supported_modules():
    print(SUPPORTED_QCHEMISTRY_BACKENDS)

def Molecule(geometry: str, basis_set: str = None, transformation: typing.Union[str, typing.Callable] = None, backend: str = None, guess_wfn=None, *args, **kwargs) -> QuantumChemistryBase:
    """
    Define a molecular geometry, basis set and fermion quibit transformation.

    Parameters
    ----------
    geometry: 
        Molecular geometry as string or as filename (needs to be in xyz format with .xyz ending)
    basis_set: 
        Quantum chemistry basis set (sto-3g, cc-pvdz, etc)
    transformation: 
        The Fermion to Qubit Transformation (jordan-wigner, bravyi-kitaev, bravyi-kitaev-tree and whatever OpenFermion supports)
    backend:
        Quantum chemistry backend (psi4, pyscf)
    guess_wfn:
        Pass down a psi4 guess wavefunction to start the scf cycle from
        can also be a filename leading to a stored wavefunction
    args:
    kwargs:

    Returns
    -------
    Molecule: object
        The Fermion to Qubit Transformation (jordan-wigner, bravyi-kitaev, bravyi-kitaev-tree and whatever OpenFermion supports)

       

    Examples
    ---------

    >>> geom = 'H 0.0 0.0 0.0\\nLi 0.0 0.0 1.6'
    >>> molecule = tq.chemistry.Molecule (geometry = geom, basis_set='sto-3g')
    >>> molecule = tq.chemistry.Molecule(geometry = geom, basis_set='sto-3g', transformation='bravyi-kitaev')

    """ 
    
    keyvals = {}
    for k, v in kwargs.items():
        if k in ParametersQC.__dict__.keys():
            keyvals[k] = v

    parameters = ParametersQC(geometry=geometry, basis_set=basis_set, multiplicity=1, **keyvals)

    if backend is None:
        if "psi4" in INSTALLED_QCHEMISTRY_BACKENDS:
            backend = "psi4"
        elif "pyscf" in INSTALLED_QCHEMISTRY_BACKENDS:
            backend = "pyscf"
        else:
            requirements = [key in kwargs for key in ["one_body_integrals", "two_body_integrals", "nuclear_repulsion", "n_orbitals"]]
            if not all(requirements):
                raise Exception("No quantum chemistry backends installed on your system\n"
                            "To use the base functionality you need to pass the following tensors via keyword\n"
                            "one_body_integrals, two_body_integrals, nuclear_repulsion, n_orbitals\n")
            else:
                backend = "base"

    if backend not in SUPPORTED_QCHEMISTRY_BACKENDS:
        raise Exception(str(backend) + " is not (yet) supported by tequila")

    if backend not in INSTALLED_QCHEMISTRY_BACKENDS:
        raise Exception(str(backend) + " was not found on your system")

    if guess_wfn is not None and backend != 'psi4':
        raise Exception("guess_wfn only works for psi4")

    if basis_set is None and backend != "base":
        raise Exception("no basis_set provided for backend={}".format(backend))
    elif basis_set is None:
        basis_set = "custom"
        parameters.basis_set=basis_set

    return INSTALLED_QCHEMISTRY_BACKENDS[backend.lower()](parameters=parameters, transformation=transformation, guess_wfn=guess_wfn, *args, **kwargs)


def MoleculeFromOpenFermion(molecule,
                            transformation: typing.Union[str, typing.Callable] = None,
                            backend: str = None,
                            *args,
                            **kwargs) -> QuantumChemistryBase:
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
        return QuantumChemistryBase.from_openfermion(molecule=molecule, transformation=transformation, *args, **kwargs)
    else:
        INSTALLED_QCHEMISTRY_BACKENDS[backend].from_openfermion(molecule=molecule, transformation=transformation, *args,
                                                                **kwargs)
