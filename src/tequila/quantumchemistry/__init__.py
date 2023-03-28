import typing
import warnings

from tequila import TequilaWarning
from .qc_base import QuantumChemistryBase
from .chemistry_tools import ParametersQC, NBodyTensor
from .madness_interface import QuantumChemistryMadness


SUPPORTED_QCHEMISTRY_BACKENDS = ["base", "psi4", "madness", "pyscf"]
INSTALLED_QCHEMISTRY_BACKENDS = {"base": QuantumChemistryBase, "madness": QuantumChemistryMadness}

try:
    from .psi4_interface import QuantumChemistryPsi4

    INSTALLED_QCHEMISTRY_BACKENDS["psi4"] = QuantumChemistryPsi4
except ImportError:
    pass

try:
    from .pyscf_interface import QuantumChemistryPySCF

    INSTALLED_QCHEMISTRY_BACKENDS["pyscf"] = QuantumChemistryPySCF
except ImportError:
    pass


def show_available_modules():
    print("Available QuantumChemistry Modules:")
    for k in INSTALLED_QCHEMISTRY_BACKENDS.keys():
        print(k)


def show_supported_modules():
    print(SUPPORTED_QCHEMISTRY_BACKENDS)


def Molecule(geometry: str = None,
             basis_set: str = None,
             transformation: typing.Union[str, typing.Callable] = None,
             orbital_type: str = None,
             backend: str = None,
             guess_wfn=None,
             name: str = None,
             *args,
             **kwargs) -> QuantumChemistryBase:
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
    name
        name of the molecule, if not given it's auto-deduced from the geometry
        can also be done vice versa (i.e. geometry is then auto-deduced to name.xyz)
    args
    kwargs

    Returns
    -------
        The Fermion to Qubit Transformation (jordan-wigner, bravyi-kitaev, bravyi-kitaev-tree and whatever OpenFermion supports)
    """

    # failsafe for common mistake
    if "basis" in kwargs:
        warnings.warn("called molecule with keyword \"basis={0}\" converting it to \"basis_set={0}\"".format(kwargs["basis"]), TequilaWarning)
        if basis_set is not None:
            warnings.warn("did not convert as \"basis_set={}\" was already given".format(basis_set), TequilaWarning)
        basis_set=kwargs["basis"]
    
    keyvals = {}
    for k, v in kwargs.items():
        if k in ParametersQC.__dict__.keys():
            keyvals[k] = v

    if "parameters" in kwargs:
        parameters = kwargs["parameters"]
        kwargs.pop("parameters")
    else:
        parameters = ParametersQC(name=name, geometry=geometry, basis_set=basis_set, multiplicity=1, **keyvals)

    integrals_provided = all([key in kwargs for key in ["one_body_integrals", "two_body_integrals"]])
    if integrals_provided and backend is None:
        backend = "base"

    if backend is None:
        if basis_set is None or basis_set.lower() in ["madness", "mra", "pno"]:
            backend = "madness"
            if orbital_type is not None and orbital_type.lower() not in ["pno", "mra-pno"]:
                warnings.warn("only PNOs supported as orbital_type without basis set. Setting to pno - You gave={}".format(orbital_type), TequilaWarning)
            orbital_type = "pno"
        else:
            if orbital_type is not None and orbital_type.lower() not in ["hf", "native"]:
                warnings.warn("only hf and native supported as orbital_type with basis-set. Setting to hf - You gave={}".format(orbital_type), TequilaWarning)
                orbital_type = "hf"
            if orbital_type is None:
                orbital_type = "hf"

            if "psi4" in INSTALLED_QCHEMISTRY_BACKENDS:
                backend = "psi4"
            elif "pyscf" in INSTALLED_QCHEMISTRY_BACKENDS:
                backend = "pyscf"
            else:
                raise Exception("No quantum chemistry backends installed on your system")
    
    elif backend == "base":
        if not integrals_provided:
            raise Exception("No quantum chemistry backends installed on your system\n"
                            "To use the base functionality you need to pass the following tensors via keyword\n"
                            "one_body_integrals, two_body_integrals\n")
        else:
            backend = "base"

    if backend not in SUPPORTED_QCHEMISTRY_BACKENDS:
        raise Exception(str(backend) + " is not (yet) supported by tequila")

    if backend not in INSTALLED_QCHEMISTRY_BACKENDS:
        raise Exception(str(backend) + " was not found on your system")

    if guess_wfn is not None and backend != 'psi4':
        raise Exception("guess_wfn only works for psi4")

    if basis_set is None and backend.lower() not in ["base", "madness"] and not integrals_provided:
        raise Exception("no basis_set or integrals provided for backend={}".format(backend))
    elif basis_set is None:
        basis_set = "custom"
        parameters.basis_set = basis_set

    return INSTALLED_QCHEMISTRY_BACKENDS[backend.lower()](parameters=parameters, transformation=transformation, orbital_type=orbital_type,
                                                          guess_wfn=guess_wfn, *args, **kwargs)


def MoleculeFromTequila(mol, transformation=None, backend=None, *args, **kwargs):
    c, h, g = mol.get_integrals()
    parameters = mol.parameters
    if backend is None:
        if "pyscf" in INSTALLED_QCHEMISTRY_BACKENDS:
            backend = "pyscf"
        else:
            backend = "base"
    if transformation is None:
        transformation = mol.transformation
    return INSTALLED_QCHEMISTRY_BACKENDS[backend.lower()](parameters=parameters, transformation=transformation,
                                                          n_electrons=mol.n_electrons, one_body_integrals=h,
                                                          two_body_integrals=g, nuclear_repulsion=c, *args, **kwargs)


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


# needs pyscf (handeled in call)
from .orbital_optimizer import optimize_orbitals
