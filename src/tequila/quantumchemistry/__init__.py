from __future__ import annotations
from tequila import TequilaException
import warnings
import typing

try:
    from sunrise.molecules.qubit_base.qc_base import QuantumChemistryBase
    from sunrise.expval.orbital_optimizer import OptimizeOrbitalsResult
except ImportError:
    pass

try:
    from sunrise.molecules.qubit_base import SUPPORTED_QCHEMISTRY_BACKENDS
except ImportError:
    SUPPORTED_QCHEMISTRY_BACKENDS = {}

try:
    from sunrise.molecules.qubit_base import INSTALLED_QCHEMISTRY_BACKENDS
except ImportError:
    INSTALLED_QCHEMISTRY_BACKENDS = {}


def Molecule(
    geometry: str = None,
    basis_set: str = None,
    transformation: typing.Union[str, typing.Callable] = None,
    orbital_type: str = None,
    backend: str = None,
    guess_wfn=None,
    name: str = None,
    *args,
    **kwargs,
) -> "QuantumChemistryBase":
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
    try:
        from sunrise.molecules.qubit_base import Molecule as _Molecule
    except ImportError:
        raise TequilaException(
            "Project Sunrise not installed. Tequila Chemistry module has been outsourced there, keeping Tequila commands.\n Please install project-sunrise (pip install project-sunrise) to keep using these features."
        )
    return _Molecule(geometry, basis_set, transformation, orbital_type, backend, guess_wfn, name, *args, **kwargs)


def MoleculeFromTequila(mol, transformation=None, backend=None, *args, **kwargs) -> "QuantumChemistryBase":
    try:
        from sunrise.molecules.qubit_base import MoleculeFromTequila as _MoleculeFromTequila
    except ImportError:
        raise TequilaException(
            "Project Sunrise not installed. Tequila Chemistry module has been outsourced there, keeping Tequila commands.\n Please install project-sunrise (pip install project-sunrise) to keep using these features."
        )
    return _MoleculeFromTequila(mol, transformation, backend, *args, **kwargs)


def MoleculeFromOpenFermion(
    molecule, transformation: typing.Union[str, typing.Callable] = None, backend: str = None, *args, **kwargs
) -> QuantumChemistryBase:
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
        The sunrise molecule
    """
    try:
        from sunrise.molecules.qubit_base import MoleculeFromOpenFermion as _MoleculeFromOpenFermion
    except ImportError:
        raise TequilaException(
            "Project Sunrise not installed. Tequila Chemistry module has been outsourced there, keeping Tequila commands.\n Please install project-sunrise (pip install project-sunrise) to keep using these features."
        )
    return _MoleculeFromOpenFermion(molecule, transformation, backend, *args, **kwargs)


def show_available_modules():
    try:
        from sunrise.molecules.qubit_base import show_available_modules as _show_available_modules

        _show_available_modules()
    except ImportError:
        warnings.warn(
            "Project Sunrise not installed. Tequila Chemistry module has been outsourced there, keeping Tequila commands.\n Please install project-sunrise (pip install project-sunrise) to keep using these features."
        )
        print("Available QuantumChemistry Modules:")


def show_supported_modules():
    try:
        from sunrise.molecules.qubit_base import show_supported_modules as _show_supported_modules

        _show_supported_modules()
    except ImportError:
        warnings.warn(
            "Project Sunrise not installed. Tequila Chemistry module has been outsourced there, keeping Tequila commands.\n Please install project-sunrise (pip install project-sunrise) to keep using these features."
        )
        print("\n")


def optimize_orbitals(
    molecule,
    circuit=None,
    vqe_solver=None,
    pyscf_arguments=None,
    silent=False,
    vqe_solver_arguments=None,
    initial_guess=None,
    return_mcscf=False,
    use_hcb=False,
    molecule_factory=None,
    molecule_arguments=None,
    restrict_to_active_space=True,
    read_chkfile: str = None,
    save_chkfile: str = None,
    *args,
    **kwargs,
) -> OptimizeOrbitalsResult:
    """

    Parameters
    ----------
    molecule: The molecule whose orbitals are to be optimized
    circuit: The circuit that defines the ansatz to the wavefunction in the VQE
             can be None, if a customized vqe_solver is passed that can construct a circuit
    vqe_solver: The VQE solver (the default - vqe_solver=None - will take the given circuit and construct an expectationvalue out of molecule.make_hamiltonian and the given circuit)
                A customized object can be passed that needs to be callable with the following signature: vqe_solver(H=H, circuit=self.circuit, molecule=molecule, **self.vqe_solver_arguments)
    pyscf_arguments: Arguments for the MCSCF structure of PySCF, if None, the defaults are {"max_cycle_macro":10, "max_cycle_micro":3} (see here https://pyscf.org/pyscf_api_docs/pyscf.mcscf.html)
    silent: silence printout
    use_hcb: indicate if the circuit is in hardcore Boson encoding
    vqe_solver_arguments: Optional arguments for a customized vqe_solver or the default solver
                          for the default solver: vqe_solver_arguments={"optimizer_arguments":A, "restrict_to_hcb":False} where A holds the kwargs for tq.minimize
                          restrict_to_hcb keyword controls if the standard (in whatever encoding the molecule structure has) Hamiltonian is constructed or the hardcore_boson hamiltonian
    initial_guess: Initial guess for the MCSCF module of PySCF (Matrix of orbital rotation coefficients)
                   The default (None) is a unit matrix
                   predefined commands are
                        initial_guess="random"
                        initial_guess="random_loc=X_scale=Y" with X and Y being floats
                        This initialized a random guess using numpy.random.normal(loc=X, scale=Y) with X=0.0 and Y=0.1 as defaults
    return_mcscf: return the PySCF MCSCF structure after optimization
    molecule_arguments: arguments to pass to molecule_factory or default molecule constructor | only change if you know what you are doing
    read_chkfile: checkpoint file to read the initial orbitals during the CASSCF optimization. For more info see https://pyscf.org/_modules/pyscf/mcscf/chkfile.html#load_mcscf
    save_chkfile: checkpoint file to save the intermediate orbitals during the CASSCF optimization.
    args: just here for convenience
    kwargs: just here for conveniece
    """

    try:
        from sunrise.expval.orbital_optimizer import optimize_orbitals as _optimize_orbitals
    except ImportError:
        raise TequilaException(
            "Project Sunrise not installed. Tequila Chemistry module has been outsourced there, keeping Tequila commands.\n Please install project-sunrise (pip install project-sunrise) to keep using these features."
        )
    return _optimize_orbitals(
        molecule=molecule,
        circuit=circuit,
        vqe_solver=vqe_solver,
        pyscf_arguments=pyscf_arguments,
        silent=silent,
        vqe_solver_arguments=vqe_solver_arguments,
        initial_guess=initial_guess,
        return_mcscf=return_mcscf,
        use_hcb=use_hcb,
        molecule_factory=molecule_factory,
        molecule_arguments=molecule_arguments,
        restrict_to_active_space=restrict_to_active_space,
        read_chkfile=read_chkfile,
        save_chkfile=save_chkfile,
        *args,
        **kwargs,
    )
