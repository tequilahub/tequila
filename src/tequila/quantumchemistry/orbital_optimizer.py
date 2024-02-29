import numpy
import typing
import copy
import warnings
from dataclasses import dataclass, field

from tequila import QCircuit, ExpectationValue, minimize, TequilaWarning
from . import QuantumChemistryBase, ParametersQC, NBodyTensor

"""
Small application that wraps a tequila VQE object and passes it to the PySCF CASSCF solver.
This way we can conveniently optimize orbitals.
This basically does what is described here in the context of orbital-optimized unitary coupled-cluster (oo-ucc): https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.2.033421
Of course we don't have to use UCC circuits but can pass whatever we want as circuit, or pass a "vqe_solver" object.

The Interface with the PySCF module follows the original PySCF article  https://arxiv.org/abs/2002.12531 (see Fig.3)

Currently this is a beta version (not extensively used in real life), so be careful when using it and please report issues on github :-)
"""

@dataclass
class OptimizeOrbitalsResult:
    
    old_molecule: QuantumChemistryBase = None # the old tequila molecule
    molecule: QuantumChemistryBase = None # the new tequila molecule with transformed orbitals
    mcscf_object:object = None # the pyscf mcscf object
    mcscf_local_data:dict = None
    mo_coeff = None # the optimized mo coefficients
    energy: float = None # the optimized energy
    iterations:int = 0

    def __call__(self, local_data, *args, **kwargs):
        # use as callback
        if "u" in local_data:
            self.rotation_matrix = copy.deepcopy(local_data["u"])
        self.mcscf_local_data=local_data
        self.iterations += 1

def optimize_orbitals(molecule, circuit=None, vqe_solver=None, pyscf_arguments=None, silent=False,
                      vqe_solver_arguments=None, initial_guess=None, return_mcscf=False, use_hcb=False, molecule_factory=None, molecule_arguments=None, *args, **kwargs):
    """

    Parameters
    ----------
    molecule: The tequila molecule whose orbitals are to be optimized
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
    args: just here for convenience
    kwargs: just here for conveniece

    Returns
    -------
        Optimized Tequila Molecule
    """

    try:
        from pyscf import mcscf
        from . import QuantumChemistryPySCF
    except Exception as exception:
        raise Exception("{}\noptimize_orbitals: Need pyscf to run (pip install pyscf)".format(str(exception)))

    if pyscf_arguments is None:
        pyscf_arguments = {"max_cycle_macro": 10, "max_cycle_micro": 3}
    no = molecule.n_orbitals
    pyscf_molecule = QuantumChemistryPySCF.from_tequila(molecule=molecule, transformation=molecule.transformation)
    mf = pyscf_molecule._get_hf()
    result=OptimizeOrbitalsResult()
    mc = mcscf.CASSCF(mf, pyscf_molecule.n_orbitals, pyscf_molecule.n_electrons)
    mc.callback=result
    c = pyscf_molecule.compute_constant_part()

    if circuit is None and vqe_solver is None:
        raise Exception("optimize_orbitals: Either provide a circuit or a callable vqe_solver")

    if use_hcb:
        if vqe_solver_arguments is None:
            vqe_solver_arguments={}
        vqe_solver_arguments["restrict_to_hcb"]=True
        # consistency check
        n_qubits = len(circuit.qubits)
        n_orbitals = molecule.n_orbitals
        if n_qubits > n_orbitals:
            warnings.warn("Potential inconsistency in orbital optimization: use_hcb is switched on but we have\n n_qubits={} in the circuit\n n_orbital={} in the molecule\n".format(n_qubits,n_orbitals), TequilaWarning)

    if molecule_arguments is None:
        molecule_arguments = {"parameters": pyscf_molecule.parameters, "transformation": molecule.transformation}

    wrapper = PySCFVQEWrapper(molecule_arguments=molecule_arguments, n_electrons=pyscf_molecule.n_electrons,
                              const_part=c, circuit=circuit, vqe_solver_arguments=vqe_solver_arguments, silent=silent,
                              vqe_solver=vqe_solver, molecule_factory=molecule_factory, *args, **kwargs)
    mc.fcisolver = wrapper
    mc.internal_rotation = True
    if pyscf_arguments is not None:
        for k, v in pyscf_arguments.items():
            if hasattr(mc, str(k)):
                setattr(mc, str(k), v)
            else:
                print("unknown arguments: {}".format(k))
    if not silent:
        print("Optimizing Orbitals with PySCF and VQE Solver:")
        print("{:25} : {}".format("pyscf_arguments", pyscf_arguments))
        print(wrapper)
    if initial_guess is not None:
        if hasattr(initial_guess, "lower"):
            if "random" in initial_guess.lower():
                scale = 0.1
                loc = 0.0
                if "scale" in kwargs:
                    scale = kwargs["scale"]
                if "loc" in kwargs:
                    loc = kwargs["loc"]
                initial_guess = numpy.eye(no) + numpy.random.normal(scale=scale, loc=loc, size=no * no).reshape(no, no)
            else:
                raise Exception("Unknown initial_guess={}".format(initial_guess.lower()))

        assert len(initial_guess.shape) == 2
        assert initial_guess.shape[0] == no
        assert initial_guess.shape[1] == no
        initial_guess = mcscf.project_init_guess(mc, initial_guess)
        mc.kernel(mo_coeff=initial_guess)
    else:
        mc.kernel()
    # make new molecule

    transformed_molecule = pyscf_molecule.transform_orbitals(orbital_coefficients=mc.mo_coeff)
    result.molecule=transformed_molecule
    result.old_molecule=molecule
    result.mo_coeff=mc.mo_coeff
    result.energy=mc.e_tot
    
    if return_mcscf:
        result.mcscf_object = mc
    
    return result

@dataclass
class PySCFVQEWrapper:
    """
    Wrapper for tequila VQE's to be compatible with PySCF orbital optimization
    """

    # needs initialization
    n_electrons: int = None
    molecule_arguments: dict = None

    # internal data
    rdm1: numpy.ndarray = None
    rdm2: numpy.ndarray = None
    one_body_integrals: numpy.ndarray = None
    two_body_integrals: numpy.ndarray = None
    history: list = field(default_factory=list)

    # optional
    const_part: float = 0.0
    silent: bool = False
    vqe_solver: typing.Callable = None
    circuit: QCircuit = None
    vqe_solver_arguments: dict = field(default_factory=dict)
    molecule_factory: typing.Callable = None

    def reorder(self, M, ordering, to):
        # convenience since we need to reorder
        # all the time
        M = NBodyTensor(elems=M, ordering=ordering)
        M.reorder(to=to)
        return M.elems

    def kernel(self, h1, h2, *args, **kwargs):
        if self.history is None:
            self.history = []
        h2of = self.reorder(h2, "mulliken", "openfermion")
        restrict_to_hcb = self.vqe_solver_arguments is not None and "restrict_to_hcb" in self.vqe_solver_arguments and \
                          self.vqe_solver_arguments["restrict_to_hcb"]

        if self.molecule_factory is None:
            molecule = QuantumChemistryBase(one_body_integrals=h1, two_body_integrals=h2of,
                                        nuclear_repulsion=self.const_part, n_electrons=self.n_electrons,
                                        **self.molecule_arguments)
        else:
            molecule = self.molecule_factory(one_body_integrals=h1, two_body_integrals=h2of,
                                        nuclear_repulsion=self.const_part, n_electrons=self.n_electrons,
                                        **self.molecule_arguments)
        if restrict_to_hcb:
            H = molecule.make_hardcore_boson_hamiltonian()
        else:
            H = molecule.make_hamiltonian()
        if self.vqe_solver is not None:
            vqe_solver_arguments = {}
            if self.vqe_solver_arguments is not None:
                vqe_solver_arguments = self.vqe_solver_arguments
            result = self.vqe_solver(H=H, circuit=self.circuit, molecule=molecule, **vqe_solver_arguments)
        elif self.circuit is None:
            raise Exception("Orbital Optimizer: Either provide a callable vqe_solver or a circuit")
        else:
            U = self.circuit
            E = ExpectationValue(H=H, U=U)
            optimizer_arguments = {}
            if self.vqe_solver_arguments is not None and "optimizer_arguments" in self.vqe_solver_arguments:
                optimizer_arguments = self.vqe_solver_arguments["optimizer_arguments"]
            if self.silent is not None and "silent" not in optimizer_arguments:
                optimizer_arguments["silent"] = True

            result = minimize(E, **optimizer_arguments)
        if hasattr(result, "circuit"):
            # potential adaptive ansatz
            U = result.circuit
            self.circuit = U
        else:
            # static ansatz
            U = self.circuit

        rdm1, rdm2 = molecule.compute_rdms(U=U, variables=result.variables, spin_free=True, get_rdm1=True, get_rdm2=True, use_hcb=restrict_to_hcb)
        rdm2 = self.reorder(rdm2, 'dirac', 'mulliken')
        if not self.silent:
            print("{:20} : {}".format("energy", result.energy))
            if len(self.history) > 0:
                print("{:20} : {}".format("deltaE", result.energy - self.history[-1].energy))
                print("{:20} : {}".format("||delta RDM1||", numpy.linalg.norm(self.rdm1 - rdm1)))
        self.history.append(result)
        self.rdm1 = rdm1
        self.rdm2 = rdm2
        self.one_body_integrals = h1
        self.two_body_integrals = h2
        return result.energy, None

    def make_rdm12(self, *args, **kwargs):
        return self.rdm1, self.rdm2

    def __str__(self):
        result = "{}\n".format(type(self).__name__)
        for k, v in self.__dict__.items():
            if k == "circuit" and v is not None:
                result += "{:30} : {}\n".format(k, "{} gates, {} parameters".format(len(v.gates),
                                                                                    len(v.extract_variables())))
            else:
                result += "{:30} : {}\n".format(k, v)
        return result
