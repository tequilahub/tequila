import numpy
import typing
from dataclasses import dataclass, field

from tequila import QCircuit, ExpectationValue, minimize
from tequila.quantumchemistry import QuantumChemistryBase, QuantumChemistryPySCF, ParametersQC, NBodyTensor

"""
Small application that wraps a tequila VQE object and passes it to the PySCF CASSCF solver.
This way we can conveniently optimize orbitals.
This basically does what is described here in the context of orbital-optimized unitary coupled-cluster (oo-ucc): https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.2.033421
Of course we don't have to use UCC circuits but can pass whatever we want as circuit, or use pass a "vqe_solver" object.

The Interface with the PySCF module follows the original PySCF article  https://arxiv.org/abs/2002.12531 (see Fig.3)
"""


def optimize_orbitals(molecule, circuit=None, vqe_solver=None, pyscf_arguments={}, silent=False, vqe_solver_arguments=None, *args, **kwargs):
    try:
        from pyscf import mcscf
    except:
        raise Exception("optimize_orbitals: Need pyscf to run")

    pyscf_molecule = QuantumChemistryPySCF.from_tequila(molecule=molecule, transformation=molecule.transformation)
    mf = pyscf_molecule._get_hf()
    mc = mcscf.CASSCF(mf, pyscf_molecule.n_orbitals, pyscf_molecule.n_electrons, **pyscf_arguments)
    c = pyscf_molecule.compute_constant_part()

    if circuit is None and vqe_solver is None:
        raise Exception("optimize_orbitals: Either provide a circuit or a callable vqe_solver")

    wrapper = PySCFVQEWrapper(molecule_arguments=pyscf_molecule.parameters, n_electrons=pyscf_molecule.n_electrons, const_part=c, circuit=circuit, vqe_solver_arguments=vqe_solver_arguments, silent=silent, vqe_solver=vqe_solver, *args, **kwargs)
    mc.fcisolver = wrapper
    mc.internal_rotation = True
    if not silent:
        print("Optimizing Orbitals with PySCF and VQE Solver:")
        print(wrapper)
    mc.kernel()
    # make new molecule
    h1 = vqe_solver.one_body_integrals
    h2 = vqe_solver.two_body_integrals

    transformed_molecule=QuantumChemistryBase(nuclear_repulsion=c,
                                              one_body_integrals=h1,
                                              two_body_integrals=h2,
                                              n_electrons=pyscf_molecule.n_electrons,
                                              transformation=pyscf_molecule.transformation.transformation,
                                              parameters=pyscf_molecule.parameters)
    return QuantumChemistryPySCF.from_tequila(molecule=transformed_molecule, transformation=pyscf_molecule.transformation.transformation)

@dataclass
class PySCFVQEWrapper:
    """
    Wrapper for tequila VQE's to be compatible with PySCF orbital optimization
    """

    # needs initialization
    n_electrons: int = None
    molecule_arguments: ParametersQC = None

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
    restrict_to_hcb: bool = False

    def reorder(self, M, ordering,to):
        # convenience since we need to reorder
        # all the time
        M = NBodyTensor(elems=M, ordering=ordering)
        M.reorder(to=to)
        return M.elems


    def kernel(self, h1, h2, *args, **kwargs):
        if self.history is None:
            self.history = []
        molecule = QuantumChemistryBase(one_body_integrals=h1, two_body_integrals=h2, nuclear_repulsion=self.const_part, n_electrons=self.n_electrons, parameters=self.molecule_arguments)
        if self.restrict_to_hcb:
            H = molecule.make_hardcore_boson_hamiltonian()
        else:
            H = molecule.make_hamiltonian()
        if self.vqe_solver is not None:
            result = self.vqe_solver(H=H, circuit=self.circuit, **self.vqe_solver_arguments)
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
        rdm1, rdm2 = molecule.compute_rdms(U=U, variables=result.variables, spin_free=True, get_rdm1=True, get_rdm2=True)
        rdm2 = self.reorder(rdm2, 'dirac', 'mulliken')
        if not self.silent:
            print("{:20} : {}".format("energy", result.energy))
            if len(self.history) > 0:
                print("{:20} : {}".format("deltaE", result.energy-self.history[-1].energy))
                print("{:20} : {}".format("||delta RDM1||", numpy.linalg.norm(self.rdm1 - rdm1)))
        self.history.append(result)
        self.rdm1 = rdm1
        self.rdm2 = rdm2
        self.one_body_integrals=h1
        self.two_body_integrals=h2
        return result.energy, [0.0]

    def make_rdm12(self, *args, **kwargs):
        return self.rdm1, self.rdm2

    def __str__(self):
        result="{}\n".format(type(self).__name__)
        for k,v in self.__dict__.items():
            if k == "circuit" and v is not None:
                result += "{:30} : {}\n".format(k, "{} gates, {} parameters".format(len(v.gates), len(v.extract_variables())))
            else:
                result += "{:30} : {}\n".format(k,v)
        return result

if __name__ == "__main__":
    import tequila as tq

    mol = tq.Molecule(geometry="Li 0.0 0.0 0.0\nH 0.0 0.0 3.0", basis_set="STO-3G")
    circuit = mol.make_upccgsd_ansatz(name="UpCCD")
    mol2 = optimize_orbitals(molecule=mol, circuit=circuit)

