from dataclasses import dataclass
from tequila import TequilaException, BitString
from tequila.hamiltonian import HamiltonianQC, QubitHamiltonian

from tequila.circuit import QCircuit, Variable, gates

import typing, numpy, numbers
import openfermion
from openfermion.hamiltonians import MolecularData

def prepare_product_state(state: BitString) -> QCircuit:
    """
    Small convenience function
    :param state: product state encoded into a bitstring
    :return: unitary circuit which prepares the product state
    """
    result = QCircuit()
    for i, v in enumerate(state.array):
        if v == 1:
            result += gates.X(target=i)
    return result

@dataclass
class ParametersQC:
    """
    Specialization of ParametersHamiltonian
    Parameters for the HamiltonianQC class
    """
    basis_set: str = ''  # Quantum chemistry basis set
    geometry: str = ''  # geometry of the underlying molecule (units: Angstrom!), this can be a filename leading to an .xyz file or the geometry given as a string
    description: str = ''
    multiplicity: int = 1
    charge: int = 0
    closed_shell: bool = True
    filename: str = "molecule"

    @property
    def molecular_data_param(self) -> dict:
        """
        :return: Give back all parameters for the MolecularData format from openfermion as dictionary
        """
        return {'basis': self.basis_set, 'geometry': self.get_geometry(), 'description': self.description,
                'charge': self.charge, 'multiplicity': self.multiplicity, 'filename': self.filename
                }

    @staticmethod
    def format_element_name(string):
        """
        OpenFermion uses case sensitive hash tables for chemical elements
        I.e. you need to name Lithium: 'Li' and 'li' or 'LI' will not work
        this conenience function does the naming
        :return: first letter converted to upper rest to lower
        """
        assert (len(string) > 0)
        assert (isinstance(string, str))
        fstring = string[0].upper() + string[1:].lower()
        return fstring

    @staticmethod
    def convert_to_list(geometry):
        """
        Convert a molecular structure given as a string into a list suitable for openfermion
        :param geometry: a string specifing a mol. structure. E.g. geometry="h 0.0 0.0 0.0\n h 0.0 0.0 1.0"
        :return: A list with the correct format for openferion E.g return [ ['h',[0.0,0.0,0.0], [..]]
        """
        result = []
        for line in geometry.split('\n'):
            words = line.split()
            if len(words) != 4:  break
            try:
                tmp = (ParametersQC.format_element_name(words[0]),
                       (float(words[1]), float(words[2]), float(words[3])))
                result.append(tmp)
            except ValueError:
                print("get_geometry list unknown line:\n ", line, "\n proceed with caution!")
        return result

    def get_geometry(self):
        """
        Returns the geometry
        If a xyz filename was given the file is read out
        otherwise it is assumed that the geometry was given as string
        which is then reformated as a list usable as input for openfermion
        :return: geometry as list
        e.g. [(h,(0.0,0.0,0.35)),(h,(0.0,0.0,-0.35))]
        Units: Angstrom!
        """
        if self.geometry.split('.')[-1] == 'xyz':
            geomstring, comment = self.read_xyz_from_file(self.geometry)
            self.description = comment
            return self.convert_to_list(geomstring)
        elif self.geometry is not None:
            return self.convert_to_list(self.geometry)
        else:
            raise Exception("Parameters.qc.geometry is None")

    @staticmethod
    def read_xyz_from_file(filename):
        """
        Read XYZ filetype for molecular structures
        https://en.wikipedia.org/wiki/XYZ_file_format
        Units: Angstrom!
        :param filename:
        :return:
        """
        with open(filename, 'r') as file:
            content = file.readlines()
            natoms = int(content[0])
            comment = str(content[1]).strip('\n')
            coord = ''
            for i in range(natoms):
                coord += content[2 + i]
            return coord, comment


class Amplitudes:
    """
    Many Body amplitudes
    stored as dictionaries with keys corresponding to indices of the operators
    key = (i,a,j,b) --> a^\dagger_a a_i a^\dagger_b a_j - h.c.
    accordingly
    key = (i,a) --> a^\dagger_a a_i
    """

    def __init__(self, closed_shell: bool = None, data: typing.Dict[typing.Tuple, numbers.Number] = None):
        self.data = dict()
        if data is not None:
            if closed_shell:
                self.data = self.transform_closed_shell_indices(data)
            else:
                self.data = data

    def __repr__(self):
        return self.data.__repr__()

    def __str__(self):
        return self.data.__str__()

    def export_parameter_dictionary(self):
        result = dict()
        for k, v in self.items():
            if not numpy.isclose(numpy.abs(v), 0.0):
                result[Variable(k)] = v
        return result

    def transform_closed_shell_indices(self, data: typing.Dict[typing.Tuple, numbers.Number]) -> typing.Dict[
        typing.Tuple, numbers.Number]:
        transformed = dict()
        for key, value in data.items():
            if len(key) == 2:
                transformed[(2 * key[0], 2 * key[1])] = value
                transformed[(2 * key[0] + 1, 2 * key[1] + 1)] = value
            if len(key) == 4:
                transformed[(2 * key[0], 2 * key[1], 2 * key[2] + 1, 2 * key[3] + 1)] = value
                transformed[(2 * key[0] + 1, 2 * key[1] + 1, 2 * key[2], 2 * key[3])] = value
            else:
                raise Exception("???")
            return transformed

    @classmethod
    def from_ndarray(cls, array: numpy.ndarray, closed_shell=None,
                     index_offset: typing.Tuple[int, int, int, int] = None):
        """
        :param array: The array to convert
        :param closed_shell: amplitudes are given as closed-shell array (i.e alpha-alpha, beta-beta)
        :param index_offsets: indices will start from 0 but are supposed to start from index_offset
        :return:
        """
        assert all([x == array.shape[0] for x in array.shape])  # all indices should run over ALL orbitals
        data = dict(numpy.ndenumerate(array))
        if index_offset is not None:
            offset_data = dict()
            for key, value in data.items():
                keyx = tuple(a + b for a, b in zip(key, index_offset))
                offset_data[keyx] = value
            data = offset_data
        return cls(data=data, closed_shell=closed_shell)

    def __rmul__(self, other):
        data = dict()
        for k, v in self.data.items():
            data[k] = v * other
        return Amplitudes(data=data)

    def __neg__(self):
        data = dict()
        for k, v in self.data.items():
            data[k] = -v
        return Amplitudes(data=data)

    def items(self):
        return self.data.items()

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def __len__(self):
        return self.data.__len__()


class QuantumChemistryBase:

    def __init__(self, parameters: ParametersQC, transformation: typing.Union[str, typing.Callable] = None):
        self.parameters = parameters
        if transformation is None:
            self.transformation = openfermion.jordan_wigner
        elif hasattr(transformation, "lower") and transformation.lower() in ["jordan-wigner", "jw", "j-w",
                                                                             "jordanwigner"]:
            self.transformation = openfermion.jordan_wigner
        elif hasattr(transformation, "lower") and transformation.lower() in ["bravyi-kitaev", "bk", "b-k",
                                                                             "bravyikitaev"]:
            self.transformation = openfermion.bravyi_kitaev
        elif hasattr(transformation, "lower") and transformation.lower() in ["bravyi-kitaev-tree", "bkt",
                                                                             "bravykitaevtree", "b-k-t"]:
            self.transformation = openfermion.bravyi_kitaev_tree
        elif hasattr(transformation, "lower"):
            self.transformation = getattr(openfermion, transformation.lower())
        else:
            assert (callable(transformation))
            self.transformation = transformation
        self.molecule = self.make_molecule()

    def make_excitation_operator(self, indices: typing.Iterable[typing.Tuple[int, int]]) -> QubitHamiltonian:
        """
        Creates the excitation operator: a^\dagger_{a_0} a_{i_0} a^\dagger{a_1}a_{i_1} ... - h.c.
        And gives it back multiplied with 1j to make it hermitian
        :param indices: List of tuples [(a_0, i_0), (a_1, i_1), ... ], in spin-orbital notation (alpha odd numbers, beta even numbers)
        :return: Transformed qubit excitation operator, depends on self.transformation
        """
        # convert to openfermion input format
        ofi = []
        dag = []
        for pair in indices:
            assert (len(pair) == 2)
            ofi += [(pair[0], 1), (pair[1], 0)]
            dag += [(pair[0], 0), (pair[1], 1)]

        op = openfermion.FermionOperator(tuple(ofi), 1.j) #1j makes it hermitian
        op += openfermion.FermionOperator(tuple(reversed(dag)), -1.j)

        return QubitHamiltonian(hamiltonian=self.transformation(op))

    
    def reference_state(self) -> BitString:
        """
        Does a really lazy workaround ... but it works
        :return: Hartree-Fock Reference as binary-number
        """

        string = ""

        n_qubits = 2 * self.n_orbitals
        l = [0] * n_qubits
        for i in range(self.n_electrons):
            string += str(i) + "^ "
            l[i] = 1
        fop = openfermion.FermionOperator(string, 1.0)

        op = QubitHamiltonian(hamiltonian=self.transformation(fop))
        from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
        wfn = QubitWaveFunction.from_int(0, n_qubits=n_qubits)
        wfn = wfn.apply_qubitoperator(operator=op)
        assert (len(wfn.keys()) == 1)
        keys = [k for k in wfn.keys()]
        return keys[-1]

    def make_molecule(self) -> MolecularData:
        """
        Creates a molecule in openfermion format by running psi4 and extracting the data
        Will check for previous outputfiles before running
        :param parameters: An instance of ParametersQC, which also holds an instance of ParametersPsi4 via parameters.psi4
        The molecule will be saved in parameters.filename, if this file exists before the call the molecule will be imported from the file
        :return: the molecule in openfermion.MolecularData format
        """
        molecule = MolecularData(**self.parameters.molecular_data_param)
        # try to load

        do_compute = True
        if self.parameters.filename:
            try:
                import os
                if os.path.exists(self.parameters.filename):
                    molecule.load()
                    do_compute = False
            except OSError:
                do_compute = True

        if do_compute:
            molecule = self.do_make_molecule(molecule)

        molecule.save()
        return molecule

    def do_make_molecule(self):
        raise TequilaException("Needs to be overwritten by inherited backend class")

    @property
    def n_orbitals(self) -> int:
        return self.molecule.n_orbitals

    @property
    def n_electrons(self) -> int:
        return self.molecule.n_electrons

    @property
    def n_alpha_electrons(self) -> int:
        return self.molecule.get_n_alpha_electrons()

    @property
    def n_beta_electrons(self) -> int:
        return self.molecule.get_n_beta_electrons()

    def make_hamiltonian(self) -> HamiltonianQC:
        return HamiltonianQC(molecule=self.molecule, transformation=self.transformation)

    def compute_ccsd_amplitudes(self) -> Amplitudes:
        raise Exception("BaseClass Method")

    def prepare_reference(self):
        return prepare_product_state(self.reference_state())

    def make_uccsd_ansatz(self,
                          trotter_steps: int,
                          initial_amplitudes: typing.Union[str, Amplitudes] = "mp2",
                          include_reference_ansatz=True,
                          trotter_parameters: gates.TrotterParameters = None) -> QCircuit:

        """
        :param initial_amplitudes: initial amplitudes given as ManyBodyAmplitudes structure or as string
        where 'mp2' 'ccsd' or 'zero' are possible initializations
        :param include_reference_ansatz: Also do the reference ansatz (prepare closed-shell Hartree-Fock)
        :return: Parametrized QCircuit
        """

        Uref = QCircuit()
        if include_reference_ansatz:
            Uref = self.prepare_reference()

        amplitudes = initial_amplitudes
        if hasattr(initial_amplitudes, "lower"):
            if initial_amplitudes.lower() == "mp2":
                amplitudes = self.compute_mp2_amplitudes()
            elif initial_amplitudes.lower() == "ccsd":
                amplitudes = self.compute_ccsd_amplitudes()
            elif initial_amplitudes.lower() == "zero":
                amplitudes = self.initialize_zero_amplitudes()
            else:
                raise TequilaException("Don't know how to initialize \'{}\' amplitudes".format(initial_amplitudes))

        generators = []
        variables = []
        for key, t in amplitudes.items():
            assert (len(key) % 2 == 0)
            if not numpy.isclose(t, 0.0):
                variables.append(2.0*Variable(name=key)) # 2.0 for convention angle/2 in ExpPauli Gates
                indices = [(key[2 * i], key[2 * i + 1]) for i in range(len(key)//2)]
                generators.append(self.make_excitation_operator(indices=indices))

        return Uref + gates.Trotterized(generators=generators, angles=variables, steps=trotter_steps, parameters=trotter_parameters)

    def initialize_zero_amplitudes(self) -> Amplitudes:
        # function not needed anymore
        return Amplitudes()

    def compute_mp2_amplitudes(self) -> Amplitudes:
        """
        Compute closed-shell mp2 amplitudes (open-shell comming at some point)

        t(a,i,b,j) = 0.25 * g(a,i,b,j)/(e(i) + e(j) -a(i) - b(j) )

        :return:
        """
        assert self.parameters.closed_shell
        g = self.molecule.two_body_integrals
        fij = self.molecule.orbital_energies
        nocc = self.n_alpha_electrons
        ei = fij[:nocc]
        ai = fij[nocc:]
        abgij = g[nocc:, nocc:, :nocc, :nocc]
        amplitudes = abgij * 1.0 / (
                ei.reshape(1, 1, -1, 1) + ei.reshape(1, 1, 1, -1) - ai.reshape(-1, 1, 1, 1) - ai.reshape(1, -1, 1,
                                                                                                         1))
        E = 2.0 * numpy.einsum('abij,abij->', amplitudes, abgij) - numpy.einsum('abji,abij', amplitudes, abgij,
                                                                                optimize='optimize')
        self.molecule.mp2_energy = E + self.molecule.hf_energy
        return Amplitudes.from_ndarray(array=0.25 * numpy.einsum('abij -> aibj', amplitudes, optimize='optimize'),
                                       closed_shell=True, index_offset=(nocc, 0, nocc, 0))
