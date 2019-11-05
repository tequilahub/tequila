from dataclasses import dataclass
from openvqe import OpenVQEParameters, typing, numpy, OpenVQEException, BitString
from openvqe.hamiltonian import HamiltonianQC, QubitHamiltonian
from openvqe.circuit import QCircuit
from openvqe.ansatz import prepare_product_state
from openvqe.circuit.exponential_gate import DecompositionFirstOrderTrotter

import openfermion


@dataclass
class ParametersQC(OpenVQEParameters):
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


class ManyBodyAmplitudes:
    """
    Class which stores ManyBodyAmplitudes
    """

    def __init__(self, one_body: numpy.ndarray = None, two_body: numpy.ndarray = None):
        self.one_body = one_body
        self.two_body = two_body

    @classmethod
    def init_from_closed_shell(cls, one_body: numpy.ndarray = None, two_body: numpy.ndarray = None):
        """
        TODO make efficient
        virt and occ are the spatial orbitals
        :param one_body: ndarray with dimensions virt, occ
        :param two_body: ndarray with dimensions virt, occ, virt, occ
        :return: correctly initialized ManyBodyAmplitudes
        """

        def alpha(ii: int) -> int:
            return 2 * ii

        def beta(ii: int) -> int:
            return 2 * ii + 1

        full_one_body = None
        if one_body is not None:
            assert (len(one_body.shape) == 2)
            nocc = one_body.shape[1]
            nvirt = one_body.shape[0]
            norb = nocc + nvirt
            full_one_body = 0.0 * numpy.ndarray(shape=[norb * 2, norb * 2])
            for i in range(nocc):
                for a in range(nvirt):
                    full_one_body[2 * a, 2 * i] = one_body[a, i]
                    full_one_body[2 * a + 1, 2 * i + 1] = one_body[a, i]

        full_two_body = None
        if two_body is not None:
            assert (len(two_body.shape) == 4)
            nocc = two_body.shape[1]
            nvirt = two_body.shape[0]
            norb = nocc + nvirt
            full_two_body = 0.0 * numpy.ndarray(shape=[norb * 2, norb * 2, norb * 2, norb * 2])
            for i in range(nocc):
                for a in range(nvirt):
                    for j in range(nocc):
                        for b in range(nvirt):
                            full_two_body[alpha(a + nocc), alpha(i), beta(b + nocc), beta(j)] = two_body[a, i, b, j]
                            full_two_body[beta(a + nocc), beta(i), alpha(b + nocc), alpha(j)] = two_body[a, i, b, j]

        return ManyBodyAmplitudes(one_body=full_one_body, two_body=full_two_body)

    def __str__(self):
        rep = type(self).__name__
        rep += "\n One-Body-Terms:\n"
        rep += str(self.one_body)
        rep += "\n Two-Body-Terms:\n"
        rep += str(self.two_body)
        return rep

    def __call__(self, i, a, j=None, b=None, *args, **kwargs):
        """
        :param i: in absolute numbers (as spin-orbital index)
        :param a: in absolute numbers (as spin-orbital index)
        :param j: in absolute numbers (as spin-orbital index)
        :param b: in absolute numbers (as spin-orbital index)
        :return: amplitude t_aijb
        """
        if j is None:
            assert (b is None)
            return self.one_body[a, i]
        else:
            return self.two_body[a, i, b, j]

    def __getitem__(self, item: tuple):
        return self.__call__(*item)

    def __setitem__(self, key: tuple, value):
        if len(key) == 2:
            self.one_body[key[0], key[1]] = value
        else:
            self.two_body[key[0], key[1], key[2], key[3]] = value
        return self

    def __rmul__(self, other):
        if self.one_body is None:
            if self.two_body is None:
                return ManyBodyAmplitudes(one_body=None, two_body=None)
            else:
                return ManyBodyAmplitudes(one_body=None, two_body=other * self.two_body)
        elif self.two_body is None:
            return ManyBodyAmplitudes(one_body=other * self.one_body, two_body=None)
        else:
            return ManyBodyAmplitudes(one_body=other * self.one_body, two_body=other * self.two_body)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.one_body) + len(self.two_body)


class QuantumChemistryBase:

    def __init__(self, parameters: ParametersQC, transformation: typing.Union[str, typing.Callable] = None):
        self.parameters = parameters
        if transformation is None:
            self.transformation = openfermion.jordan_wigner
        elif hasattr(transformation, "lower") and transformation.lower() in ["jordan-wigner", "jw", "j-w", "jordanwigner"]:
            self.transformation = openfermion.jordan_wigner
        elif hasattr(transformation, "lower") and transformation.lower() in ["bravyi-kitaev", "bk", "b-k", "bravyikitaev"]:
            self.transformation = openfermion.bravyi_kitaev
        else:
            assert(callable(transformation))
            self.transformation = transformation
        self.molecule = self.make_molecule()

    def reference_state(self) -> BitString:
        """
        :return: Hartree-Fock Reference as binary-number
        """
        n_qubits = 2 * self.n_orbitals
        l = [0]*n_qubits
        for i in range(self.n_electrons):
            l[i] = 1

        return BitString.from_array(array=l, nbits=n_qubits)

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

    def make_molecule(self) -> openfermion.MolecularData:
        raise Exception("BaseClass Method")

    def compute_ccsd_amplitudes(self) -> ManyBodyAmplitudes:
        raise Exception("BaseClass Method")

    def make_uccsd_ansatz(self, decomposition: typing.Union[DecompositionFirstOrderTrotter, typing.Callable],
                          initial_amplitudes: typing.Union[str, ManyBodyAmplitudes] = "mp2",
                          include_reference_ansatz = True) -> QCircuit:

        """
        :param decomposition: A function which is able to decompose an unitary generated by a hermitian operator
        over exp(-i t/2 H)
        :param initial_amplitudes: initial amplitudes given as ManyBodyAmplitudes structure or as string
        where 'mp2' 'ccsd' or 'zero' are possible initializations
        :param include_reference_ansatz: Also do the reference ansatz (prepare closed-shell Hartree-Fock)
        :return: Parametrized QCircuit
        """

        Uref = QCircuit()
        if include_reference_ansatz:
            Uref = prepare_product_state(self.reference_state())

        amplitudes = initial_amplitudes
        if hasattr(initial_amplitudes, "lower"):
            if initial_amplitudes.lower() == "mp2":
                amplitudes = self.compute_mp2_amplitudes()
            elif initial_amplitudes.lower() == "ccsd":
                amplitudes = self.compute_ccsd_amplitudes()
            elif initial_amplitudes.lower() == "zero":
                amplitudes = self.initialize_zero_amplitudes()
            else:
                raise OpenVQEException("Don't know how to initialize \'{}\' amplitudes".format(initial_amplitudes))

        # factor 2 counters the -1/2 convention in rotational gates
        # 1.0j makes the anti-hermitian cluster operator hermitian
        # another factor 1.0j will be added which counters the minus sign in the -1/2 convention
        generator = 1.0j * QubitHamiltonian(hamiltonian=self.__make_cluster_operator(amplitudes=2.0 * amplitudes))
        return Uref + decomposition(generators=[generator])

    def __make_cluster_operator(self, amplitudes: ManyBodyAmplitudes) -> openfermion.QubitOperator:
        """
        Creates the clusteroperator
        :param amplitudes: CCSD amplitudes
        :return: UCCSD Cluster Operator as openfermion.QubitOperator
        """

        singles = amplitudes.one_body
        if singles is None:
            singles = numpy.ndarray([self.n_orbitals*2]*2)

        op = openfermion.utils.uccsd_generator(
            single_amplitudes=singles,
            double_amplitudes=amplitudes.two_body
        )

        # @todo opernfermion has some problems with bravyi_kitaev and interactionoperators
        return self.transformation(op)

    def initialize_zero_amplitudes(self, one_body=True, two_body=True) -> ManyBodyAmplitudes:
        singles = None
        if one_body:
            singles = numpy.ndarray([self.n_orbitals*2]*2)
        doubles = None
        if two_body:
            doubles = numpy.ndarray([self.n_orbitals*2]*4)

        return ManyBodyAmplitudes(one_body=singles, two_body=doubles)

    def compute_mp2_amplitudes(self) -> ManyBodyAmplitudes:
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
        return ManyBodyAmplitudes.init_from_closed_shell(
            two_body=0.25 * numpy.einsum('abij -> aibj', amplitudes, optimize='optimize'))
