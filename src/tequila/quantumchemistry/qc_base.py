from dataclasses import dataclass
from tequila import TequilaException, BitString, QubitWaveFunction
from tequila.hamiltonian import QubitHamiltonian, paulis

from tequila.circuit import QCircuit, gates
from tequila.objective.objective import Variable
from tequila.utils import to_float

import typing, numpy, numbers

import openfermion
from openfermion.hamiltonians import MolecularData


def prepare_product_state(state: BitString) -> QCircuit:
    """Small convenience function

    Parameters
    ----------
    state :
        product state encoded into a bitstring
    state: BitString :
        

    Returns
    -------
    type
        unitary circuit which prepares the product state

    """
    result = QCircuit()
    for i, v in enumerate(state.array):
        if v == 1:
            result += gates.X(target=i)
    return result


@dataclass
class ParametersQC:
    """Specialization of ParametersHamiltonian"""
    basis_set: str = ''  # Quantum chemistry basis set
    geometry: str = ''  # geometry of the underlying molecule (units: Angstrom!), this can be a filename leading to an .xyz file or the geometry given as a string
    description: str = ''
    multiplicity: int = 1
    charge: int = 0
    closed_shell: bool = True
    name: str = "molecule"

    @property
    def filename(self):
        """ """
        return "{}_{}".format(self.name, self.basis_set)

    @property
    def molecular_data_param(self) -> dict:
        """:return: Give back all parameters for the MolecularData format from openfermion as dictionary"""
        return {'basis': self.basis_set, 'geometry': self.get_geometry(), 'description': self.description,
                'charge': self.charge, 'multiplicity': self.multiplicity, 'filename': self.filename
                }

    @staticmethod
    def format_element_name(string):
        """OpenFermion uses case sensitive hash tables for chemical elements
        I.e. you need to name Lithium: 'Li' and 'li' or 'LI' will not work
        this conenience function does the naming
        :return: first letter converted to upper rest to lower

        Parameters
        ----------
        string :
            

        Returns
        -------

        """
        assert (len(string) > 0)
        assert (isinstance(string, str))
        fstring = string[0].upper() + string[1:].lower()
        return fstring

    @staticmethod
    def convert_to_list(geometry):
        """Convert a molecular structure given as a string into a list suitable for openfermion

        Parameters
        ----------
        geometry :
            a string specifing a mol. structure. E.g. geometry="h 0.0 0.0 0.0\n h 0.0 0.0 1.0"

        Returns
        -------
        type
            A list with the correct format for openferion E.g return [ ['h',[0.0,0.0,0.0], [..]]

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

    def get_geometry_string(self) -> str:
        """returns the geometry as a string
        :return: geometrystring

        Parameters
        ----------

        Returns
        -------

        """
        if self.geometry.split('.')[-1] == 'xyz':
            geomstring, comment = self.read_xyz_from_file(self.geometry)
            if comment is not None:
                self.description = comment
            return geomstring
        else:
            return self.geometry

    def get_geometry(self):
        """Returns the geometry
        If a xyz filename was given the file is read out
        otherwise it is assumed that the geometry was given as string
        which is then reformated as a list usable as input for openfermion
        :return: geometry as list
        e.g. [(h,(0.0,0.0,0.35)),(h,(0.0,0.0,-0.35))]
        Units: Angstrom!

        Parameters
        ----------

        Returns
        -------

        """
        if self.geometry.split('.')[-1] == 'xyz':
            geomstring, comment = self.read_xyz_from_file(self.geometry)
            if self.description == '':
                self.description = comment
            if self.name == "molecule":
                self.name = self.geometry.split('.')[0]
            return self.convert_to_list(geomstring)
        elif self.geometry is not None:
            return self.convert_to_list(self.geometry)
        else:
            raise Exception("Parameters.qc.geometry is None")

    @staticmethod
    def read_xyz_from_file(filename):
        """Read XYZ filetype for molecular structures
        https://en.wikipedia.org/wiki/XYZ_file_format
        Units: Angstrom!

        Parameters
        ----------
        filename :
            return:

        Returns
        -------

        """
        with open(filename, 'r') as file:
            content = file.readlines()
            natoms = int(content[0])
            comment = str(content[1]).strip('\n')
            coord = ''
            for i in range(natoms):
                coord += content[2 + i]
            return coord, comment


@dataclass
class ClosedShellAmplitudes:
    """ """
    tIjAb: numpy.ndarray = None
    tIA: numpy.ndarray = None

    def make_parameter_dictionary(self, threshold=1.e-8):
        """

        Parameters
        ----------
        threshold :
             (Default value = 1.e-8)

        Returns
        -------

        """
        variables = {}
        if self.tIjAb is not None:
            nvirt = self.tIjAb.shape[2]
            nocc = self.tIjAb.shape[0]
            assert (self.tIjAb.shape[1] == nocc and self.tIjAb.shape[3] == nvirt)
            for (I, J, A, B), value in numpy.ndenumerate(self.tIjAb):
                if not numpy.isclose(value, 0.0, atol=threshold):
                    variables[(nocc + A, I, nocc + B, J)] = value
        if self.tIA is not None:
            nocc = self.tIA.shape[0]
            for (I, A), value, in numpy.ndenumerate(self.tIA):
                if not numpy.isclose(value, 0.0, atol=threshold):
                    variables[(A + nocc, I)] = value

        return dict(sorted(variables.items(), key=lambda x: numpy.abs(x[1]), reverse=True))


@dataclass
class Amplitudes:
    """Coupled-Cluster Amplitudes
    We adopt the Psi4 notation for consistency
    I,A for alpha
    i,a for beta

    Parameters
    ----------

    Returns
    -------

    """

    @classmethod
    def from_closed_shell(cls, cs: ClosedShellAmplitudes):
        """
        Initialize from closed-shell Amplitude structure

        Parameters
        ----------
        cs: ClosedShellAmplitudes :
            

        Returns
        -------

        """
        tijab = cs.tIjAb - numpy.einsum("ijab -> ijba", cs.tIjAb, optimize='greedy')
        return cls(tIjAb=cs.tIjAb, tIA=cs.tIA, tiJaB=cs.tIjAb, tia=cs.tIA, tijab=tijab, tIJAB=tijab)

    tIjAb: numpy.ndarray = None
    tIA: numpy.ndarray = None
    tiJaB: numpy.ndarray = None
    tijab: numpy.ndarray = None
    tIJAB: numpy.ndarray = None
    tia: numpy.ndarray = None

    def make_parameter_dictionary(self, threshold=1.e-8):
        """

        Parameters
        ----------
        threshold :
             (Default value = 1.e-8)
             Neglect amplitudes below the threshold

        Returns
        -------
        Dictionary of tequila variables (hash is in the style of (a,i,b,j))

        """
        variables = {}
        if self.tIjAb is not None:
            nvirt = self.tIjAb.shape[2]
            nocc = self.tIjAb.shape[0]
            assert (self.tIjAb.shape[1] == nocc and self.tIjAb.shape[3] == nvirt)

            for (I, j, A, b), value in numpy.ndenumerate(self.tIjAb):
                if not numpy.isclose(value, 0.0, atol=threshold):
                    variables[(2 * (nocc + A), 2 * I, 2 * (nocc + b) + 1, j + 1)] = value
            for (i, J, a, B), value in numpy.ndenumerate(self.tiJaB):
                if not numpy.isclose(value, 0.0, atol=threshold):
                    variables[(2 * (nocc + a) + 1, 2 * i + 1, 2 * (nocc + B), J)] = value
            for (i, j, a, b), value in numpy.ndenumerate(self.tijab):
                if not numpy.isclose(value, 0.0, atol=threshold):
                    variables[(2 * (nocc + a) + 1, 2 * i + 1, 2 * (nocc + b) + 1, j + 1)] = value
            for (I, J, A, B), value in numpy.ndenumerate(self.tijab):
                if not numpy.isclose(value, 0.0, atol=threshold):
                    variables[(2 * (nocc + A), 2 * I, 2 * (nocc + B), J)] = value

        if self.tIA is not None:
            nocc = self.tIjAb.shape[0]
            assert (self.tia.shape[0] == nocc)
            for (I, A), value, in numpy.ndenumerate(self.tIA):
                if not numpy.isclose(value, 0.0, atol=threshold):
                    variables[(2 * (A + nocc), 2 * I)] = value
            for (i, a), value, in numpy.ndenumerate(self.tIA):
                if not numpy.isclose(value, 0.0, atol=threshold):
                    variables[(2 * (a + nocc) + 1, 2 * i + 1)] = value

        return variables


@dataclass
class TwoBodyTensor:
    """
    Convenience class for reordering of two-body tensors
    """
    hPQrs: numpy.ndarray = None
    # default ordering is 'chem', assumes that we get data from psi4
    scheme: str = 'chem'

    def is_openfermion(self) -> bool:
        """
        Checks whether current ordering scheme is 'openfermion'
        """
        if self.scheme == 'openfermion' or self.scheme == 'of':
            return True
        else:
            return False

    def is_chem(self) -> bool:
        """
        Checks whether current ordering scheme is 'chem'
        """
        if self.scheme == 'chem' or self.scheme == 'c':
            return True
        else:
            return False

    def is_phys(self) -> bool:
        """
        Checks whether current ordering scheme is 'phys'
        """
        if self.scheme == 'phys' or self.scheme == 'p':
            return True
        else:
            return False

    def reorder(self, to: str = 'of'):
        """
        Function to reorder tensors according to some convention

        Parameters
        ----------

        to :
            Ordering scheme of choice.
            'openfermion', 'of' (default) :
                openfermion - ordering, corresponds to integrals of the type
                h^pq_rs = int p(1)* q(2)* O(1,2) r(2) s(1) (O(1,2)
                with operators a^pq_rs = a^p a^q a_r a_s (a^p == a^dagger_p)
                currently needed for dependencies on openfermion-library
            'chem', 'c' :
                quantum chemistry ordering, collect particle terms,
                more convenient for real-space methods
                h^pq_rs = int p(1) q(1) O(1,2) r(2) s(2)
                This is output by psi4
            'phys', 'p' :
                typical physics ordering, integrals of type
                h^pq_rs = int p(1)* q(2)* O(1,2) r(1) s(2)
                with operators a^pq_rs = a^p a^q a_s a_r

            Returns
            -------
            type
                the two-body tensors in chosen ordering (openfermion per default)
        """
        to = to.lower()

        if self.is_chem():
            if to == 'chem' or to == 'c':
                pass
            elif to == 'openfermion' or to == 'of':
                self.hPQrs = numpy.einsum("psqr -> pqrs", self.hPQrs, optimize='greedy')
                self.scheme = 'openfermion'
            elif to == 'phys' or to == 'p':
                self.hPQrs = numpy.einsum("prqs -> pqrs", self.hPQrs, optimize='greedy')
                self.scheme = 'phys'
        elif self.is_openfermion():
            if to == 'chem' or to == 'c':
                self.hPQrs = numpy.einsum("pqrs -> psqr", self.hPQrs, optimize='greedy')
                self.scheme = 'chem'
            elif to == 'openfermion' or to == 'of':
                pass
            elif to == 'phys' or to == 'p':
                self.hPQrs = numpy.einsum("pqrs -> pqsr", self.hPQrs, optimize='greedy')
                self.scheme = 'phys'
        elif self.is_phys():
            if to == 'chem' or to == 'c':
                self.hPQrs = numpy.einsum("pqrs -> prqs", self.hPQrs, optimize='greedy')
                self.scheme = 'chem'
            elif to == 'openfermion' or to == 'of':
                self.hPQrs = numpy.einsum("pqsr -> pqrs", self.hPQrs, optimize='greedy')
                self.scheme = 'openfermion'
            elif to == 'phys' or to == 'p':
                pass

    def get_hPQrs(self) -> numpy.ndarray:
        """ """
        return self.hPQrs


class QuantumChemistryBase:
    """ """

    def __init__(self, parameters: ParametersQC,
                 transformation: typing.Union[str, typing.Callable] = None,
                 active_orbitals: list = None,
                 reference: list = None,
                 *args,
                 **kwargs):

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
            trafo = getattr(openfermion, transformation.lower())
            self.transformation = lambda x: trafo(x, *args, **kwargs)
        else:
            assert (callable(transformation))
            self.transformation = transformation

        if "molecule" in kwargs:
            self.molecule = kwargs["molecule"]
        else:
            self.molecule = self.make_molecule(*args, **kwargs)

        assert (parameters.basis_set.lower() == self.molecule.basis.lower())
        assert (parameters.multiplicity == self.molecule.multiplicity)
        assert (parameters.charge == self.molecule.charge)
        self.active_space = self._make_active_space_data(active_orbitals=active_orbitals, reference=reference)

    def _make_active_space_data(self, active_orbitals, reference=None):
        """
        Small helper function
        Internal use only
        Parameters
        ----------
        active_orbitals: dictionary :
            list: Give a list of spatial orbital indices
            i.e. occ = [0,1,3] means that spatial orbital 0, 1 and 3 are used
        reference: (Default value=None)
            List of orbitals which form the reference
            Can be given in the same format as active_orbitals
            If given as None then the first N_electron/2 orbitals are taken
            for closed-shell systems.

        Returns
        -------
        Dataclass with active indices and reference indices (in spatial notation)

        """

        if active_orbitals is None:
            return None

        @dataclass
        class ActiveSpaceData:
            active_orbitals: list  # active orbitals (spatial, c1)
            reference_orbitals: list  # reference orbitals (spatial, c1)

            def __str__(self):
                result = "Active Space Data:\n"
                result += "{key:15} : {value:15} \n".format(key="active_orbitals", value=str(self.active_orbitals))
                result += "{key:15} : {value:15} \n".format(key="reference_orbitals",
                                                            value=str(self.reference_orbitals))
                result += "{key:15} : {value:15} \n".format(key="frozen_docc", value=str(self.frozen_docc))
                result += "{key:15} : {value:15} \n".format(key="frozen_uocc", value=str(self.frozen_uocc))
                return result

            @property
            def frozen_reference_orbitals(self):
                return [i for i in self.reference_orbitals if i not in self.active_orbitals]

            @property
            def active_reference_orbitals(self):
                return [i for i in self.reference_orbitals if i in self.active_orbitals]

        if reference is None:
            # auto assignment only for closed-shell
            assert (self.n_electrons % 2 == 0)
            reference = sorted([i for i in range(self.n_electrons // 2)])

        return ActiveSpaceData(active_orbitals=sorted(active_orbitals),
                               reference_orbitals=sorted(reference))

    @classmethod
    def from_openfermion(cls, molecule: openfermion.MolecularData,
                         transformation: typing.Union[str, typing.Callable] = None,
                         *args,
                         **kwargs):
        """
        Initialize direclty from openfermion MolecularData object

        Parameters
        ----------
        molecule
            The openfermion molecule
        Returns
        -------
            The Tequila molecule
        """
        parameters = ParametersQC(basis_set=molecule.basis, geometry=molecule.geometry,
                                  description=molecule.description, multiplicity=molecule.multiplicity,
                                  charge=molecule.charge)
        return cls(parameters=parameters, transformation=transformation, molecule=molecule, *args, **kwargs)

    def make_excitation_generator(self, indices: typing.Iterable[typing.Tuple[int, int]]) -> QubitHamiltonian:
        """
        Notes
        ----------
        Creates the transformed hermitian generator of UCC type unitaries:
              M(a^\dagger_{a_0} a_{i_0} a^\dagger{a_1}a_{i_1} ... - h.c.)
              where the qubit map M depends is self.transformation

        Parameters
        ----------
        indices : typing.Iterable[typing.Tuple[int, int]] :
            List of tuples [(a_0, i_0), (a_1, i_1), ... ] - recommended format, in spin-orbital notation (alpha odd numbers, beta even numbers)
            can also be given as one big list: [a_0, i_0, a_1, i_1 ...]
        Returns
        -------
        type
            1j*Transformed qubit excitation operator, depends on self.transformation
        """
        # check indices and convert to list of tuples if necessary
        if len(indices) == 0:
            raise TequilaException("make_excitation_operator: no indices given")
        elif not isinstance(indices[0], typing.Iterable):
            if len(indices) % 2 != 0:
                raise TequilaException("make_excitation_generator: unexpected input format of indices\n"
                                       "use list of tuples as [(a_0, i_0),(a_1, i_1) ...]\n"
                                       "or list as [a_0, i_0, a_1, i_1, ... ]\n"
                                       "you gave: {}".format(indices))
            converted = [(indices[2 * i], indices[2 * i + 1]) for i in range(len(indices) // 2)]
        else:
            converted = indices

        # convert to openfermion input format
        ofi = []
        dag = []
        for pair in converted:
            assert (len(pair) == 2)
            ofi += [(int(pair[0]), 1),
                    (int(pair[1]), 0)]  # openfermion does not take other types of integers like numpy.int64
            dag += [(int(pair[0]), 0), (int(pair[1]), 1)]

        op = openfermion.FermionOperator(tuple(ofi), 1.j)  # 1j makes it hermitian
        op += openfermion.FermionOperator(tuple(reversed(dag)), -1.j)
        qop = QubitHamiltonian(qubit_hamiltonian=self.transformation(op))

        # check if the operator is hermitian and cast coefficients to floats
        # in order to avoid trouble with the simulation backends
        assert qop.is_hermitian()
        for k, v in qop.qubit_operator.terms.items():
            qop.qubit_operator.terms[k] = to_float(v)

        qop = qop.simplify()
        return qop

    def reference_state(self, reference_orbitals: list = None, n_qubits: int = None) -> BitString:
        """Does a really lazy workaround ... but it works
        :return: Hartree-Fock Reference as binary-number

        Parameters
        ----------
        reference_orbitals: list:
            give list of doubly occupied orbitals
            default is None which leads to automatic list of the
            first n_electron/2 orbitals

        Returns
        -------

        """

        if reference_orbitals is None:
            reference_orbitals = [i for i in range(self.n_electrons // 2)]

        spin_orbitals = sorted([2 * i for i in reference_orbitals] + [2 * i + 1 for i in reference_orbitals])

        if n_qubits is None:
            n_qubits = 2 * self.n_orbitals

        string = ""

        for i in spin_orbitals:
            string += str(i) + "^ "

        fop = openfermion.FermionOperator(string, 1.0)

        op = QubitHamiltonian(qubit_hamiltonian=self.transformation(fop))
        from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
        wfn = QubitWaveFunction.from_int(0, n_qubits=n_qubits)
        wfn = wfn.apply_qubitoperator(operator=op)
        assert (len(wfn.keys()) == 1)
        keys = [k for k in wfn.keys()]
        return keys[-1]

    def make_molecule(self, *args, **kwargs) -> MolecularData:
        """Creates a molecule in openfermion format by running psi4 and extracting the data
        Will check for previous outputfiles before running
        Will not recompute if a file was found

        Parameters
        ----------
        parameters :
            An instance of ParametersQC, which also holds an instance of ParametersPsi4 via parameters.psi4
            The molecule will be saved in parameters.filename, if this file exists before the call the molecule will be imported from the file

        Returns
        -------
        type
            the molecule in openfermion.MolecularData format

        """
        molecule = MolecularData(**self.parameters.molecular_data_param)
        # try to load

        do_compute = True
        try:
            import os
            if os.path.exists(self.parameters.filename):
                molecule.load()
                do_compute = False
        except OSError:
            do_compute = True

        if do_compute:
            molecule = self.do_make_molecule(*args, **kwargs)

        molecule.save()
        return molecule

    def do_make_molecule(self, *args, **kwargs):
        """

        Parameters
        ----------
        args
        kwargs

        Returns
        -------

        """
        # integrals need to be passed in base class
        assert ("one_body_integrals" in kwargs)
        assert ("two_body_integrals" in kwargs)
        assert ("nuclear_repulsion" in kwargs)
        assert ("n_orbitals" in kwargs)

        molecule = MolecularData(**self.parameters.molecular_data_param)

        molecule.one_body_integrals = kwargs["one_body_integrals"]
        molecule.two_body_integrals = kwargs["two_body_integrals"]
        molecule.nuclear_repulsion = kwargs["nuclear_repulsion"]
        molecule.n_orbitals = kwargs["n_orbitals"]
        molecule.save()
        return molecule

    @property
    def n_orbitals(self) -> int:
        """ """
        if self.active_space is None:
            return self.molecule.n_orbitals
        else:
            return len(self.active_space.active_orbitals)

    @property
    def n_electrons(self) -> int:
        """ """
        if self.active_space is None:
            return self.molecule.n_electrons
        else:
            return 2 * len(self.active_space.active_reference_orbitals)

    def make_hamiltonian(self, occupied_indices=None, active_indices=None) -> QubitHamiltonian:
        """ """
        if occupied_indices is None and self.active_space is not None:
            occupied_indices = self.active_space.frozen_reference_orbitals
        if active_indices is None and self.active_space is not None:
            active_indices = self.active_space.active_orbitals

        fop = openfermion.transforms.get_fermion_operator(
            self.molecule.get_molecular_hamiltonian(occupied_indices, active_indices))
        return QubitHamiltonian(qubit_hamiltonian=self.transformation(fop))

    def compute_one_body_integrals(self):
        """ """
        if hasattr(self, "molecule"):
            return self.molecule.one_body_integrals

    def compute_two_body_integrals(self):
        """ """
        if hasattr(self, "molecule"):
            return self.molecule.two_body_integrals

    def compute_ccsd_amplitudes(self) -> ClosedShellAmplitudes:
        """ """
        raise Exception("BaseClass Method")

    def prepare_reference(self, *args, **kwargs):
        """

        Returns
        -------
        A tequila circuit object which prepares the reference of this molecule in the chosen transformation
        """

        return prepare_product_state(self.reference_state(*args, **kwargs))

    def make_uccsd_ansatz(self, trotter_steps: int,
                          initial_amplitudes: typing.Union[str, Amplitudes, ClosedShellAmplitudes] = "mp2",
                          include_reference_ansatz=True,
                          parametrized=True,
                          threshold=1.e-8,
                          trotter_parameters: gates.TrotterParameters = None) -> QCircuit:
        """

        Parameters
        ----------
        initial_amplitudes :
            initial amplitudes given as ManyBodyAmplitudes structure or as string
            where 'mp2', 'cc2' or 'ccsd' are possible initializations
        include_reference_ansatz :
            Also do the reference ansatz (prepare closed-shell Hartree-Fock) (Default value = True)
        parametrized :
            Initialize with variables, otherwise with static numbers (Default value = True)
        trotter_steps: int :

        initial_amplitudes: typing.Union[str :

        Amplitudes :

        ClosedShellAmplitudes] :
             (Default value = "mp2")
        trotter_parameters: gates.TrotterParameters :
             (Default value = None)

        Returns
        -------
        type
            Parametrized QCircuit

        """

        if self.n_electrons % 2 != 0:
            raise TequilaException("make_uccsd_ansatz currently only for closed shell systems")

        nocc = self.n_electrons // 2
        nvirt = self.n_orbitals // 2 - nocc

        Uref = QCircuit()
        if include_reference_ansatz:
            Uref = self.prepare_reference()

        amplitudes = initial_amplitudes
        if hasattr(initial_amplitudes, "lower"):
            if initial_amplitudes.lower() == "mp2":
                amplitudes = self.compute_mp2_amplitudes()
            elif initial_amplitudes.lower() == "ccsd":
                amplitudes = self.compute_ccsd_amplitudes()
            else:
                try:
                    amplitudes = self.compute_amplitudes(method=initial_amplitudes.lower())
                except Exception as exc:
                    raise TequilaException(
                        "{}\nDon't know how to initialize \'{}\' amplitudes".format(exc, initial_amplitudes))

        if amplitudes is None:
            amplitudes = ClosedShellAmplitudes(
                tIjAb=numpy.zeros(shape=[nocc, nocc, nvirt, nvirt]),
                tIA=numpy.zeros(shape=[nocc, nvirt]))

        closed_shell = isinstance(amplitudes, ClosedShellAmplitudes)
        generators = []
        variables = []

        if not isinstance(amplitudes, dict):
            amplitudes = amplitudes.make_parameter_dictionary(threshold=threshold)
            amplitudes = dict(sorted(amplitudes.items(), key=lambda x: x[1]))

        for key, t in amplitudes.items():
            assert (len(key) % 2 == 0)
            if not numpy.isclose(t, 0.0, atol=threshold):

                if closed_shell:
                    spin_indices = []
                    if len(key) == 2:
                        spin_indices = [[2 * key[0], 2 * key[1]], [2 * key[0] + 1, 2 * key[1] + 1]]
                        partner = None
                    else:
                        spin_indices.append([2 * key[0] + 1, 2 * key[1] + 1, 2 * key[2], 2 * key[3]])
                        spin_indices.append([2 * key[0], 2 * key[1], 2 * key[2] + 1, 2 * key[3] + 1])
                        if key[0] != key[1] and key[2] != key[3]:
                            spin_indices.append([2 * key[0], 2 * key[1], 2 * key[2], 2 * key[3]])
                            spin_indices.append([2 * key[0] + 1, 2 * key[1] + 1, 2 * key[2] + 1, 2 * key[3] + 1])
                        partner = tuple([key[2], key[1], key[0], key[3]])  # taibj -> tbiaj

                    for idx in spin_indices:
                        idx = [(idx[2 * i], idx[2 * i + 1]) for i in range(len(idx) // 2)]
                        generators.append(self.make_excitation_generator(indices=idx))

                    if parametrized:
                        variables.append(Variable(name=key))  # abab
                        variables.append(Variable(name=key))  # baba
                        if partner is not None and key[0] != key[1] and key[2] != key[3]:
                            variables.append(Variable(name=key) - Variable(partner))  # aaaa
                            variables.append(Variable(name=key) - Variable(partner))  # bbbb
                    else:
                        variables.append(t)
                        variables.append(t)
                        if partner is not None and key[0] != key[1] and key[2] != key[3]:
                            variables.append(t - amplitudes[partner])
                            variables.append(t - amplitudes[partner])
                else:
                    generators.append(self.make_excitation_operator(indices=spin_indices))
                    if parametrized:
                        variables.append(Variable(name=key))
                    else:
                        variables.append(t)

        return Uref + gates.Trotterized(generators=generators, angles=variables, steps=trotter_steps,
                                        parameters=trotter_parameters)

    def compute_amplitudes(self, method: str, *args, **kwargs):
        """
        Compute closed-shell CC amplitudes

        Parameters
        ----------
        method :
            coupled-cluster methods like cc2, ccsd, cc3, ccsd(t)
            Success might depend on backend
            got an extra function for MP2
        *args :

        **kwargs :


        Returns
        -------

        """
        raise TequilaException("compute amplitudes: Needs to be overwridden by backend")

    def compute_mp2_amplitudes(self) -> ClosedShellAmplitudes:
        """

        Compute closed-shell mp2 amplitudes

        .. math::
            t(a,i,b,j) = 0.25 * g(a,i,b,j)/(e(i) + e(j) -a(i) - b(j) )

        :return:

        Parameters
        ----------

        Returns
        -------

        """
        assert self.parameters.closed_shell
        g = self.molecule.two_body_integrals
        fij = self.molecule.orbital_energies
        nocc = self.molecule.n_electrons // 2  # this is never the active space
        ei = fij[:nocc]
        ai = fij[nocc:]
        abgij = g[nocc:, nocc:, :nocc, :nocc]
        amplitudes = abgij * 1.0 / (
                ei.reshape(1, 1, -1, 1) + ei.reshape(1, 1, 1, -1) - ai.reshape(-1, 1, 1, 1) - ai.reshape(1, -1, 1, 1))
        E = 2.0 * numpy.einsum('abij,abij->', amplitudes, abgij) - numpy.einsum('abji,abij', amplitudes, abgij,
                                                                                optimize='greedy')

        self.molecule.mp2_energy = E + self.molecule.hf_energy
        return ClosedShellAmplitudes(tIjAb=numpy.einsum('abij -> ijab', amplitudes, optimize='greedy'))

    def compute_cis_amplitudes(self):
        """
        Compute the CIS amplitudes of the molecule
        """

        @dataclass
        class ResultCIS:
            """ """
            omegas: typing.List[numbers.Real]  # excitation energies [omega0, ...]
            amplitudes: typing.List[ClosedShellAmplitudes]  # corresponding amplitudes [x_{ai}_0, ...]

            def __getitem__(self, item):
                return (self.omegas[item], self.amplitudes[item])

            def __len__(self):
                return len(self.omegas)

        g = self.molecule.two_body_integrals
        fij = self.molecule.orbital_energies

        nocc = self.n_alpha_electrons
        nvirt = self.n_orbitals - nocc

        pairs = []
        for i in range(nocc):
            for a in range(nocc, nocc + nvirt):
                pairs.append((a, i))
        M = numpy.ndarray(shape=[len(pairs), len(pairs)])

        for xx, x in enumerate(pairs):
            eia = fij[x[0]] - fij[x[1]]
            a, i = x
            for yy, y in enumerate(pairs):
                b, j = y
                delta = float(y == x)
                gpart = 2.0 * g[a, i, b, j] - g[a, i, j, b]
                M[xx, yy] = eia * delta + gpart

        omega, xvecs = numpy.linalg.eigh(M)

        # convert amplitudes to ndarray sorted by excitation energy
        nex = len(omega)
        amplitudes = []
        for ex in range(nex):
            t = numpy.ndarray(shape=[nvirt, nocc])
            exvec = xvecs[ex]
            for xx, x in enumerate(pairs):
                a, i = x
                t[a - nocc, i] = exvec[xx]
            amplitudes.append(ClosedShellAmplitudes(tIA=t))

        return ResultCIS(omegas=list(omega), amplitudes=amplitudes)

    def __str__(self) -> str:
        result = str(type(self)) + "\n"
        for k, v in self.parameters.__dict__.items():
            result += "{key:15} : {value:15} \n".format(key=str(k), value=str(v))
        return result
