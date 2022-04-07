import os
from dataclasses import dataclass

import numpy

from tequila import BitString, QCircuit, TequilaException
from tequila.circuit import gates

@dataclass
class ActiveSpaceData:
    """
    Small dataclass to keep the overview in active spaces
    Class is used internally
    """
    active_orbitals: list  # active orbitals (spatial, c1)
    reference_orbitals: list  # reference orbitals (spatial, c1)

    def __str__(self):
        result = "Active Space Data:\n"
        result += "{key:15} : {value:15} \n".format(key="active_orbitals", value=str(self.active_orbitals))
        result += "{key:15} : {value:15} \n".format(key="reference_orbitals",
                                                    value=str(self.reference_orbitals))
        return result

    @property
    def frozen_reference_orbitals(self):
        return [i for i in self.reference_orbitals if i not in self.active_orbitals]

    @property
    def active_reference_orbitals(self):
        return [i for i in self.reference_orbitals if i in self.active_orbitals]


class FermionicGateImpl(gates.QubitExcitationImpl):
    """
    Small helper class for Fermionic Excictation Gates
    Mainly so that "FermionicGate is displayed when circuits are printed
    """

    def __init__(self, generator, p0, transformation, indices=None, *args, **kwargs):
        super().__init__(generator=generator, target=generator.qubits, p0=p0, *args, **kwargs)
        self._name = "FermionicExcitation"
        self.transformation = transformation
        self.indices = indices

    def compile(self, *args, **kwargs):
        if self.transformation.lower() == "jordanwigner" and self.i_am_spin_paired_pair_excitation():
            target = []
            for x in self.indices:
                for y in x:
                    target.append(y)
            return gates.QubitExcitation(target=target, angle=self.parameter, control=self.control)
        else:
            return gates.Trotterized(generator=self.generator, control=self.control, angle=self.parameter, steps=1)

    def __str(self):
        if self.indices is not None:
            return "FermionicExcitation({})".format(str(self.indices))
        return "FermionicExcitation"

    def __repr__(self):
        return self.__str__()

    def i_am_spin_paired_pair_excitation(self):
        if len(self.indices) != 2: return False
        if self.indices[0][0]//2 != self.indices[1][0]//2: return False
        if self.indices[0][1]//2 != self.indices[1][1]//2: return False
        return True



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
    basis_set: str = None  # Quantum chemistry basis set
    geometry: str = None  # geometry of the underlying molecule (units: Angstrom!),
    # this can be a filename leading to an .xyz file or the geometry given as a string
    description: str = ""
    multiplicity: int = 1
    charge: int = 0
    name: str = None

    @property
    def n_electrons(self, *args, **kwargs):
        return self.get_nuc_charge() - self.charge

    def get_nuc_charge(self):
        return sum(self.get_atom_number(name=atom) for atom in self.get_atoms())

    def get_atom_number(self, name):
        atom_numbers = {"h": 1, "he": 2, "li": 3, "be": 4, "b": 5, "c": 6, "n": 7, "o": 8, "f": 9, "ne": 10, "na": 11,
                        "mg": 12, "al": 13, "si": 14, "ph": 15, "s": 16, "cl": 17, "ar": 18}
        if name.lower() in atom_numbers:
            return atom_numbers[name.lower()]
        try:
            import periodictable as pt
            atom = name.lower()
            atom[0] = atom[0].upper()
            element = pt.elements.symbol(atom)
            return element.number()
        except:
            raise TequilaException(
                "can not assign atomic number to element {}\npip install periodictable will fix it".format(atom))

    def get_atoms(self):
        return [x[0] for x in self.get_geometry()]

    def __post_init__(self, *args, **kwargs):

        if self.name is None and self.geometry is None:
            raise TequilaException(
                "no geometry or name given to molecule\nprovide geometry=filename.xyz or geometry=`h 0.0 0.0 0.0\\n...`\nor name=whatever with file whatever.xyz being present")
        # auto naming
        if self.name is None:
            if ".xyz" in self.geometry:
                self.name = self.geometry.split(".xyz")[0]
                if self.description is None:
                    coord, description = self.read_xyz_from_file()
                    self.description = description
            else:
                atoms = self.get_atoms()
                atom_names = sorted(list(set(atoms)), key=lambda x: self.get_atom_number(x), reverse=True)
                if self.name is None:
                    drop_ones = lambda x: "" if x == 1 else x
                    self.name = "".join(["{}{}".format(x, drop_ones(atoms.count(x))) for x in atom_names])
        self.name = self.name.lower()

        if self.geometry is None:
            self.geometry = self.name + ".xyz"

        if ".xyz" in self.geometry and not os.path.isfile(self.geometry):
            raise TequilaException("could not find file for molecular coordinates {}".format(self.geometry))

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
        this convenience function does the naming
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
            a string specifying a mol. structure. E.g. geometry="h 0.0 0.0 0.0\n h 0.0 0.0 1.0"

        Returns
        -------
        type
            A list with the correct format for openfermion E.g return [ ['h',[0.0,0.0,0.0], [..]]

        """
        result = []
        # Remove blank lines
        lines = [l for l in geometry.split("\n") if l]

        for line in lines:
            words = line.split()

            # Pad coordinates
            if len(words) < 4:
                words += [0.0] * (4 - len(words))

            try:
                tmp = (ParametersQC.format_element_name(words[0]),
                       (float(words[1]), float(words[2]), float(words[3])))
                result.append(tmp)
            except ValueError:
                print("get_geometry list unknown line:\n ", line, "\n proceed with caution!")
        return result

    def get_geometry_string(self) -> str:
        """returns the geometry as a string
        :return: geometry string

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
        which is then reformatted as a list usable as input for openfermion
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
    """
    Helper Class for clasical amplitudes
    used internally
    """
    tIjAb: numpy.ndarray = None
    tIA: numpy.ndarray = None

    def make_parameter_dictionary(self, threshold=1.e-8, screening=True):
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
                if not numpy.isclose(value, 0.0, atol=threshold) or not screening:
                    variables[(nocc + A, I, nocc + B, J)] = value
        if self.tIA is not None:
            nocc = self.tIA.shape[0]
            for (I, A), value, in numpy.ndenumerate(self.tIA):
                if not numpy.isclose(value, 0.0, atol=threshold) or not screening:
                    variables[(A + nocc, I)] = value
        return dict(sorted(variables.items(), key=lambda x: numpy.abs(x[1]), reverse=True))


@dataclass
class Amplitudes:
    """
    Helper class for classical Coupled-Cluster Amplitudes
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


class NBodyTensor:
    """ Convenience class for handling N-body tensors """

    class Ordering:
        """
        Convenience to keep track of aliases in odering names for two body integrals
        i.e. Mulliken/Chem/1122
             Dirac/Phys/1212
             openfermion/1221
        """

        def __init__(self, scheme):
            if hasattr(scheme, "_scheme"):
                scheme = scheme._scheme
            elif hasattr(scheme, "scheme"):
                scheme = scheme.scheme
            self._scheme = self.assign_scheme(scheme)

        def assign_scheme(self, scheme):
            if scheme is None:
                return "chem"
            else:
                scheme = str(scheme)

            if scheme.lower() in ["mulliken", "chem", "c", "1122"]:
                return "chem"
            elif scheme.lower() in ["dirac", "phys", "p", "1212"]:
                return "phys"
            elif scheme.lower() in ["openfermion", "of", "o", "1221"]:
                return "of"
            else:
                raise TequilaException(
                    "Unknown two-body tensor scheme {}. Supported are dirac, mulliken, and openfermion".format(scheme))

        def is_phys(self):
            return self._scheme == "phys"

        def is_chem(self):
            return self._scheme == "chem"

        def is_of(self):
            return self._scheme == "of"

    def __init__(self, elems: numpy.ndarray = None, active_indices: list = None, ordering: str = None,
                 size_full: int = None):
        """
        Parameters
        ----------
        elems: Tensor data as numpy array
        active_indices: List of active indices in total ordering
        ordering: Ordering scheme for two body tensors
        "dirac" or "phys": <12|g|12>
            .. math::
                g_{pqrs} = \\int d1 d2 p(1)q(2) g(1,2) r(1)s(2)
        "mulliken" or "chem": (11|g|22)
            .. math::
                g_{pqrs} = \\int d1 d2 p(1)r(2) g(1,2) q(1)s(2)
        "openfermion":
            .. math:: [12|g|21]
                g_{gqprs} = \\int d1 d2 p(1)q(2) g(1,2) s(1)r(2)

        size_full
        """

        # Set elements
        self.elems = elems
        # Active indices only as list of indices (e.g. spatial orbital indices), not as a dictionary of irreducible
        # representations
        if active_indices is not None:
            self.active_indices = active_indices
        self._passive_indices = None
        self._full_indices = None
        self._indices_set: bool = False

        # Determine order of tensor
        # Assume, that tensor is entered in desired shape, not as flat array.
        self.order = len(self.elems.shape)
        # Can use size_full < self.elems.shape[0] -> 'full' space is to be considered a subspace as well
        if size_full is None:
            self._size_full = self.elems.shape[0]
        else:
            self._size_full = size_full
        # 2-body tensors (<=> order 4) currently allow reordering
        if self.order == 4:
            self.ordering = self.Ordering(ordering)
        else:
            if ordering is not None:
                raise Exception("Ordering only implemented for tensors of order 4 / 2-body tensors.")
            self.ordering = None

    def sub_lists(self, idx_lists: list = None) -> numpy.ndarray:
        """
        Get subspace of tensor by a set of index lists
        according to hPQ.sub_lists(idx_lists=[p, q]) = [hPQ for P in p and Q in q]

        This essentially is an implementation of a non-contiguous slicing using numpy.take

        Parameters
        ----------
            idx_lists :
                List of lists, each defining the desired subspace per axis
                Size needs to match order of tensor, and lists successively correspond to axis=0,1,2,...,N

        Returns
        -------
            out :
                Sliced tensor as numpy.ndarray
        """
        # Check if index list has correct size
        if len(idx_lists) != self.order:
            raise Exception("Need to pass an index list for each dimension!" +
                            " Length of idx_lists needs to match order of tensor.")

        # Perform slicing via numpy.take
        out = self.elems
        for ax in range(self.order):
            if idx_lists[ax] is not None:  # None means, we want the full space in this direction
                out = numpy.take(out, idx_lists[ax], axis=ax)

        return out

    def set_index_lists(self):
        """ Set passive and full index lists based on class inputs """
        tmp_size = self._size_full
        if self._size_full is None:
            tmp_size = self.elems.shape[0]

        self._passive_indices = [i for i in range(tmp_size)
                                 if i not in self.active_indices]
        self._full_indices = [i for i in range(tmp_size)]

    def sub_str(self, name: str) -> numpy.ndarray:
        """
        Get subspace of tensor by a string
        Currently is able to resolve an active space, named 'a', full space 'f', and the complement 'p' = 'f' - 'a'.
        Full space in this context may also be smaller than actual tensor dimension.

        The specification of active space in this context only allows to pick a set from a list of orbitals, and
        is not able to resolve an active space from irreducible representations.

        Example for one-body tensor:
        hPQ.sub_lists(name='ap') = [hPQ for P in active_indices and Q in _passive_indices]

        Parameters
        ----------
            name :
                String specifying the desired subspace, elements need to be a (active), f (full), p (full - active)

        Returns
        -------
            out :
                Sliced tensor as numpy.ndarray
        """
        if not self._indices_set:
            self.set_index_lists()
            self._indices_set = True

        if name is None:
            raise Exception("No name specified.")
        if len(name) != self.order:
            raise Exception("Name does not match order of the tensor.")
        if self.active_indices is None:
            raise Exception("Need to set an active space in order to call this function.")

        idx_lists = []
        # Parse name as string of space indices
        for char in name:
            if char.lower() == 'a':
                idx_lists.append(self.active_indices)
            elif char.lower() == 'p':
                idx_lists.append(self._passive_indices)
            elif char.lower() == 'f':
                if self._size_full is None:
                    idx_lists.append(None)
                else:
                    idx_lists.append(self._full_indices)
            else:
                raise Exception("Need to specify a valid letter (a,p,f).")

        out = self.sub_lists(idx_lists)

        return out

    def reorder(self, to: str = 'of'):
        """
        Function to reorder tensors according to some convention.

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
        """
        if self.order != 4:
            raise Exception('Reordering currently only implemented for two-body tensors.')

        to = self.Ordering(to)

        if self.ordering == to:
            return self
        elif self.ordering.is_chem():
            if to.is_of():
                self.elems = numpy.einsum("psqr -> pqrs", self.elems, optimize='greedy')
            elif to.is_phys():
                self.elems = numpy.einsum("prqs -> pqrs", self.elems, optimize='greedy')
        elif self.ordering.is_of():
            if to.is_chem():
                self.elems = numpy.einsum("pqrs -> psqr", self.elems, optimize='greedy')
            elif to.is_phys():
                self.elems = numpy.einsum("pqrs -> pqsr", self.elems, optimize='greedy')
        elif self.ordering.is_phys():
            if to.is_chem():
                self.elems = numpy.einsum("pqrs -> prqs", self.elems, optimize='greedy')
            elif to.is_of():
                self.elems = numpy.einsum("pqsr -> pqrs", self.elems, optimize='greedy')

        return self