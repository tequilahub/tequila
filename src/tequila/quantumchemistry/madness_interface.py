from tequila.quantumchemistry.qc_base import QuantumChemistryBase, TequilaException, TequilaWarning, \
    QCircuit, gates
from tequila.quantumchemistry import ParametersQC, NBodyTensor
from tequila import ExpectationValue
from .chemistry_tools import OrbitalData
import typing
import numpy
import warnings
import os
import shutil


# Examples how to initialize the madness backend
# tq.Molecule(geometry="...") will initialize a molecule with n_qubits=n_electrons (n_electrons//2 HF orbitals + n_electrons//2 PNOs)
# tq.Molecule(geometry="...", n_pno="read") will read in files (name is auto-generated from geometry)
# tq.Molecule(geometry="...", name="X", n_pno="read") will read in files X_htensor.npy, X_gtensor.npy, X_pnoinfo.txt
# tq.Molecule(geometry="...", name="X", datadir="asd/Y/", n_pno="read") reads in files from directory asd/Y/
# control madness input sections with dictionaries
# tq.Molecule(geometry="...", pno={"maxrank":10, "freeze":0}, dft={"k":9, "L":25.0})
# compute more orbitals
# tq.Molecule(geometry="...", n_pno=10) # computes 10 PNOs additional to the occupied HF orbitals


class TequilaMadnessException(TequilaException):
    def __str__(self):
        return "Error in madness backend:" + self.message


class QuantumChemistryMadness(QuantumChemistryBase):

    @staticmethod
    def find_executable(madness_root_dir=None):
        executable = shutil.which("pno_integrals")
        if madness_root_dir is None:
            madness_root_dir = str(os.environ.get("MAD_ROOT_DIR"))
        if executable is None and madness_root_dir is not None:
            executable = shutil.which("{}/src/apps/pno/pno_integrals".format(madness_root_dir))
        return executable

    def plot2cube(self, orbital, filename=None, *args, **kwargs):
        """
        plot orbitals to cube file (needs madtequila backend installed)
        Parameters
        ----------
        method: orbital, the orbital index (starting from 0 on the active orbitals)
                if you want to plot frozen orbitals you can hand in a Tequila Orbital structure with idx_total defined
        filename: name of the cubefile (default: mra_orbital_X.cube where X is the total index of the active orbital)
        args: further arguments for plot2cube
        kwargs further keyword arguments for plot2cube

        see here for more https://github.com/kottmanj/madness/tree/tequila/src/apps/plot
        """

        plot2cube = shutil.which("plot2cube")
        if plot2cube is None:
            raise TequilaMadnessException(
                "can't plot to cube file. Couldn't find plot2cube executable.\n\nTry installing\n\t conda install madtequila -c kottmann\nand assure the version is >2.3")

        if hasattr(orbital,"idx"):
            idx = orbital.idx
        else:
            idx = self.orbitals[orbital].idx_total

        callist = [plot2cube, "file=mra_orbital_{}".format(idx)]

        if filename is not None:
            callist.append("outfile={}".format(filename))
        for k, v in kwargs.items():
            callist.append("{}={}".format(k, v))
        for k in args:
            callist.append("{}".format(k))

        import subprocess
        try:
            with open("plot2cube_{}.log".format(orbital), "w") as logfile:
                subprocess.call(callist, stdout=logfile)
        except:
            print("plotting failed ....")
            print("see plot2cube_{}.log".format(orbital))

    def __init__(self, parameters: ParametersQC,
                 transformation: typing.Union[str, typing.Callable] = None,
                 active_orbitals: list = "auto",
                 executable: str = None,
                 n_pno: int = None,
                 n_virt: int = 0,
                 keep_mad_files=False,
                 datadir=None,
                 *args,
                 **kwargs):

        self.datadir = datadir

        # see if MAD_ROOT_DIR is defined
        self.madness_root_dir = os.environ.get("MAD_ROOT_DIR")
        # see if the pno_integrals executable can be found
        if executable is None:
            executable = self.find_executable()
            if executable is None and self.madness_root_dir is not None:
                warnings.warn("MAD_ROOT_DIR={} found\nbut couldn't find executable".format(self.madness_root_dir),
                              TequilaWarning)


        else:
            executable = shutil.which(executable)

        self.executable = executable
        self.n_pno = n_pno
        self.n_virt = n_virt
        self.kwargs = kwargs

        # if no n_pno is given, look for MRA data (default)
        name = parameters.name

        # try to read in data in the following cases
        # - no executable found
        # - executable found but read in explicitly demanded through n_pno="read"
        if (n_pno is None and executable is None) or (hasattr(n_pno, "lower") and n_pno.lower() == "read"):
            h, g = self.read_tensors(name=name, datadir=datadir)
            n_pno = None
        else:
            h = "failed"
            g = "failed"

        if "failed" in h or "failed" in g:
            status = "found {}_htensor.npy={}\n".format(name, "failed" not in h)
            status += "found {}_gtensor.npy={}\n".format(name, "failed" not in g)
            try:
                # try to run madness
                self.parameters = parameters
                status += "madness="
                madness_status = self.run_madness(*args, **kwargs)
                if int(madness_status) != 0:
                    warnings.warn("MADNESS did not terminate as expected! status = {}".format(status), TequilaWarning)
                status += str(madness_status) + "\n"
            except Exception as E:
                status += str(E) + "\n"

            # will read the binary files, convert them and save them with the right name
            h, g, pinfo = self.convert_madness_output_from_bin_to_npy(name=name, datadir=datadir)
            status += "found {}_htensor.npy={}\n".format(name, "failed" not in h)
            status += "found {}_gtensor.npy={}\n".format(name, "failed" not in g)
            status += "found {}_pnoinfo.txt={}\n".format(name, "failed" not in pinfo)
            status += "h_tensor report:\n"
            status += str(h)
            status += "g_tensor report:\n"
            status += str(g)
            status += "pnoinfo report:\n"
            status += str(pinfo)

            solution = "Solution 1: Assuming precomputed files are available:\n    provide {name}_gtensor.npy, {name}_htensor.npy and {name}_pnoinfo.txt\n    and call the Molecule constructor with n_pno='read' keyword \n\nSolution 2: Try installing with conda\n    conda install madtequila -c kottmann\n\nSolution 3: Install from source\n    follow instructions on github.com/kottmanj/madness".format(
                name=name)
            if self.executable is not None:
                solution = "madness executable was found, but calculation did not succeed, check {name}_pno_integrals.out for clues".format(
                    name=name)

            if "failed" in h or "failed" in g:
                raise TequilaMadnessException("Could not initialize the madness interface\n"
                                              "Status report is\n"
                                              "{status}\n\n".format(status=status) + solution)
        # get additional information from madness file
        nuclear_repulsion = 0.0
        pairinfo = None
        occinfo = None
        path = parameters.name
        if datadir is not None:
            path = "{}/{}".format(datadir, path)
        for name in [path + "_pnoinfo.txt", parameters.name + "_pnoinfo.txt", "pnoinfo.txt"]:
            try:
                with open(name, "r") as f:
                    for line in f.readlines():
                        if "nuclear_repulsion" in line:
                            nuclear_repulsion = float(line.split("=")[1])
                        elif "pairinfo" in line:
                            pairinfo = line.split("=")[1].split(",")
                            pairinfo = [tuple([int(i) for i in x.split(".")]) for x in pairinfo]
                        elif "occinfo" in line:
                            occinfo = line.split("=")[1].split(",")
                            occinfo = [float(x) for x in occinfo]

                if pairinfo is not None:
                    break
            except:
                continue

        if pairinfo is None:
            raise TequilaMadnessException("Pairinfo from madness calculation not found\nPlease provide pnoinfo.txt")

        n_orbitals_total = h.shape[0]
        if "n_orbitals" in kwargs:
            # this would be the active orbitals
            kwargs.pop("n_orbitals")

        assert h.shape[1] == n_orbitals_total
        assert sum(g.shape) == 4 * n_orbitals_total
        assert len(g.shape) == 4
        assert len(h.shape) == 2

        g = NBodyTensor(elems=g, ordering='mulliken')

        orbitals = []
        if pairinfo is not None:
            orbitals = [OrbitalData(idx_total=i, idx=i, pair=p, occ=occinfo[i]) for i, p in
                        enumerate(pairinfo)]
            reference_orbitals = [x for x in orbitals if x.occ == 2.0]
            if active_orbitals == "auto":
                not_active = [i for i in reference_orbitals if
                              sum([1 for x in orbitals if i.idx_total in x.pair]) < 2]
                active_orbitals = [x.idx_total for x in orbitals if x not in not_active]

            if active_orbitals is not None:
                i = 0
                for x in orbitals:
                    if x.idx_total in active_orbitals:
                        orbitals[x.idx_total].idx = i
                        i += 1
                    else:
                        orbitals[x.idx_total].idx = None
        else:
            raise TequilaMadnessException("No pairinfo given: madness interface needs a file moleculename_pnoinfo.txt")

        # convert to indices only
        # active space data will be set in baseclass constructor
        reference_orbitals = [x.idx_total for x in reference_orbitals]
        super().__init__(parameters=parameters,
                         transformation=transformation,
                         active_orbitals=active_orbitals,
                         one_body_integrals=h,
                         two_body_integrals=g,
                         nuclear_repulsion=nuclear_repulsion,
                         n_orbitals=n_orbitals_total,
                         orbitals=orbitals,
                         reference_orbitals=reference_orbitals,
                         *args,
                         **kwargs)

        # print warning if read data does not match expectations
        if n_pno is not None:
            nrefs = len(self.reference_orbitals)
            if n_pno + nrefs + n_virt != self.n_orbitals:
                warnings.warn(
                    "read in data has {} pnos/virtuals, but n_pno and n_virt where set to {} and {}".format(
                        self.n_orbitals - nrefs, n_pno, n_virt), TequilaWarning)

        # delete *.bin files and pnoinfo.txt form madness calculation
        if not keep_mad_files:
            self.cleanup(warn=False, delete_all_files=False)

    def cleanup(self, warn=False, delete_all_files=False):

        filenames = ["pnoinfo.txt", "molecule_htensor.bin", "molecule.gtensor.bin"]
        if delete_all_files:
            filenames = ["{}_htensor.npy".format(self.parameters.name), "{}_gtensor.npy".format(self.parameters.name),
                         "{}_pnoinfo.txt".format(self.parameters.name),
                         "{}_pno_integrals.out".format(self.parameters.name)]
        for filename in filenames:
            if os.path.exists(filename):
                if warn:
                    warnings.warn("Found file {} from previous calculation ... deleting it".format(filename),
                                  TequilaWarning)
                os.remove(filename)

    def run_madness(self, *args, **kwargs):
        if self.executable is None:
            return "\n\n----> pno_integrals executable not found <----\n\n"
        self.write_madness_input(n_pno=self.n_pno, n_virt=self.n_virt, *args, **kwargs)

        # prevent reading in old files
        self.cleanup(warn=True, delete_all_files=True)

        import subprocess
        import time
        start = time.time()
        filename = "{}_pno_integrals.out".format(self.parameters.name)
        print("Starting madness calculation with executable: ", self.executable)
        print("output redirected to {} logfile".format(filename))
        with open(filename, "w") as logfile:
            madout = subprocess.call([self.executable], stdout=logfile)
        print("finished after {}s".format(time.time() - start))

        os.rename("pnoinfo.txt", "{}_pnoinfo.txt".format(self.parameters.name))

        return madout

    def read_tensors(self, name="molecule", filetype="npy", datadir=None):
        """
        Try to read files "name_htensor.npy" and "name_gtensor.npy"
        """

        path = name
        if datadir is not None:
            path = "{}/{}".format(datadir, name)

        try:
            h = numpy.load("{}_htensor.{}".format(path, filetype))
        except:
            h = "failed"

        try:
            g = numpy.load("{}_gtensor.{}".format(path, filetype))
        except:
            g = "failed"

        return h, g

    def get_pair_orbitals(self, i: OrbitalData, j: OrbitalData,
                          exclude: typing.List[OrbitalData] = None):
        if isinstance(i, int):
            i = self.orbitals[i]
        if isinstance(j, int):
            j = self.orbitals[j]
        if isinstance(exclude, int):
            exclude = [self.orbitals[exclude]]
        if exclude is None or isinstance(exclude, OrbitalData):
            exclude = [exclude]
        return [x for x in self.orbitals if (i.idx_total, j.idx_total) == x.pair and x not in exclude]

    def get_virtual_orbitals(self):
        return [x for x in self.orbitals if len(x.pair) == 1 and x.pair[0] < 0]

    def compute_energy(self, method, *args, **kwargs):
        """
        Call classical methods over PySCF (needs to be installed) or
        use as a shortcut to calculate quantum energies (see make_upccgsd_ansatz)

        Parameters
        ----------
        method: method name
                classical: HF, MP2, CCSD, CCSD(T), FCI
                quantum: SPA-GASD (SPA can be dropped as well as letters in GASD)
                examples: GSD is the same as UpCCGSD, SPA alone is equivalent to SPA-D
                see make_upccgsd_ansatz of the this class for more information
        args
        kwargs

        Returns
        -------

        """

        if any([x in method.upper() for x in ["U", "SPA", "PNO", "HCB"]]):
            # simulate entirely in HCB representation if no singles are involved
            if "S" not in method.upper().split("-")[-1] and "HCB" not in method.upper():
                method = "HCB-" + method
            U = self.make_upccgsd_ansatz(name=method)
            if "hcb" in method.lower():
                H = self.make_hardcore_boson_hamiltonian()
            else:
                H = self.make_hamiltonian()
            E = ExpectationValue(H=H, U=U)
            from tequila import minimize
            return minimize(objective=E, *args, **kwargs).energy
        else:
            return super().compute_energy(method=method, *args, **kwargs)

    def local_qubit_map(self, hcb=False, up_then_down=False):
        # re-arrange orbitals to result in more local circuits
        # does not make the circuit more local, but rather will show locality better in pictures
        # transform circuits and Hamiltonians with this map
        # H = H.map_qubits(qubit_map), U = U.map_qubits(qubit_map)
        # hcb: same for the harcore_boson representation
        ordered_qubits = []
        pairs = [i for i in range(self.n_electrons // 2)]
        for i in pairs:
            pnos = [i] + [a.idx for a in self.get_pair_orbitals(i=i, j=i, exclude=i)]
            if hcb:
                up = [i for i in pnos]
                ordered_qubits += up
            else:
                if up_then_down:
                    up = [self.transformation.up(i) for i in pnos]
                    down = [self.transformation.down(i) for i in pnos]
                    ordered_qubits += up + down
                else:
                    for i in pnos:
                        ordered_qubits.append(self.transformation.up(i))
                        ordered_qubits.append(self.transformation.down(i))

        qubit_map = {x: i for i, x in enumerate(ordered_qubits)}
        return qubit_map

    def make_spa_ansatz(self, label=None, hcb=False):
        """
        Shortcut for convenience
        Parameters
        ----------
        label:
           label for the angles
        hcb:
           if True the circuit will not map from HCB to JW (or other encodings that might be supported in the future)
        Returns
        -------
        Default SPA ansatz (equivalent to PNO-UpCCD with madness PNOs)

        """
        name = "SPA-UpCCD"
        if hcb and "HCB" not in name.upper():
            name = "HCB-" + name
        return self.make_upccgsd_ansatz(name=name, label=label)

    def make_upccgsd_ansatz(self, name="UpCCGSD", label=None, direct_compiling=None, order=None, neglect_z=None,
                            hcb_optimization=None, include_reference=True, *args, **kwargs):
        """
        Overwriting baseclass to allow names like : PNO-UpCCD etc
        Parameters
        ----------
        label: label the variables of the ansatz ( variables will be labelled (indices, X, (label, layer) witch X=D/S)
        direct_compiling: Directly compile the first layer (works only for transformation that implement the hcb_to_me function)
        name: ansatz name (SPA-UpCCD or SPA-D, SPA-UpCCGD or SPA-GD, SPA-UpCCGSD or SPA-GSD, UpCCGSD or GSD ..., in general {HCB}-{SPA}-{Excitations}
              if HCB is included in name: do not map from hard-core Boson to qubit encoding of this molecule
              if SPA is included in name: Use the separable pair ansatz (excitations will be restricted to the PNO structure of the surrogate model)
              Excitations: can be "S" (use only singles), "D" (use only doubles), "GSD" (generalized singles and doubles), "GASD" (approximate singles, neglecting Z terms in JW)
        neglect_z: neglect all Z terms in singles excitations generators
        order: repetition of layers, can be given over the name as well, the order needs to be the first in the name then (i.e. 2-UpCCGSD, 2-SPA-GSD, etc)
        args
        kwargs

        Returns
        -------

        """
        # check if the used qubit encoding has a hcb transformation
        have_hcb_trafo = self.transformation.hcb_to_me() is not None
        name = name.upper()

        # Default Method
        if "SPA" in name and name.split("-")[-1] in ["SPA", "HCB"]:
            name += "-D"

        excitations = name.split("-")[-1]

        if "HCB" in name and "S" in excitations:
            raise TequilaMadnessException("name={}, HCB + Singles can't be realized".format(name))

        if "HCB" in name and "D" not in excitations:
            raise warnings.warn("name={}, HCB without Doubles has no result ".format(name), TequilaWarning)

        if "S" not in excitations and "D" not in excitations:
            raise warnings.warn("name={}, neither singles nor doubles requested".format(name), TequilaWarning)

        if "T" in excitations or "Q" in excitations:
            raise warnings.warn("name={}, only singles and doubles supported".format(name), TequilaWarning)

        if (have_hcb_trafo and "D" in excitations or "HCB" in name) and include_reference and hcb_optimization is None:
            hcb_optimization = True

        if hcb_optimization and direct_compiling is None:
            direct_compiling = True

        if ("A" in excitations) and neglect_z is None:
            neglect_z = True
            # spin adaption does not work with z neglected
            if "spin_adapt_singles" not in kwargs:
                kwargs["spin_adapt_singles"] = False
        else:
            neglect_z = False
            # switch on by default
            if "spin_adapt_singles" not in kwargs:
                kwargs["spin_adapt_singles"] = True

        if direct_compiling and not have_hcb_trafo and not "HCB" in name:
            raise TequilaMadnessException(
                "direct_compiling={} demanded but no hcb_to_me in transformation={}\ntry transformation=\'ReorderedJordanWigner\' ".format(
                    direct_compiling, self.transformation))

        name = name.upper()
        if order is None:
            try:
                if "-" in name:
                    order = int(name.split("-")[0])
                else:
                    order = 1
            except:
                order = 1

        # first layer
        U = QCircuit()
        if hcb_optimization:
            if "D" in excitations:
                U = self.make_hardcore_boson_pno_upccd_ansatz(include_reference=include_reference,
                                                              direct_compiling=direct_compiling,
                                                              label=(label, 0))
            elif include_reference:
                U = self.prepare_hardcore_boson_reference()

            indices0 = [k.name[0] for k in U.extract_variables()]
            indices1 = self.make_upccgsd_indices(label=label, name=name, exclude=indices0, *args, **kwargs)
            if "D" in excitations:
                U += self.make_hardcore_boson_upccgd_layer(indices=indices1, label=(label, 0), *args, **kwargs)
            indices = indices0 + indices1
            if "HCB" not in name and len(U.gates) > 0:
                U = self.hcb_to_me(U=U)
            else:
                assert "S" not in excitations
            if "S" in excitations:
                U += self.make_upccgsd_singles(indices=indices, label=(label, 0), neglect_z=neglect_z, *args, **kwargs)
        else:
            indices = self.make_upccgsd_indices(label=(label, 0), name=name, *args, **kwargs)
            if include_reference:
                U = self.prepare_reference()
            U += self.make_upccgsd_layer(indices=indices, include_singles="S" in excitations,
                                         include_doubles="D" in excitations, label=(label, 0), neglect_z=neglect_z,
                                         *args, **kwargs)

        if order > 1:
            for layer in range(1, order):
                indices = self.make_upccgsd_indices(label=(label, layer), name=name, *args, **kwargs)
                if "HCB" in name:
                    U += self.make_hardcore_boson_upccgd_layer(indices=indices, label=(label, layer), *args, **kwargs)
                else:
                    U += self.make_upccgsd_layer(indices=indices, include_singles="S" in excitations,
                                                 include_doubles="D" in excitations, label=(label, layer),
                                                 neglect_z=neglect_z, *args, **kwargs)
        return U

    def make_hardcore_boson_pno_upccd_ansatz(self, pairs=None, label=None, include_reference=True,
                                             direct_compiling=False):
        if pairs is None:
            pairs = [x for x in self.reference_orbitals]
        U = QCircuit()
        if direct_compiling:
            if not include_reference:
                raise Exception("HCB_PNO_UPCCD: Direct compiling needs reference included")
            for x in pairs:
                U += gates.X(x.idx)
                c = [None, x.idx]
                for a in self.get_pair_orbitals(i=x, j=x):
                    if a == x:
                        continue
                    idx = self.format_excitation_indices([(x.idx, a.idx)])
                    U += gates.Ry(angle=(idx, "D", label), target=a.idx, control=c[0])
                    U += gates.X(target=c[1], control=a.idx)
                    if hasattr(direct_compiling, "lower") and direct_compiling.lower() == "ladder":
                        c = [a.idx, a.idx]
                    else:
                        c = [x.idx, x.idx]

            alpha_map = {k.idx: self.transformation.up(k.idx) for k in self.orbitals}
            U = U.map_qubits(alpha_map)
        else:
            for x in pairs:
                if include_reference:
                    U += gates.X(self.transformation.up(x.idx))
                for a in self.get_pair_orbitals(i=x, j=x):
                    if x == a:
                        continue
                    idx = self.format_excitation_indices([(x.idx, a.idx)])
                    U += self.make_hardcore_boson_excitation_gate(indices=idx, angle=(idx, "D", label))

        return U

    def make_upccgsd_indices(self, label=None, name="UpCCGD", exclude: list = None, *args, **kwargs):
        """
        :param label: label the angles
        :param generalized: if true the complement to UpCCGD is created (otherwise UpCCD)
        :param exclude: list of indices to exclude
        :return: All gates missing between PNO-UpCCD and UpCC(G)D
        """

        if exclude is None:
            exclude = []
        name = name.upper()

        indices = []

        # HF-X -- PNO-XX indices
        for i in self.reference_orbitals:
            for a in self.get_pair_orbitals(i=i, j=i, exclude=i):
                idx = self.format_excitation_indices([(i.idx, a.idx)])
                if idx not in exclude and idx not in indices:
                    indices.append(idx)

        if "G" in name:
            for i in self.reference_orbitals:
                for a in self.get_pair_orbitals(i=i, j=i, exclude=i):
                    for b in self.get_pair_orbitals(i=i, j=i, exclude=i):
                        if a.idx <= b.idx:
                            continue

                        idx = self.format_excitation_indices([(a.idx, b.idx)])
                        if idx not in exclude and idx not in indices:
                            indices.append(idx)

        if "PNO" in name or "SPA" in name:
            return indices

        virtuals = [i for i in self.orbitals if len(i.pair) == 2 and i.occ != 2.0]
        virtuals += self.get_virtual_orbitals()  # this is usually empty
        # HF-X -- PNO-XY/PNO-YY indices
        for i in self.reference_orbitals:
            for a in virtuals:
                idx = self.format_excitation_indices([(i.idx, a.idx)])
                if idx not in exclude and idx not in indices:
                    indices.append(idx)
        # HF-HF and PNO-PNO indices
        if "G" in name:
            for i in self.reference_orbitals:
                for j in self.reference_orbitals:
                    if i.idx <= j.idx:
                        continue
                    idx = self.format_excitation_indices([(i.idx, j.idx)])
                    if idx not in exclude and idx not in indices:
                        indices.append(idx)
            for a in virtuals:
                for b in virtuals:
                    if a.idx_total <= b.idx_total:
                        continue
                    idx = self.format_excitation_indices([(a.idx, b.idx)])
                    if idx not in exclude and idx not in indices:
                        indices.append(idx)
        return indices

    def make_pno_upccgsd_ansatz(self, generalized=False, include_offdiagonals=False,
                                **kwargs):
        indices = []
        refs = self.reference_orbitals
        for i in self.reference_orbitals:
            for a in self.get_pair_orbitals(i=i, j=i, exclude=i):
                indices.append((i.idx, a.idx))
            if generalized:
                for a in self.get_pair_orbitals(i=i, j=i, exclude=i):
                    for b in self.get_pair_orbitals(i=i, j=i, exclude=i):
                        if b.idx_total <= a.idx_total:
                            continue
                        indices.append((i.idx, a.idx))

        if include_offdiagonals:
            for i in self.reference_orbitals:
                for j in self.reference_orbitals:
                    pairs_ij = self.get_pair_orbitals(i=i, j=j, exclude=[i, j])
                    if i.idx <= j.idx:
                        continue
                    for a in pairs_ij:
                        indices.append((j.idx, a.idx))

                    if generalized:
                        for a in pairs_ij:
                            for b in pairs_ij:
                                if a.idx <= b.idx:
                                    continue
                                indices.append((a.idx, b.idx))

        return self.make_upccgsd_ansatz(indices=indices, **kwargs)

    def write_madness_input(self, n_pno=None, n_virt=0, filename="input", maxrank=None,
                            n_orbitals=None, *args, **kwargs):

        if n_pno is not None and n_orbitals is not None:
            raise TequilaMadnessException(
                "n_pno={} and n_orbitals={} given ... please pick one".format(n_pno, n_orbitals))

        n_electrons = self.parameters.n_electrons
        if self.parameters.frozen_core:
            # only count active electrons (will not compute pnos for frozen pairs)
            n_core_electrons = self.parameters.get_number_of_core_electrons()
            n_electrons -= n_core_electrons

        n_pairs = n_electrons // 2
        if n_orbitals is None:
            n_orbitals = n_electrons  # minimal correlated (each active pair will have one virtual)

        if n_pno is None:
            n_pno = n_orbitals - n_pairs

        if maxrank is None:
            # need at least maxrank=1, otherwise no PNOs are computed
            # this was a bug in <=v1.8.5 
            maxrank = max(1, int(numpy.ceil(n_pno // n_pairs)))

        if maxrank <= 0:
            warnings.warn(
                "maxrank={} in tequila madness backend! No PNOs will be computed. Set the value when initializing the Molecule as tq.Molecule(..., pno={\"maxrank\":1, ...})".format(
                    maxrank), TequilaWarning)

        data = {}
        if self.parameters.multiplicity != 1:
            raise TequilaMadnessException(
                "Currently only closed shell supported for MRA-PNO-MP2, you demanded multiplicity={} for the surrogate".format(
                    self.parameters.multiplicity))
        data["dft"] = {"charge": self.parameters.charge, "xc": "hf", "k": 7, "econv": 1.e-4, "dconv": 5.e-4,
                       "localize": "boys",
                       "ncf": "( none , 1.0 )"}
        data["pno"] = {"maxrank": maxrank, "f12": "false", "thresh": 1.e-4, "diagonal": True}
        if not self.parameters.frozen_core:
            data["pno"]["freeze"] = 0
        data["pnoint"] = {"n_pno": n_pno, "n_virt": n_virt, "orthog": "symmetric"}
        data["plot"] = {}
        data["f12"] = {}
        for key in data.keys():
            if key in kwargs:
                data[key] = {**data[key], **kwargs[key]}

        if filename is not None:
            with open(filename, "w") as f:
                for k1, v1 in data.items():
                    print(k1, file=f)
                    for k2, v2 in v1.items():
                        print("{} {}".format(k2, v2), file=f)
                    print("end\n", file=f)

                print("geometry", file=f)
                print("units angstrom", file=f)
                print("eprec 1.e-6", file=f)
                for line in self.parameters.get_geometry_string().split("\n"):
                    line = line.strip()
                    if line != "":
                        print(line, file=f)
                print("end", file=f)

        return data

    def convert_madness_output_from_bin_to_npy(self, name, datadir=None):
        path = name
        if datadir is not None:
            # if the datadir does not exist then tequila will crash
            try:
                import os
                if not os.path.exists(datadir):
                    os.makedirs(datadir)
            except Exception as E:
                warnings.warn("tried to create datadir={} and caught\n{}".format(datadir, str(E)), TequilaWarning)

            path = "{}/{}".format(datadir, name)
        try:
            g_data = numpy.fromfile("molecule_gtensor.bin".format())
            sd = int(numpy.power(g_data.size, 0.25))
            assert (sd ** 4 == g_data.size)
            sds = [sd] * 4
            g = g_data.reshape(sds)
            numpy.save("{}_gtensor.npy".format(path), arr=g)
        except Exception as E:
            g = "failed\n{}\n".format(str(E))

        try:
            h_data = numpy.fromfile("molecule_htensor.bin")
            sd = int(numpy.sqrt(h_data.size))
            assert (sd ** 2 == h_data.size)
            sds = [sd] * 2
            h = h_data.reshape(sds)
            numpy.save("{}_htensor.npy".format(path), arr=h)
        except Exception as E:
            h = "failed\n{}\n".format(str(E))

        try:
            with open("{}_pnoinfo.txt".format(name), "r") as f1:
                pnoinfo = f1.read().strip()
        except Exception as E:
            pnoinfo = "failed\n{}\n".format(str(E))

        if datadir is not None:
            try:
                with open("{}_pnoinfo.txt".format(name), "r") as f1, open("{}_pnoinfo.txt".format(path), "w") as f2:
                    f2.write(f1.read().strip())
            except Exception as E:
                pnoinfo = "failed\n{}\n".format(str(E))
            try:
                with open("{}_pno_integrals.out".format(name), "r") as f1, open("{}_pno_integrals.out".format(path),
                                                                                "w") as f2:
                    f2.write(f1.read().strip())
            except Exception as E:
                pass

        return h, g, pnoinfo

    def perturbative_f12_correction(self, rdm1: numpy.ndarray = None, rdm2: numpy.ndarray = None, n_ri: int = None,
                                    f12_filename: str = "molecule_f12tensor.bin", **kwargs) -> float:
        """
        Computes the spin-free [2]_R12 correction, needing only the 1- and 2-RDM of a reference method
        Requires either 1-RDM, 2-RDM or information to compute them in kwargs

        Parameters
        ----------
        rdm1 :
            1-electron reduced density matrix
        rdm2 :
            2-electron reduced density matrix
        gamma :
            f12-exponent, for a correlation factor f_12 = -1/gamma * exp[-gamma*r_12]
        n_ri :
            dimensionality of RI-basis; if None, then the maximum available via tensors / basis-set is used
        f12_filename :
            when using madness_interface, <q|h|p> and <rs|1/r_12|pq> already available;
            need to provide f12-tensor <rs|f_12|pq> as ".bin" from madness or ".npy", assuming Mulliken ordering
        kwargs :
            e.g. RDM-information via {"U": QCircuit, "variables": optimal angles}, needs to be passed if rdm1,rdm2 not
            yet computed

        Returns
        -------
            the f12 correction for the energy
        """
        from .f12_corrections._f12_correction_madness import ExplicitCorrelationCorrectionMadness
        correction = ExplicitCorrelationCorrectionMadness(mol=self, rdm1=rdm1, rdm2=rdm2, n_ri=n_ri,
                                                          f12_filename=f12_filename, **kwargs)

        return correction.compute()

    def __str__(self):
        path = self.parameters.name
        if hasattr(self, "datadir") and self.datadir is not None:
            path = "{}/{}".format(self.datadir, path)
        info = super().__str__()
        info += "\n"
        info += "{:15} : {}\n".format("executable", self.executable)
        info += "{:15} : {}\n".format("htensor", "{}_htensor.npy".format(path))
        info += "{:15} : {}\n".format("gtensor", "{}_gtensor.npy".format(path))

        return info

    def __repr__(self):
        return self.__str__()
