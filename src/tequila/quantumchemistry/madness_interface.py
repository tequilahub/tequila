from tequila.quantumchemistry.qc_base import ParametersQC, QuantumChemistryBase, TequilaException, TequilaWarning, \
    QCircuit, gates
from tequila import ExpectationValue, PauliString, QubitHamiltonian, simulate

import typing
import numpy
import warnings
import os
import shutil

from dataclasses import dataclass


class TequilaMadnessException(TequilaException):
    def __str__(self):
        return "Error in madness backend:" + self.message


class QuantumChemistryMadness(QuantumChemistryBase):
    @dataclass
    class OrbitalData:
        idx: int = None  # active index
        idx_total: int = None  # total index
        pno_pair: tuple = None  # pno origin if tuple of len 2, otherwise occupied or virtual orbital
        occ: float = None  # original MP2 occupation number, or orbital energies

        def __str__(self):
            if len(self.pno_pair) == 2:
                return "orbital {}, pno from pair {}, MP2 occ {} ".format(self.idx_total, self.pno_pair, self.occ)
            elif self.pno_pair[0] < 0:
                return "orbital {}, virtual orbital {}, energy {} ".format(self.idx_total, self.pno_pair, self.occ)
            else:
                return "orbital {}, occupied reference orbital {}, energy {} ".format(self.idx_total, self.pno_pair,
                                                                                      self.occ)

        def __repr__(self):
            return self.__str__()

        def __hash__(self):
            return hash((self.idx, self.idx_total, self.pno_pair, self.occ))

    @staticmethod
    def find_executable(madness_root_dir=None):
        executable = shutil.which("pno_integrals")
        if madness_root_dir is None:
            madness_root_dir = str(os.environ.get("MAD_ROOT_DIR"))
        if executable is None and madness_root_dir is not None:
            executable = shutil.which("{}/src/apps/pno/pno_integrals".format(madness_root_dir))
        return executable

    def __init__(self, parameters: ParametersQC,
                 transformation: typing.Union[str, typing.Callable] = None,
                 active_orbitals: list = None,
                 executable: str = None,
                 n_pno: int = None,
                 frozen_core=False,
                 n_virt: int = 0,
                 *args,
                 **kwargs):

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
        self.frozen_core = frozen_core
        self.kwargs = kwargs

        # if no n_pno is given, look for MRA data (default)
        name = parameters.name

        if n_pno is None:
            h, g = self.read_tensors(name=name)

            if h == "failed" or g == "failed":
                warnings.warn(
                    "Could not find data for {}. Looking for binary files from potential madness calculation".format(
                        name), TequilaWarning)
                # try if madness was run manually without conversion before
                h, g = self.convert_madness_output_from_bin_to_npy(name=name)
        else:
            h = "failed"
            g = "failed"

        if h == "failed" or g == "failed":
            status = "found {}_htensor.npy={}\n".format(name, h != "failed")
            status += "found {}_gtensor.npy={}\n".format(name, h != "failed")
            try:
                # try to run madness
                self.parameters = parameters
                status += "madness="
                madness_status = self.run_madness(*args, **kwargs)
                if int(madness_status) != 0:
                    warnings.warn("MADNESS did not terminate as expected! status = {}".format(status), TequilaWarning)
                status += str(madness_status)
            except Exception as E:
                status += "madness_run={}\n".format(str(E))

            # will read the binary files, convert them and save them with the right name
            h, g = self.convert_madness_output_from_bin_to_npy(name=name)
            status += "found {}_htensor.npy={}\n".format(name, h != "failed")
            status += "found {}_gtensor.npy={}\n".format(name, h != "failed")
            if h == "failed" or g == "failed":
                raise TequilaMadnessException("Could not initialize the madness interface\n"
                                              "Status report is\n"
                                              "{status}\n"
                                              "either provide {name}_gtensor.npy and {name}_htensor.npy files\n"
                                              "or provide the number of pnos over by giving the n_pnos keyword to run madness\n"
                                              "in order for madness to run you need to make sure that the pno_integrals executable can be found in your environment\n"
                                              "alternatively you can provide the path to the madness_root_dir: the directory where you compiled madness\n".format(
                    name=name, status=status))

        # get additional information from madness file
        nuclear_repulsion = 0.0
        pairinfo = None
        occinfo = None
        for name in [parameters.name + "_pnoinfo.txt"]:
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
                if name == "pnoinfo.txt":
                    with open("pnoinfo.txt", "r") as f1, open(parameters.name + "_pnoinfo.txt", "w") as f2:
                        f2.write(f1.read().strip())
                if pairinfo is not None:
                    break
            except:
                continue

        if pairinfo is None:
            raise TequilaMadnessException("Pairinfo from madness calculation not found\nPlease provide pnoinfo.txt")

        n_orbitals = h.shape[0]
        assert h.shape[1] == n_orbitals
        assert sum(g.shape) == 4 * n_orbitals
        assert len(g.shape) == 4
        assert len(h.shape) == 2

        # openfermion conventions
        g = numpy.einsum("psqr", g, optimize='optimize')

        orbitals = []
        if pairinfo is not None:
            orbitals = [self.OrbitalData(idx_total=i, idx=i, pno_pair=p, occ=occinfo[i]) for i, p in
                        enumerate(pairinfo)]
            if active_orbitals == "auto":
                reference_orbitals = [x for x in orbitals if len(x.pno_pair) == 1]
                not_active = [i for i in reference_orbitals if
                              sum([1 for x in orbitals if i.idx_total in x.pno_pair]) < 2]
                active_orbitals = [x.idx_total for x in orbitals if x not in not_active]

            if active_orbitals is not None:
                orbitals = [x for x in orbitals if x.idx_total in active_orbitals]
                for i, x in enumerate(orbitals):
                    orbitals[i].idx = i
        else:
            raise TequilaMadnessException("No pairinfo given")
        self.orbitals = tuple(orbitals)

        super().__init__(parameters=parameters,
                         transformation=transformation,
                         active_orbitals=active_orbitals,
                         one_body_integrals=h,
                         two_body_integrals=g,
                         nuclear_repulsion=nuclear_repulsion,
                         n_orbitals=n_orbitals,
                         *args,
                         **kwargs)

        # print warning if read data does not match expectations
        if n_pno is not None:
            nrefs = len(self.get_reference_orbitals())
            if n_pno + nrefs + n_virt != self.n_orbitals:
                warnings.warn(
                    "read in data has {} pnos/virtuals, but n_pno and n_virt where set to {} and {}".format(
                        self.n_orbitals - nrefs, n_pno, n_virt), TequilaWarning)

        # delete *.bin files and pnoinfo.txt form madness calculation
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
            return "pno_integrals executable not found\n" \
                   "pass over executable keyword or export MAD_ROOT_DIR to system environment"
        self.write_madness_input(n_pno=self.n_pno, frozen_core=self.frozen_core, n_virt=self.n_virt, *args, **kwargs)

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

    def read_tensors(self, name="molecule", filetype="npy"):
        """
        Try to read files "name_htensor.npy" and "name_gtensor.npy"
        """

        try:
            h = numpy.load("{}_htensor.{}".format(name, filetype))
        except:
            h = "failed"

        try:
            g = numpy.load("{}_gtensor.{}".format(name, filetype))
        except:
            g = "failed"

        return h, g

    def get_pno_indices(self, i: OrbitalData, j: OrbitalData):
        if isinstance(i, int):
            i = self.orbitals[i]
        if isinstance(j, int):
            j = self.orbitals[j]
        return [x for x in self.orbitals if (i.idx_total, j.idx_total) == x.pno_pair]

    def get_reference_orbital(self, i):
        return [x for x in self.orbitals if (i) == x.pno_pair]

    def get_reference_orbitals(self):
        return [x for x in self.orbitals if len(x.pno_pair) == 1 and x.pno_pair[0] >= 0]

    def get_virtual_orbitals(self):
        return [x for x in self.orbitals if len(x.pno_pair) == 1 and x.pno_pair[0] < 0]

    def local_qubit_map(self, hcb=False):
        # re-arrange orbitals to result in more local circuits
        # does not make the circuit more local, but rather will show locality better in pictures
        # transform circuits and Hamiltonians with this map
        # H = H.map_qubits(qubit_map), U = U.map_qubits(qubit_map)
        # hcb: same for the harcore_boson representation
        ordered_qubits = []
        pairs = [i for i in range(self.n_electrons // 2)]
        for i in pairs:
            pnos = [i] + [a.idx for a in self.get_pno_indices(i=i, j=i)]
            if hcb:
                up = [i for i in pnos]
                ordered_qubits += up
            else:
                up = [self.transformation.up(i) for i in pnos]
                down = [self.transformation.down(i) for i in pnos]
                ordered_qubits += up + down

        qubit_map = {x: i for i, x in enumerate(ordered_qubits)}
        return qubit_map

    def make_upccgsd_ansatz(self, name="UpCCGSD", label=None, direct_compiling=None, order=None, *args, **kwargs):
        """
        Overwriting baseclass to allow names like : PNO-UpCCD etc
        Parameters
        ----------
        label: label the variables of the ansatz ( variables will be labelled (indices, X, (label, layer) witch X=D/S)
        direct_compiling: Directly compile the first layer (works only for transformation that implement the hcb_to_me function)
        name: ansatz name (PNO-UpCCD, PNO-UpCCGD, PNO-UpCCGSD, UpCCGSD ...
        order: repetition of layers
        args
        kwargs

        Returns
        -------

        """
        # check if the used qubit encoding has a hcb transformation
        have_hcb_trafo = self.transformation.hcb_to_me() is not None

        if "HCB" in name and "S" in name:
            raise Exception("name={}, HCB + Singles can't be realized".format(name))

        if ("HCB" in name or have_hcb_trafo) and direct_compiling is None:
            direct_compiling = True

        if direct_compiling and not have_hcb_trafo and not "HCB" in name:
            raise TequilaMadnessException(
                "direct_compiling={} demanded but no hcb_to_me in transformation={}\ntry transformation=\'ReorderedJordanWigner\' ".format(
                    direct_compiling, self.transformation))

        name = name.upper()

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
        if have_hcb_trafo or "HCB" in name:
            U = self.make_hardcore_boson_pno_upccd_ansatz(include_reference=True, direct_compiling=direct_compiling,
                                                          label=(label, 0))
            indices0 = [k.name[0] for k in U.extract_variables()]
            indices1 = self.make_upccgsd_indices(label=label, name=name, exclude=indices0, *args, **kwargs)
            U += self.make_hardcore_boson_upccgd_layer(indices=indices1, label=(label, 0), *args, **kwargs)
            indices = indices0 + indices1
            if "HCB" not in name:
                U = self.hcb_to_me(U=U)
            else:
                assert "S" not in name
            if "S" in name:
                U += self.make_upccgsd_singles(indices=indices, label=(label, 0), *args, **kwargs)
        else:
            indices = self.make_upccgsd_indices(label=(label, 0), name=name, *args, **kwargs)
            U = self.prepare_reference()
            U += self.make_upccgsd_layer(indices=indices, include_singles="S" in name, label=(label, 0), *args,
                                         **kwargs)

        if order > 1:
            for layer in range(1, order):
                indices = self.make_upccgsd_indices(label=(label, layer), name=name, *args, **kwargs)
                if "HCB" in name:
                    U += self.make_hardcore_boson_upccgd_layer(indices=indices, label=(label, layer), *args, **kwargs)
                else:
                    U += self.make_upccgsd_layer(indices=indices, include_singles="S" in name, label=(label, layer),
                                                 *args, **kwargs)
        return U

    def make_hardcore_boson_pno_upccd_ansatz(self, pairs=None, label=None, include_reference=True,
                                             direct_compiling=False):
        if pairs is None:
            pairs = [x for x in self.get_reference_orbitals()] #[i for i in range(self.n_electrons // 2)]
        U = QCircuit()
        if direct_compiling:
            if not include_reference:
                raise Exception("HCB_PNO_UPCCD: Direct compiling needs reference included")
            for x in pairs:
                U += gates.X(x.idx)
                c = [None, x.idx]
                for a in self.get_pno_indices(i=x, j=x):
                    idx = self.format_excitation_indices([(x.idx, a.idx)])
                    U += gates.Ry(angle=(idx, "D", label), target=a.idx, control=c[0])
                    U += gates.X(target=c[1], control=a.idx)
                    if hasattr(direct_compiling, "lower") and direct_compiling.lower() == "ladder":
                        c = [a.idx, a.idx]
                    else:
                        c = [x.idx, x.idx]
        else:
            for x in pairs:
                if include_reference:
                    U += gates.X(x.idx)
                for a in self.get_pno_indices(i=x, j=x):
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
        for i in self.get_reference_orbitals():
            for a in self.get_pno_indices(i=i, j=i):
                idx = self.format_excitation_indices([(i.idx, a.idx)])
                if idx not in exclude and idx not in indices:
                    indices.append(idx)
        if "G" in name:
            for i in self.get_reference_orbitals():
                for a in self.get_pno_indices(i=i, j=i):
                    for b in self.get_pno_indices(i=i, j=i):
                        if a.idx <= b.idx:
                            continue
                        idx = self.format_excitation_indices([(a.idx, b.idx)])
                        if idx not in exclude and idx not in indices:
                            indices.append(idx)

        if "PNO" in name:
            return indices

        virtuals = [i for i in self.orbitals if len(i.pno_pair) == 2]
        virtuals += self.get_virtual_orbitals()  # this is usually empty

        # HF-X -- PNO-XY/PNO-YY indices
        for i in self.get_reference_orbitals():
            for a in virtuals:
                idx = self.format_excitation_indices([(i.idx, a.idx)])
                if idx not in exclude and idx not in indices:
                    indices.append(idx)
        # HF-HF and PNO-PNO indices
        if "G" in name:
            for i in self.get_reference_orbitals():
                for j in self.get_reference_orbitals():
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

    def make_separated_objective(self, pairs=None, label=None, neglect_coupling=False, direct_compiling=True):
        if pairs is None:
            pairs = [x for x in self.get_reference_orbitals()]

        H = self.make_hardcore_boson_hamiltonian()

        def assign_pair(k):
            orbital = self.orbitals[k]
            assert len(orbital.pno_pair) == 1 or orbital.pno_pair[0] == orbital.pno_pair[1]
            for pair in pairs:
                if pair.pno_pair[0] == orbital.pno_pair[0]:
                    return pair

        objective = 0.0
        implemented_ops = {}
        for ps in H.paulistrings:
            c = float(ps.coeff.real)
            if numpy.isclose(c, 0.0):
                continue
            ops = {}
            for k, v in ps.items():
                pair = assign_pair(k)
                if pair not in pairs:
                    continue
                if pair in ops:
                    ops[pair][k] = v
                else:
                    ops[pair] = {k: v}
            if len(ops) == 0:
                objective += c
            elif len(ops) == 1:
                assert len(ops) > 0
                if neglect_coupling and len(ops) == 2:
                    continue
                tmp = c
                for pair, ps1 in ops.items():
                    Up = self.make_hardcore_boson_pno_upccd_ansatz(pairs=[pair], label=(label,0),
                                                                   direct_compiling=direct_compiling)
                    print(Up)
                    ps = PauliString(data=ps1)
                    Hp = QubitHamiltonian.from_paulistrings([ps])
                    Ep = ExpectationValue(H=Hp, U=Up)
                    if len(Ep.extract_variables()) == 0:
                        Ep = numpy.float64(simulate(Ep))
                    if pair not in implemented_ops.keys():
                        implemented_ops.update({pair: {str(ps): Ep}})
                    else:
                        if str(ps) not in implemented_ops[pair].keys():
                            implemented_ops[pair].update({str(ps): Ep})
                        else:
                            Ep = implemented_ops[pair][str(ps)]
                    tmp *= Ep
                objective += tmp
            elif len(ops) == 2:
                assert len(ops) > 0
                if neglect_coupling and len(ops) == 2:
                    continue
                tmp = c
                for pair, ps1 in ops.items():
                    Up = self.make_hardcore_boson_pno_upccd_ansatz(pairs=[pair], label=(label,0),
                                                                   direct_compiling=direct_compiling)
                    ps = PauliString(data=ps1)
                    Hp = QubitHamiltonian.from_paulistrings([ps])
                    Ep = ExpectationValue(H=Hp, U=Up)
                    if len(Ep.extract_variables()) == 0:
                        Ep = numpy.float64(simulate(Ep))
                    if pair not in implemented_ops.keys():
                        implemented_ops.update({pair: {str(ps): Ep}})
                    else:
                        if str(ps) not in implemented_ops[pair].keys():
                            implemented_ops[pair].update({str(ps): Ep})
                        else:
                            Ep = implemented_ops[pair][str(ps)]
                    tmp *= Ep
                objective += tmp
            else:
                raise Exception("don't know how to handle paulistring: {}".format(ps))

        return objective

    def make_pno_upccgsd_ansatz(self, generalized=False, include_offdiagonals=False,
                                **kwargs):
        indices = []
        refs = self.get_reference_orbitals()
        for i in self.get_reference_orbitals():
            for a in self.get_pno_indices(i=i, j=i):
                indices.append((i.idx, a.idx))
            if generalized:
                for a in self.get_pno_indices(i, i):
                    for b in self.get_pno_indices(i, i):
                        if b.idx_total <= a.idx_total:
                            continue
                        indices.append((i.idx, a.idx))

        if include_offdiagonals:
            for i in self.get_reference_orbitals():
                for j in self.get_reference_orbitals():
                    if i.idx <= j.idx:
                        continue
                    for a in self.get_pno_indices(i, j):
                        indices.append((j.idx, a.idx))

                    if generalized:
                        for a in self.get_pno_indices(i, j):
                            for b in self.get_pno_indices(i, j):
                                if a.idx <= b.idx:
                                    continue
                                indices.append((a.idx, b.idx))

        return self.make_upccgsd_ansatz(indices=indices, **kwargs)

    def write_madness_input(self, n_pno, n_virt=0, frozen_core=False, filename="input", *args, **kwargs):
        if n_pno is None:
            raise TequilaMadnessException("Can't write madness input without n_pno keyword!")
        data = {}
        if self.parameters.multiplicity != 1:
            raise TequilaMadnessException(
                "Currently only closed shell supported for MRA-PNO-MP2, you demanded multiplicity={} for the surrogate".format(
                    self.parameters.multiplicity))
        data["dft"] = {"charge": self.parameters.charge, "xc": "hf", "k": 7, "econv": 1.e-4, "dconv": 3.e-4,
                       "ncf": "( none , 1.0 )"}
        data["pno"] = {"maxrank": n_pno, "f12": "false", "thresh": 1.e-4}
        if not frozen_core:
            data["pno"]["freeze"] = 0
        data["pnoint"] = {"n_pno": n_pno, "n_virt": n_virt, "orthog": "cholesky"}
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

    def convert_madness_output_from_bin_to_npy(self, name):
        try:
            g_data = numpy.fromfile("molecule_gtensor.bin".format())
            sd = int(numpy.power(g_data.size, 0.25))
            assert (sd ** 4 == g_data.size)
            sds = [sd] * 4
            g = g_data.reshape(sds)
            numpy.save("{}_gtensor.npy".format(name), arr=g)
        except:
            g = "failed"

        try:
            h_data = numpy.fromfile("molecule_htensor.bin")
            sd = int(numpy.sqrt(h_data.size))
            assert (sd ** 2 == h_data.size)
            sds = [sd] * 2
            h = h_data.reshape(sds)
            numpy.save("{}_htensor.npy".format(name), arr=h)
        except:
            h = "failed"

        return h, g

    def __str__(self):
        info = super().__str__()
        info += "{key:15} :\n".format(key="MRA Orbitals")
        for orb in self.orbitals:
            info += "{}\n".format(orb)
        info += "\n"
        info += "{:15} : {}\n".format("executable", self.executable)
        info += "{:15} : {}\n".format("htensor", "{}_htensor.npy".format(self.parameters.name))
        info += "{:15} : {}\n".format("gtensor", "{}_gtensor.npy".format(self.parameters.name))

        return info

    def __repr__(self):
        return self.__str__()
