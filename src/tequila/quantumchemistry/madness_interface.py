from tequila.quantumchemistry.qc_base import ParametersQC, QuantumChemistryBase, TequilaException, TequilaWarning

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
                warnings.warn("MAD_ROOT_DIR={} found\nbut couldn't find executable".format(self.madness_root_dir), TequilaWarning)


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
                warnings.warn("Could not find data for {}. Looking for binary files from potential madness calculation".format(name), TequilaWarning)
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
        for name in [parameters.name + "_pnoinfo.txt", "pnoinfo.txt"]:
            try:
                filecontent = ""
                with open(name, "r") as f:
                    for line in f.readlines():
                        filecontent += line
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
        super().__init__(parameters=parameters,
                         transformation=transformation,
                         active_orbitals=active_orbitals,
                         one_body_integrals=h,
                         two_body_integrals=g,
                         nuclear_repulsion = nuclear_repulsion,
                         n_orbitals=n_orbitals,
                         *args,
                         **kwargs)

        orbitals = []
        if pairinfo is not None:
            for i, p in enumerate(pairinfo):
                if active_orbitals is None or i in active_orbitals:
                    orbitals.append(self.OrbitalData(idx_total=i, idx=len(orbitals), pno_pair=p, occ=occinfo[i]))
        else:
            raise TequilaMadnessException("No pairinfo given")
        self.orbitals = tuple(orbitals)

        # print warning if read data does not match expectations
        if n_pno is not None:
            nrefs = len(self.get_reference_orbitals())
            if n_pno + nrefs + n_virt != self.n_orbitals:
                warnings.warn(
                    "read in data has {} pnos/virtuals, but n_pno and n_virt where set to {} and {}".format(
                        self.n_orbitals - nrefs, n_pno, n_virt), TequilaWarning)

    def run_madness(self, *args, **kwargs):
        if self.executable is None:
            return "pno_integrals executable not found\n" \
                   "pass over executable keyword or export MAD_ROOT_DIR to system environment"
        self.write_madness_input(n_pno=self.n_pno, frozen_core=self.frozen_core, n_virt=self.n_virt, *args, **kwargs)
        import subprocess
        import time
        start = time.time()
        filename="{}_pno_integrals.out".format(self.parameters.name)
        print("Starting madness calculation with executable: ", self.executable)
        print("output redirected to {} logfile".format(filename))
        with open(filename, "w") as logfile:
            madout = subprocess.call([self.executable], stdout=logfile)
        print("finished after {}s".format(time.time() - start))
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

    def make_pno_upccgsd_ansatz(self, include_singles: bool = True, generalized=False, include_offdiagonals=False,
                                **kwargs):
        indices_d = []
        indices_s = []
        refs = self.get_reference_orbitals()
        for i in self.get_reference_orbitals():
            for a in self.get_pno_indices(i=i, j=i):
                u = (2 * i.idx, 2 * a.idx)
                d = (2 * i.idx + 1, 2 * a.idx + 1)
                indices_d.append((u, d))
                indices_s.append((u))
                indices_s.append((d))
            if generalized:
                for a in self.get_pno_indices(i, i):
                    for b in self.get_pno_indices(i, i):
                        if b.idx_total <= a.idx_total:
                            continue
                        u = (2 * a.idx, 2 * b.idx)
                        d = (2 * a.idx + 1, 2 * b.idx + 1)
                        indices_d.append((u, d))
                        indices_s.append((u))
                        indices_s.append((d))

        if include_offdiagonals:
            for i in self.get_reference_orbitals():
                for j in self.get_reference_orbitals():
                    if i.idx <= j.idx:
                        continue
                    for a in self.get_pno_indices(i, j):
                        ui = (2 * i.idx, 2 * a.idx)
                        di = (2 * i.idx + 1, 2 * a.idx + 1)
                        uj = (2 * j.idx, 2 * a.idx)
                        dj = (2 * j.idx + 1, 2 * a.idx + 1)
                        indices_d.append((ui, dj))
                        indices_d.append((uj, di))
                        indices_s.append((ui))
                        indices_s.append((uj))
                        indices_s.append((di))
                        indices_s.append((dj))

                    if generalized:
                        for a in self.get_pno_indices(i, j):
                            for b in self.get_pno_indices(i, j):
                                if a.idx <= b.idx:
                                    continue
                                u = (2 * a.idx, 2 * b.idx)
                                d = (2 * a.idx + 1, 2 * b.idx + 1)
                                indices_d.append((u, d))
                                indices_s.append((u))
                                indices_s.append((d))

        indices = indices_d
        if include_singles:
            indices += indices_s

        return self.make_upccgsd_ansatz(indices=indices, **kwargs)

    def write_madness_input(self, n_pno, n_virt=0, frozen_core=False, filename="input", *args, **kwargs):
        if n_pno is None:
            raise TequilaMadnessException("Can't write madness input without n_pno keyword!")
        data = {}
        if self.parameters.multiplicity != 1:
            raise TequilaMadnessException("Currently only closed shell supported for MRA-PNO-MP2, you demanded multiplicity={} for the surrogate".format(self.parameters.multiplicity))
        data["dft"] = {"charge":self.parameters.charge, "xc": "hf", "k": 7, "econv": 1.e-4, "dconv": 3.e-4, "ncf": "( none , 1.0 )"}
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
