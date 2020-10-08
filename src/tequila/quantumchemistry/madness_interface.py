from tequila.quantumchemistry.qc_base import ParametersQC, QuantumChemistryBase, TequilaException, TequilaWarning

import typing
import numpy
import warnings

from dataclasses import dataclass


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
                return "orbital {}, occupied reference orbital {}, energy {} ".format(self.idx_total, self.pno_pair, self.occ)

        def __repr__(self):
            return self.__str__()

    def __init__(self, parameters: ParametersQC,
                 transformation: typing.Union[str, typing.Callable] = None,
                 active_orbitals: list = None,
                 madness_root_dir: str = None,
                 n_pno: int = None,
                 frozen_core = False,
                 n_virt: int = 0,
                 *args,
                 **kwargs):

        self.madness_root_dir = madness_root_dir
        self.n_pno = n_pno
        self.n_virt = n_virt
        self.frozen_core = frozen_core
        self.kwargs=kwargs

        # if no n_pno is given, look for MRA data (default)
        name = "gs"
        if parameters.name != "molecule":
            name = parameters.name

        if n_pno is None:
            h, g = self.read_tensors(name=name)

            if h == "failed" or g == "failed":
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
                raise TequilaException("Could not initialize the madness interface\n"
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
        for name in ["pnoinfo.txt", parameters.name + "_pairinfo.txt"]:
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
            except:
                continue

        if "nuclear_repulsion" not in kwargs:
            kwargs["nuclear_repulsion"] = nuclear_repulsion

        if pairinfo is None:
            raise TequilaException("Pairinfo from madness calculation not found\nPlease provide pnoinfo.txt")

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
                         n_orbitals=n_orbitals,
                         *args,
                         **kwargs)

        orbitals = []
        if pairinfo is not None:
            for i, p in enumerate(pairinfo):
                if active_orbitals is None or i in active_orbitals:
                    orbitals.append(self.OrbitalData(idx_total=i, idx=len(orbitals), pno_pair=p, occ=occinfo[i]))
        else:
            raise TequilaException("No pairinfo given")
        self.orbitals = tuple(orbitals)

        # print warning if read data does not match expectations
        if n_pno is not None:
            nrefs = len(self.get_reference_orbitals())
            if n_pno+nrefs+n_virt != self.n_orbitals:
                warnings.warn(
                    "read in data has {} pnos/virtuals, but n_pno and n_virt where set to {} and {}".format(self.n_orbitals - nrefs, n_pno, n_virt), TequilaWarning)

    def run_madness(self, *args, **kwargs):
        executable = "pno_integrals"
        if self.madness_root_dir is not None:
            executable = "{}/src/apps/pno/pno_integrals".format(self.madness_root_dir)
        self.make_madness_input(n_pno=self.n_pno, frozen_core=self.frozen_core, n_virt=self.n_virt, *args, **kwargs)
        import subprocess
        import time
        start = time.time()
        print("Starting madness calculation with executable: ", executable)
        with open("{}_pno.out".format(self.parameters.filename), "w") as logfile:
            madout = subprocess.call([executable], stdout=logfile)
        print("finished after {}s".format(time.time() - start))
        return madout

    def read_tensors(self, name="gs", filetype=".npy"):
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

    def make_pno_upccgsd_ansatz(self, include_singles: bool = True, generalized=False, **kwargs):
        indices_d = []
        indices_s = []
        refs = self.get_reference_orbitals()
        print("refs=", refs)
        for i in self.get_reference_orbitals():
            for a in self.get_pno_indices(i=i, j=i):
                u = (2 * i.idx, 2 * a.idx)
                d = (2 * i.idx + 1, 2 * a.idx + 1)
                indices_d.append((u, d))
                indices_s.append((u))
                indices_s.append((d))
            if generalized:
                for a in self.get_pnos_indices(i, i):
                    for b in self.get_pnos_indices(i, i):
                        if b.idx_total <= a.idx_total:
                            continue
                        u = (2 * a.idx, 2 * b.idx)
                        d = (2 * a.idx + 1, 2 * b.idx + 1)
                        indices_d.append(u, d)
                        indices_s.append(u)
                        indices_s.append(d)

        indices = indices_d
        if include_singles:
            indices += indices_s

        print("indidces=", indices)
        return self.make_upccgsd_ansatz(indices=indices, **kwargs)

    def make_madness_input(self, n_pno, n_virt=0, frozen_core=False, filename="input", *args, **kwargs):
        if n_pno is None:
            raise TequilaException("Can't write madness input without n_pnos")
        data = {}
        data["dft"] = {"xc": "hf", "k": 7, "econv": "1.e-4", "dconv": "1.e-4"}
        data["pno"] = {"maxrank": n_pno, "f12": "false", "thresh":1.e-4}
        if not frozen_core:
            data["pno"]["freeze"] = 0
        data["pnoint"] = {"n_pno": n_pno, "n_virt": n_virt}
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
                print(self.parameters.get_geometry_string(), file=f)
                print("end", file=f)

        return data

    def convert_madness_output_from_bin_to_npy(self, name="gs"):
        try:
            g_data = numpy.fromfile("gs_gtensor.bin")
            sd = int(numpy.power(g_data.size, 0.25))
            assert (sd ** 4 == g_data.size)
            sds = [sd] * 4
            g = g_data.reshape(sds)
            numpy.save("{}_gtensor.npy".format(name), arr=g)
        except:
            g = "failed"

        try:
            h_data = numpy.fromfile("gs_htensor.bin")
            sd = int(numpy.sqrt(h_data.size))
            assert (sd ** 2 == h_data.size)
            sds = [sd] * 2
            h = h_data.reshape(sds)
            numpy.save("{}_htensor.npy".format(name), arr=h)
        except:
            h = "failed"

        return h, g

    def __str__(self):
        info=super().__str__()
        info+="{key:15} :\n".format(key="MRA Orbitals")
        for orb in self.orbitals:
            info+="{}\n".format(orb)
        return info

    def __repr__(self):
        return self.__str__()
