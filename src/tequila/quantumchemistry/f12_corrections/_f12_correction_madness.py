import tequila as tq
from tequila import TequilaException, TequilaWarning

from tequila.quantumchemistry.qc_base import NBodyTensor
from tequila.quantumchemistry.f12_corrections._f12_correction_base import ExplicitCorrelationCorrection

import numpy
from itertools import product


class ExplicitCorrelationCorrectionMadness(ExplicitCorrelationCorrection):
    """
    Class for computation of explicitly correlated correction, using PNO-orbitals from madness
    """

    def __init__(self, mol=None, rdm1: numpy.ndarray = None, rdm2: numpy.ndarray = None, n_ri: int = None,
                 gamma: int = 1.4, f12_filename: str = "molecule_f12tensor.bin", **kwargs):
        """
        Parameters
        ----------
        mol :
            the molecule given by a QuantumChemistryMadness
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
        """
        # Set up known external info for madness
        if f12_filename.endswith(".bin"):
            try:
                f12_data = numpy.fromfile(f12_filename)
                sd = int(numpy.power(f12_data.size, 0.25))
                assert (sd ** 4 == f12_data.size)
                sds = [sd] * 4
                f12 = f12_data.reshape(sds)
                f12_filename = f12_filename[:-3] + "npy"
                numpy.save(f12_filename, arr=f12)
            except:
                print("Error while trying to read {}!".format(f12_filename))
        elif not f12_filename.endswith(".npy"):
            raise TequilaException
        external_info = {"f12_filename": f12_filename, "ordering": "chem"}
        super().__init__(mol=mol, rdm1=rdm1, rdm2=rdm2, n_ri=n_ri, gamma=gamma,
                         external_info=external_info, **kwargs)

    def setup_tensors_madness(self):
        """
        Setup tensors with integrals provided externally
        Pretty much equivalent to super().setup_tensors_external

        This assumes the f12-integrals r_PQrs = < rs | f_12 | pq > with f_12 = exp(-gamma*r_12),
        while for the correction, we want f_12 = -1/gamma * exp(-gamma*r_12)
        Also, assumes, that f12-integrals are provided in Mulliken ("chem") - ordering

        Returns
        -------
        NBodyTensor
            one-body tensor h, Coulomb tensor g, f_12 tensor r
        """

        h, g, r = super().setup_tensors_external()
        # Assuming a f12-operator of the kind exp(-gamma*r12)
        r.elems *= -1 / self.gamma

        return h, g, r

    def compute(self) -> float:
        """
        Computes universal explicitly correlated correction based on parameters of class instance
        Uses madness-tensors read from an external file

        Returns
        -------
            the explicitly correlated correction
        """
        print("Set up universal f12-correction using external integrals from madness.")
        # Prepare tensors
        h, g, r = self.setup_tensors_madness()
        fock = super().build_fock_operator(h=h, g=g)

        # Compute correction using base class method
        correction = super().compute_correction(g=g, r=r, fock=fock)

        return correction
