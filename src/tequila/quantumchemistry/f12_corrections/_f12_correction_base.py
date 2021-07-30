import tequila as tq
from tequila import TequilaException, TequilaWarning

from tequila.quantumchemistry.qc_base import NBodyTensor

import numpy
from itertools import product


class ExplicitCorrelationCorrection:
    """
    Computes universal explicitly correlated correction
    following Kong, L. & Valeev, E. F. SF-[2] R12: A spin-adapted explicitly correlated method applicable to
                                       arbitrary electronic states. J. Chem. Phys. 135, (2011)

    We use notation following the paper for orbital indices:
        p,q,r,s,t,u,v,w,x,y: active space, equivalent to OBS;
                            x,y denote uncorrelated orbitals, but for now, we correlate all in this implementation
        k,l,m,n: full space, equivalent to CBS
        a,b,c,d: cabs space, full = active + cabs
    Also, we use a tensor notation, e.g. in O_KLrs O^kl_rs = <rs | O | kl>
        upper case ~ covariant/creation indices,
        lower case ~ contravariant/annihilation indices
    """
    def __init__(self, mol=None, rdm1: numpy.ndarray = None, rdm2: numpy.ndarray = None, gamma: int = 1.4,
                 n_ri: int = None, external_info: dict = None, **kwargs):
        """
        Parameters
        ----------
        mol :
            the molecule given by a QuantumChemistryBase
        rdm1 :
            1-electron reduced density matrix
        rdm2 :
            2-electron reduced density matrix
        gamma :
            f12-exponent, for a correlation factor f_12 = -1/gamma * exp[-gamma*r_12]
        n_ri :
            dimensionality of RI-basis; specify only, if want to truncate available RI-basis
            if None, then the maximum available via tensors / basis-set is used
            must not be larger than size of available RI-basis, and not smaller than size of OBS
            for n_ri==dim(OBS), the correction returns zero
        external_info :
            for usage in qc_base, need to provide information where to find one-body tensor f12-tensor <rs|f_12|pq>;
            pass dictionary with {"f12_filename": where to find f12-tensor, "ordering": ordering scheme of tensor}
        kwargs :
            e.g. RDM-information via {"U": QCircuit, "variables": optimal angles}, needs to be passed if rdm1,rdm2 not
            yet computed
        """
        # Associated molecule
        self.mol = mol

        # F12-exponent
        self.gamma = gamma

        # Check if rdm"s are provided (only check for both, do not care for providing only one)
        if (rdm1 is not None) and (rdm2 is not None):
            self.rdm1, self.rdm2 = rdm1, rdm2
        # If not provided, need to specify a circuit "U" (if parametrized, including a set of "variables")
        else:
            if "U" in kwargs:
                if "variables" in kwargs:
                    self.mol.compute_rdms(U=kwargs["U"], variables=kwargs["variables"])
                else:
                    self.mol.compute_rdms(U=kwargs["U"])
                self.rdm1, self.rdm2 = self.mol.rdm1, self.mol.rdm2
            # Check whether a psi4_method for rdms is specified, to be computed later
            elif "rdm__psi4_method" in kwargs:
                self.rdm1, self.rdm2 = None, None
            else:
                raise TequilaException("Need to either specify rdm's or provide a way how to compute them.")

        # Set dimensionalities and active space if applicable
        self.n_ri = n_ri
        if self.mol.active_space:
            # (does not need to correspond to a chemical active space)
            self.active = self.mol.active_space.active_orbitals
            self.n_obs = len(self.active)  # this might get overwritten in child classes
        else:
            self.active = None
            self.n_obs = -1

        # External info to read data from external files
        self.external_info = external_info
       
        # Set silent if in kwargs
        self.silent = False
        if "silent" in kwargs:
            self.silent = kwargs["silent"]

        # Amplitudes to be defined later
        self.t_PQrs = None

    def _sp_ansatz(self) -> numpy.ndarray:
        """
        Set amplitudes for Hylleraas functional
        Options: 1) minimize s.th. t* = argmin_t H^(2)
                 2) fix s.th. cusp conditions are fulfilled
                    following Ten-No, S. Explicitly correlated second order perturbation theory:
                    Introduction of a rational generator and numerical quadratures. J. Chem. Phys. 121, 117–129 (2004)
        here, use option 2) (computationally by far less expensive, LSE resulting out of 1) is often ill-conditioned)

        Returns
        -------
        amplitudes t_PQrs according to SP-ansatz
        """
        kron_d = numpy.eye(len(self.active))
        t_PQrs = 3 / 8 * numpy.einsum("pr,qs -> pqrs", kron_d, kron_d, optimize="greedy") \
                 + 1 / 8 * numpy.einsum("pr,qs -> qprs", kron_d, kron_d, optimize="greedy")
        return t_PQrs

    def build_fock_operator(self, h: NBodyTensor, g: NBodyTensor) -> NBodyTensor:
        """
        Assemble generalized (spin-free) Fock operator,
        fock = core_hamiltonian + rdm1*( coulomb_matrix - 1/2*coulomb_matrix_permuted )
        .. math::
                f^k_l = h^k_l + \Gamma^s_r ( g^{kr}_{ls} - 1/2*g^{kr}_{sl} ), k,l \in OBS \cup CABS, s,r \in CABS

        Returns
        -------
        NBodyTensor
            spin-free, generalized fock operator
        """
        g_1 = numpy.einsum("sr, krls -> kl", self.rdm1, g.sub_str("fafa"), optimize="greedy")
        g_2 = numpy.einsum("sr, krsl -> kl", self.rdm1,
                           numpy.einsum("krls->krsl", g.sub_str("fafa"), optimize="greedy"), optimize="greedy")
        fock = NBodyTensor(elems=h.sub_str("ff") + g_1 - 1 / 2 * g_2, active_indices=self.active, size_full=self.n_ri)

        return fock

    def size_check(self, n_ri_max: int):
        """
        Raises an error, if choice of n_ri invalid

        Parameters
        ----------
        n_ri_max :
            size of available RI-basis

        Returns
        -------

        """
        # If dimensionality of OBS is same as dimensionality of (formally-)CBS (or its approx),
        # then correction does not make sense and returns 0
        # intermediates V, B, X would be all 0
        if self.n_obs >= self.n_ri:
            raise TequilaException("Correction ineffective. Returns 0.")
        # n_ri needs to be smaller than dim(C_CBS)
        if self.n_ri > n_ri_max:
            raise TequilaException("Dimensionality of CBS too large.")

    def setup_tensors_external(self):
        """
        Setup tensors with integrals provided externally

        This assumes the f12-integrals r_PQrs = < rs | f_12 | pq > with f_12 = -1/gamma * exp(-gamma*r_12),

        Returns
        -------
        NBodyTensor
            one-body tensor h, Coulomb tensor g, f_12 tensor r
        """

        if not self.active:
            raise TequilaException("Need to use an active space ~ subset of integrals for OBS"
                                   " in order to compute correction using external integrals.")

        # Load f12-tensor
        if ("f12_filename" not in self.external_info) and ("ordering" not in self.external_info):
            raise TequilaException("Need to specify information where to find f12-integrals and ordering scheme.")
        r_elems = numpy.load(self.external_info["f12_filename"])
        # Get n_ri via size of tensors, if has not been specified
        n_ri_max = r_elems.shape[0]
        if self.n_ri is None:
            self.n_ri = n_ri_max
        self.size_check(n_ri_max=n_ri_max)
        # This assumes a f12-operator of the kind -1/gamma*exp(-gamma*r12), adjust your integrals if necessary
        r = NBodyTensor(elems=r_elems, ordering=self.external_info["ordering"],
                        active_indices=self.active, size_full=self.n_ri)
        r.reorder(to="phys")

        # Load coulomb-tensor elements
        if "two_body_ordering" in self.mol.kwargs:
            ordering = self.mol.kwargs["two_body_ordering"]
        else:
            ordering = "openfermion"
        g = NBodyTensor(elems=self.mol.molecule.two_body_integrals, ordering=ordering,
                        active_indices=self.active, size_full=self.n_ri)
        g.reorder(to="phys")

        # Load one-body tensor
        h = NBodyTensor(elems=self.mol.molecule.one_body_integrals, active_indices=self.active, size_full=self.n_ri)

        return h, g, r

    def _compute_intermediate_V(self, g: NBodyTensor, r: NBodyTensor):
        """ computes intermediate V, mixed geminal-reference block """
        rdm1, rdm2 = self.rdm1, self.rdm2
        gKLxy_rRSkl = numpy.einsum("klxy, rskl -> rsxy", g.sub_str("ffaa"),
                                   r.sub_str("aaff"), optimize="greedy")
        gTUxy_rRStu = numpy.einsum("tuxy, rstu -> rsxy", g.sub_str("aaaa"),
                                   r.sub_str("aaaa"), optimize="greedy")

        gATxy_rdm1Ut_rRSau = numpy.einsum("atxy,ut,rsau -> rsxy", g.sub_str("paaa"), rdm1,
                                          r.sub_str("aapa"), optimize="greedy")
        V_mid = gKLxy_rRSkl - gTUxy_rRStu - gATxy_rdm1Ut_rRSau

        V = numpy.einsum("pqrs, xypq, rsxy", self.t_PQrs, rdm2, V_mid, optimize="greedy")

        return V

    def _compute_intermediate_B(self, r: NBodyTensor, fock: NBodyTensor):
        """ computes intermediate B, geminal-geminal block with Fock operator """
        rdm1, rdm2 = self.rdm1, self.rdm2
        rZYpq_fockXy_rTUzx = numpy.einsum("zypq, xy, tuzx -> tupq", r.sub_str("aaaa"), fock.sub_str("aa"),
                                          r.sub_str("aaaa"), optimize="greedy")
        rAYpq_fockXa_rTUxy = numpy.einsum("aypq, xa, tuxy -> tupq", r.sub_str("paaa"),
                                          fock.sub_str("ap"), r.sub_str("aaaa"), optimize="greedy")
        rYXpq_fockAx_rTUya = numpy.einsum("yxpq, ax, tuya -> tupq", r.sub_str("aaaa"), fock.sub_str("pa"),
                                          r.sub_str("aaap"), optimize="greedy")
        rMLpq_fockKl_rTUmk = numpy.einsum("mlpq, kl, tumk -> tupq", r.sub_str("ffaa"), fock.sub_str("ff"),
                                          r.sub_str("aaff"), optimize="greedy")

        rBYpq_rdm1Xy_fockAb_rTUax = numpy.einsum("bypq, xy, ab, tuax -> tupq", r.sub_str("paaa"),
                                                 rdm1, fock.sub_str("pp"),
                                                 r.sub_str("aapa"), optimize="greedy")
        rAYpq_rdm1Xy_fockKx_rTUak = numpy.einsum("aypq, xy, kx, tuak -> tupq", r.sub_str("paaa"),
                                                 rdm1, fock.sub_str("fa"),
                                                 r.sub_str("aapf"), optimize="greedy")

        B_mid = rMLpq_fockKl_rTUmk - rZYpq_fockXy_rTUzx - rAYpq_fockXa_rTUxy - rYXpq_fockAx_rTUya \
                - 1 / 2 * rBYpq_rdm1Xy_fockAb_rTUax - 1 / 2 * rAYpq_rdm1Xy_fockKx_rTUak
        B = numpy.einsum("pqrs, vwtu, rsvw, tupq", self.t_PQrs, self.t_PQrs, rdm2, B_mid, optimize="greedy")

        return B

    def _compute_intermediate_X(self, r: NBodyTensor, fock: NBodyTensor):
        """ computes intermediate X, geminal-geminal overlap """
        rdm1, rdm2 = self.rdm1, self.rdm2

        rTUkl_rKLpq = numpy.einsum("tukl, klpq -> tupq", r.sub_str("aaff"),
                                   r.sub_str("ffaa"), optimize="greedy")
        rTUyz_rYZpq = numpy.einsum("tuyz, yzpq -> tupq", r.sub_str("aaaa"),
                                   r.sub_str("aaaa"), optimize="greedy")

        # rTUya_rdm1Yz_rZApq = numpy.einsum("tuya, yz, zapq -> tupq", r.sub_str("aaap"), rdm1,\
        # r.sub_str("apaa"), optimize="greedy")
        rUTya_rdm1Yz_rAZpq = numpy.einsum("utya, yz, azpq -> tupq", r.sub_str("aaap"), rdm1,
                                          r.sub_str("paaa"), optimize="greedy")
        rTUay_rdm1Yz_rAZqp = numpy.einsum("tuay, yz, azqp -> tupq", r.sub_str("aapa"), rdm1,
                                          r.sub_str("paaa"), optimize="greedy")

        # X_mid = rTUkl_rKLpq - rTUyz_rYZpq - 1/2*rTUya_rdm1Yz_rZApq # in paper
        X_mid = rTUkl_rKLpq - rTUyz_rYZpq - 1 / 2 * rUTya_rdm1Yz_rAZpq - 1 / 2 * rTUay_rdm1Yz_rAZqp  # adjusted to python script
        X = -1 * numpy.einsum("pqrs, vwtu, rsvx, xw, tupq", self.t_PQrs, self.t_PQrs, rdm2,
                              fock.sub_str("aa"), X_mid, optimize="greedy")

        return X

    def _compute_intermediate_Delta_paper(self, r: NBodyTensor, fock: NBodyTensor):
        """ computes Delta intermediate according to formulas in the SF-[2]_R12 paper """
        # Delta intermediate
        # Delta from paper:
        rdm1, rdm2 = self.rdm1, self.rdm2
        tPQrs_rAYpq = numpy.einsum("pqrs, aypq -> ayrs", self.t_PQrs, r.sub_str("paaa"), optimize="greedy")
        # build cumulant-like terms (call such bcs look a little bit alike, but are not same)
        cum_like_1 = numpy.einsum("sv, xrwy -> sxrvwy", rdm1,
                                  (-1 / 2 * rdm2
                                   + 1 / 2 * numpy.einsum("xw, ry -> xrwy", rdm1, rdm1, optimize="greedy")
                                   - 1 / 2 * numpy.einsum("rw, xy -> xrwy", rdm1, rdm1, optimize="greedy")),
                                  optimize="greedy")

        cum_like_2 = numpy.einsum("sw, xryv -> sxrvwy", rdm1,
                                  (-1 / 2 * rdm2
                                   + numpy.einsum("rv, xy -> xryv", rdm1, rdm1, optimize="greedy")
                                   - 1 / 4 * numpy.einsum("ry, xv -> xryv", rdm1, rdm1)),
                                  optimize="greedy")

        cum_like_3 = numpy.einsum("rv, xswy -> sxrvwy", rdm1,
                                  (rdm2
                                   - numpy.einsum("sy, xw -> xswy", rdm1, rdm1, optimize="greedy")),
                                  optimize="greedy")

        cum_like_4 = numpy.einsum("rw, xsvy -> sxrvwy", rdm1,
                                  (-1 / 2 * rdm2
                                   + 1 / 2 * numpy.einsum("sy, xv -> xsvy", rdm1, rdm1, optimize="greedy")),
                                  optimize="greedy")

        Delta_mid = numpy.einsum("ayrs, sxrvwy -> axvw", tPQrs_rAYpq,
                                 (cum_like_1 + cum_like_2 + cum_like_3 + cum_like_4),
                                 optimize="greedy")

        Delta = -1 * numpy.einsum("tuak, kx, vwtu, axvw", r.sub_str("aapf"), fock.sub_str("fa"),
                                  self.t_PQrs, Delta_mid, optimize="greedy")
        return Delta

    def _compute_intermediate_Delta_MBeq(self, r: NBodyTensor, fock: NBodyTensor):
        """ computes Delta intermediate according to formulas generated by MBEq-tool of Valeev group"""
        rdm1, rdm2, t_PQrs = self.rdm1, self.rdm2, self.t_PQrs
        # Delta from MBeq-tool
        Delta1 = -1 / 2 * numpy.einsum("pqrs, aypq, vwtu, xrvy, kx, sw, utak", t_PQrs, r.sub_str("paaa"), t_PQrs, rdm2,
                                       fock.sub_str("fa"), rdm1, r.sub_str("aapf"), optimize="greedy") \
                 - 1 / 2 * numpy.einsum("pqrs, aypq, vwtu, xryv, kx, sw, tuak", t_PQrs, r.sub_str("paaa"), t_PQrs, rdm2,
                                        fock.sub_str("fa"), rdm1, r.sub_str("aapf"), optimize="greedy") \
                 - 1 / 2 * numpy.einsum("pqrs, aypq, vwtu, kx, rv, sw, xy, utak", t_PQrs, r.sub_str("paaa"), t_PQrs,
                                        fock.sub_str("fa"), rdm1, rdm1, rdm1, r.sub_str("aapf"), optimize="greedy") \
                 + numpy.einsum("pqrs, aypq, vwtu, kx, rv, sw, xy, tuak", t_PQrs, r.sub_str("paaa"), t_PQrs,
                                fock.sub_str("fa"), rdm1, rdm1, rdm1, r.sub_str("aapf"), optimize="greedy") \
                 + 1 / 2 * numpy.einsum("pqrs, aypq, vwtu, kx, ry, sv, xw, tuak", t_PQrs, r.sub_str("paaa"), t_PQrs,
                                        fock.sub_str("fa"), rdm1, rdm1, rdm1, r.sub_str("aapf"), optimize="greedy") \
                 - 1 / 4 * numpy.einsum("pqrs, aypq, vwtu, kx, ry, sv, xw, utak", t_PQrs, r.sub_str("paaa"), t_PQrs,
                                        fock.sub_str("fa"), rdm1, rdm1, rdm1, r.sub_str("aapf"), optimize="greedy")
        Delta2 = numpy.einsum("pqrs, ayqp, vwtu, xrvy, kx, sw, utak", self.t_PQrs, r.sub_str("paaa"), self.t_PQrs, rdm2,
                              fock.sub_str("fa"), rdm1, r.sub_str("aapf"), optimize="greedy") \
                 - 1 / 2 * numpy.einsum("pqrs, ayqp, vwtu, xrvy, kx, sw, tuak", self.t_PQrs, r.sub_str("paaa"),
                                        self.t_PQrs, rdm2,
                                        fock.sub_str("fa"), rdm1, r.sub_str("aapf"), optimize="greedy") \
                 - numpy.einsum("pqrs, ayqp, vwtu, kx, ry, sv, xw, tuak", self.t_PQrs, r.sub_str("paaa"), self.t_PQrs,
                                fock.sub_str("fa"), rdm1, rdm1, rdm1, r.sub_str("aapf"), optimize="greedy") \
                 + 1 / 2 * numpy.einsum("pqrs, ayqp, vwtu, kx, ry, sv, xw, utak", self.t_PQrs, r.sub_str("paaa"),
                                        self.t_PQrs,
                                        fock.sub_str("fa"), rdm1, rdm1, rdm1, r.sub_str("aapf"), optimize="greedy")
        Delta = Delta1 + Delta2

        return Delta

    def _compute_intermediates(self, g: NBodyTensor, r: NBodyTensor, fock: NBodyTensor) -> list:
        """ calls computation of intermediates and returns them as list """
        V = self._compute_intermediate_V(g, r)
        B = self._compute_intermediate_B(r, fock)
        X = self._compute_intermediate_X(r, fock)
        Delta = self._compute_intermediate_Delta_MBeq(r, fock)
        # alternative: Delta = self._compute_intermediate_Delta_paper(r, fock), seems to perform a little worse
        if not self.silent:
            print("Intermediates:")
            print("\tV\t= ", V)
            print("\tB\t= ", B)
            print("\tX\t= ", X)
            print("\tDelta\t= ", Delta)

        return [V, B, X, Delta]

    def compute_correction(self, g: NBodyTensor, r: NBodyTensor, fock: NBodyTensor) -> float:
        """
        Computes correction based on Coulomb tensor, f12-tensor and generalized Fock operator
        Invokes SP-ansatz
        """

        # Here, if len(active_!=n_obs, then active is effective OBS, and n_ri-len(active) is effective CABS
        print("Computing with dim(OBS): " + str(self.n_obs) + ", of which " + str(len(self.active))
              + " active and dim(RI): " + str(self.n_ri) + ".")
        print("Therefore effective OBS: " + str(len(self.active)) +
              " and effective CABS: " + str(self.n_ri-len(self.active)))

        # Set amplitudes via SP-ansatz by Ten-No (fix s.th. cusp condition is fulfilled)
        self.t_PQrs = self._sp_ansatz()

        # Compute intermediates
        intermediates = self._compute_intermediates(g, r, fock)
        # Sum up intermediates for correction
        correction = numpy.sum(intermediates)

        return correction

    def compute(self) -> float:
        """
        Computes universal explicitly correlated correction based on parameters of class instance
        Tensors are read from file in base function

        Returns
        -------
            the explicitly correlated correction
        """
        print("Set up universal f12-correction using external integrals.")
        h, g, r = self.setup_tensors_external()
        fock = self.build_fock_operator(h=h, g=g)

        correction = self.compute_correction(g=g, r=r, fock=fock)

        return correction
