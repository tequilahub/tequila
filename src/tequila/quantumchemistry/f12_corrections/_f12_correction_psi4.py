import tequila as tq
from tequila import TequilaException, TequilaWarning

from tequila.quantumchemistry.qc_base import NBodyTensor
from tequila.quantumchemistry.f12_corrections._f12_correction_base import ExplicitCorrelationCorrection

import numpy
from itertools import product

import psi4


# since this needs psi4, put in external file! maybe own folder for corrections
class ExplicitCorrelationCorrectionPsi4(ExplicitCorrelationCorrection):
    """
    Class for computation of explicitly correlated correction using integrals from psi4
    """

    def __init__(self, mol=None, rdm1: numpy.ndarray = None, rdm2: numpy.ndarray = None, gamma: float = 1.4,
                 n_ri: int = None, cabs_type: str = "active", cabs_options: dict = None, **kwargs):
        """

        Parameters
        ----------
        mol :
            the molecule given by a QuantumChemistryPsi4
        rdm1 :
            1-electron reduced density matrix
        rdm2 :
            2-electron reduced density matrix
        gamma :
            f12-exponent, for a correlation factor f_12 = -1/gamma * exp[-gamma*r_12]
        n_ri :
            dimensionality of RI-basis; if None, then the maximum available via tensors / basis-set is used
        cabs_type :
            - either "active" for using a given basis set as is as approximative CBS (complete basis set), and specify
            OBS (orbital basis) by an active space
            - or "cabs+" for CABS+-approach by Valeev in [...] -> pass cabs_name in cabs_options
        cabs_options :
            dict, which needs at least {"cabs_name": some CABS basis set} if cabs_type=="cabs+"
        kwargs :
            e.g. RDM-information via {"U": QCircuit, "variables": optimal angles} if computation via VQE,
            or {"rdm__psi4_method": some CI method, "rdm__psi4_options": dict with psi4 options} if computation via
            psi4, compare to psi4_interface.compute_rdms
            one of the above needs to be passed if rdm1,rdm2 not yet computed
        """
        super().__init__(mol=mol, rdm1=rdm1, rdm2=rdm2, n_ri=n_ri, gamma=gamma, **kwargs)

        self.cabs_type = cabs_type.lower()
        self.cabs_options = cabs_options
        # n_obs and active are set in setup_tensor-function for cabs+

        # If cabs_type == "cabs+", need C1-symmetry
        if self.cabs_type == "cabs+":
            if not self.mol._point_group.lower() == 'c1':
                raise TequilaException("CABS+ approach currently requires C1 symmetry!")

        if (self.rdm1 is None) or (self.rdm2 is None):
            if "rdm__psi4_method" in kwargs:
                if "rdm__psi4_options" in kwargs:
                    self.mol.compute_rdms(psi4_method=kwargs["rdm__psi4_method"],
                                          psi4_options=kwargs["rdm__psi4_options"])
                else:
                    self.mol.compute_rdms(psi4_method=kwargs["rdm__psi4_method"])
                self.rdm1, self.rdm2 = self.mol.rdm1, self.mol.rdm2
            # This exception would already have been caught in base class
            else:
                raise TequilaException("Need to either specify rdm's or provide a way how to compute them.")

    # "Active-space" approach, requires psi4
    def setup_tensors_active(self):
        """
        Easiest but least effective way to get a correction.
        Use some GBS, of which an active space is used as OBS, the inactive ("passive") space as CABS.
        This is not equivalent to CABS+, however still <CABS, OBS> = 0 via MOs

        cabs_options can be empty here, only necessary info is active space of mol

        Returns
        -------
        NBodyTensor
            one-body tensor h, Coulomb tensor g, f_12 tensor r
        """

        # HF - ref_wfn
        ref_wfn = self.mol.logs["hf"].wfn
        if ref_wfn.nirrep() != 1:
            wfn = ref_wfn.c1_deep_copy(ref_wfn.basisset())
        else:
            wfn = ref_wfn

        # Workaround: active space for VQE, full space in HF as RI-space
        C_RI = wfn.Ca()  # == C_OBS
        n_ri_max = C_RI.shape[0]
        if self.n_ri is None:
            self.n_ri = n_ri_max
        self.size_check(n_ri_max=n_ri_max)

        # Calculate r-intermediates using a fitted slater-type correlation factor -1*exp(-gamma*r)
        mints = psi4.core.MintsHelper(wfn.basisset())
        correlationFactor = psi4.core.FittedSlaterCorrelationFactor(self.gamma)
        r_elems = numpy.asarray(mints.mo_f12(correlationFactor, C_RI, C_RI, C_RI, C_RI))
        r = NBodyTensor(elems=r_elems, ordering="chem", active_indices=self.active, size_full=self.n_ri)
        r.reorder(to="phys")
        r.elems /= self.gamma  # ensure that f_12 = -1/gamma * exp(-gamma*r)

        # Load Coulomb matrix elements
        if "two_body_ordering" in self.mol.kwargs:
            ordering = self.mol.kwargs["two_body_ordering"]
        else:
            ordering = "openfermion"
        g = NBodyTensor(elems=self.mol.molecule.two_body_integrals, ordering=ordering, active_indices=self.active,
                        size_full=self.n_ri)
        g.reorder(to="phys")

        h = NBodyTensor(elems=self.mol.molecule.one_body_integrals, active_indices=self.active, size_full=self.n_ri)

        return h, g, r

    def setup_tensors_psi4_cabsplus(self):
        """
        For now, this works only with C1-symmetry!

        Computes the required integrals via the CABS+ approach in
            Valeev, E. F. Improving on the resolution of the identity in
            linear R12 ab initio theories. Chem. Phys. Lett. 395, 190–195 (2004).

        Currently, this requires that psi4 from the following fork is installed:
            https://github.com/philipp-q/psi4/tree/ri_space
            (integration into psi4 is planned)

        Requires cabs_options with name of CABS basis by keyword "cabs_name"

        Returns
        -------
        NBodyTensor
            one-body tensor h, Coulomb tensor g, f_12 tensor r
        """
        if "lindep_tol" in self.cabs_options:
            lindep_tol = self.cabs_options["lindep_tol"]
        else:
            lindep_tol = 0.00001

        if "cabs_name" in self.cabs_options:
            cabs_name = self.cabs_options["cabs_name"]
        else:
            raise TequilaException("Need to specify a CABS basis set!")

        # Set up bases
        ref_wfn = self.mol.logs["hf"].wfn
        ref_mol = self.mol.logs["hf"].mol
        obs_basis = ref_wfn.basisset()
        obs_space = psi4.core.OrbitalSpace("p", "AO", ref_wfn)

        # Set RI-basis (RI = OBS + (1-P_OBS)*CABS) -> CABS+ - approach
        ri_mol = ref_mol  # ok, psi4.core.Molecule does not have a member BasisSet; way to inherit geometry information
        obs_key = ref_wfn.basisset().name()
        aux_key = cabs_name
        # This function is commented in psi4-github (the following procedure is a lousy way to get a working
        # variant of psi4.core.BasisSet.build_ri_space
        ri_basis_dict = psi4.driver.qcdb.BasisSet.pyconstruct_combined(psi4.driver.qcdb.Molecule(ri_mol.to_dict()),
                                                                       [obs_key, aux_key],
                                                                       [obs_key, aux_key],
                                                                       ["ORBITAL", "F12"], [obs_key, aux_key])
        # Augment libints-basis-dict by key and blend for psi4
        ri_basis_dict["key"] = "RI-BASIS"
        ri_basis_dict["blend"] = "RI-BASIS"
        ri_basis = psi4.core.BasisSet.construct_from_pydict(ri_mol, ri_basis_dict, 0)
        # Get RI-space by orthogonalizing ri_basis (function not in public psi4-github)
        ri_space = psi4.core.OrbitalSpace.cheap_ri_space(ri_basis, lindep_tol=lindep_tol)
        # Re-overwrite ri_basis, to play safe
        ri_basis = ri_space.basisset()

        # Build CABS via CABS = (1-P_OBS)RI (orthogonal complement of OBS in RI/CBS)
        cabs_space = psi4.core.OrbitalSpace.build_cabs_space(obs_space, ri_space, linear_tol=lindep_tol)

        # Get coefficient matrices
        if ref_wfn.nirrep() != 1:
            raise TequilaException("CABS+ approach works only with C1 symmetry currently!")
        else:
            wfn = ref_wfn
        C_OBS = wfn.Ca()
        # Project C_OBS onto ri_basis (noccpi = number of relevant eigenvectors in old_basis here, not occ orbitals)
        C_OBS = ref_wfn.basis_projection(C_OBS, psi4.core.Dimension([C_OBS.shape[0]]), obs_basis, ri_basis)

        # psi4_cabsplus needs customary treatment of active, a specification from outside is not necessary
        # active corresponds to size of "OBS"-summands in correction ~ dim(rdm1)
        if self.active is None:
            if self.rdm1.shape[0] < C_OBS.shape[1]:     # this case distinction is semantically pointless, but there to
                self.active = list(range(self.rdm1.shape[0]))  # emphasize the meaning of active here
            else:
                self.active = list(range(C_OBS.shape[1]))
        # n_obs != len(active) here (active space is a "real" active space)
        self.n_obs = C_OBS.shape[1]

        C_RI = ri_space.C()
        C_CABS = cabs_space.C()
        n_ri_max = C_RI.shape[0]
        if self.n_ri is None:
            self.n_ri = n_ri_max
        self.size_check(n_ri_max=n_ri_max)

        # Print size of individual spaces (needs to sum up)
        # print("OBS-size", C_OBS.shape)
        # print("RI-size", C_RI.shape)
        # print("CABS-size", C_CABS.shape)

        # From now on, we DO NOT use C_RI, but C_OBS with C_CABS, and divide up summations over CBS
        # into summations over OBS and CABS
        # This is necessary here, since using C_RI ~> C_RI is orthogonal within itself, but we generally want
        # CABS = orthogonal_complement(OBS, RI), and OBS basis needs to be same as for generation of RDM's

        # Set up mints
        mints = psi4.core.MintsHelper(ri_basis)
        correlationFactor = psi4.core.FittedSlaterCorrelationFactor(self.gamma)

        # Compute and assemble tensors
        # Helper functions
        def get_irange(i: int):
            """ Returns indexing range for OBS and CABS part """
            irange = None
            if i == 0:  # OBS
                irange = range(self.n_obs)
            elif i == 1:  # CABS
                irange = range(self.n_obs, self.n_ri)
            return irange

        def ind(i: int, p: int):
            """ Translates CBS to CABS indices for CABS-integrals """
            if i == 0:  # OBS -> leave as is
                return p
            elif i == 1:  # CABS -> translate to range [0,...,|CABS|]
                return p - self.n_obs

        orb_matrices = ((0, C_OBS), (1, C_CABS))
        g_elems = numpy.zeros((self.n_ri, self.n_ri, self.n_ri, self.n_ri))
        r_elems = numpy.zeros((self.n_ri, self.n_ri, self.n_ri, self.n_ri))

        # Assemble tensors now in the following way: For a 1/2-body tensor, compute elements by integrals over
        # OBS and CABS-space, and then assemble s.th. CBS=OBS+CABS; for any 2/4 set of basis-functions, provide coefficient
        # matrix for OBS and CABS, giving 2*2=4/2*2*2*2=16 sub-tensors, which need to be correctly assembled
        # Keep Mulliken-ordering within here, else associativity (i1,C1) is lost!
        # TODO: FOR OBS,OBS WE ALREADY HAVE THE G- AND H-INTEGRALS! NO NEED TO RECOMPUTE, BUT THEN NEED TO PROJECT ON RI-BASIS
        # TODO: CHECK WHAT IS MORE EFFICIENT -- THIS WAY, COMPUTING THE AO_INTEGRALS TOO OFTEN BUT C++ PERFORMANCE FOR MO-CONSTRUCTION
        # TODO -- OR, COMPUTING THE A0_F12, AO_ERI, AND THEN CONTRACT LIKE 1-ELECTRON INTEGRALS HERE IN PYTHON
        for o1, o2, o3, o4 in product(orb_matrices, repeat=4):
            i1, i2, i3, i4 = o1[0], o2[0], o3[0], o4[0]
            C1, C2, C3, C4 = o1[1], o2[1], o3[1], o4[1]
            tmp_g_ints = numpy.asarray(mints.mo_eri(C1, C2, C3, C4))
            tmp_r_ints = numpy.asarray(mints.mo_f12(correlationFactor, C1, C2, C3, C4))
            for p in get_irange(i1):
                t = ind(i1, p)
                for q in get_irange(i2):
                    u = ind(i2, q)
                    for rr in get_irange(i3):
                        v = ind(i3, rr)
                        for s in get_irange(i4):
                            w = ind(i4, s)
                            g_elems[p, q, rr, s] = tmp_g_ints[t, u, v, w]
                            r_elems[p, q, rr, s] = tmp_r_ints[t, u, v, w]

        r = NBodyTensor(elems=r_elems, ordering="chem", active_indices=self.active, size_full=self.n_ri)
        r.reorder(to="phys")
        r.elems /= self.gamma  # ensure that f_12 = -1/gamma * exp(-gamma*r)

        g = NBodyTensor(elems=g_elems, ordering="chem", active_indices=self.active, size_full=self.n_ri)
        g.reorder(to="phys")

        # h_tensor, h = T + V, same procedure as before
        h_elems = numpy.zeros((self.n_ri, self.n_ri))
        t_Kl = numpy.asarray(mints.ao_kinetic())
        v_Kl = numpy.asarray(mints.ao_potential())
        for o1, o2 in product(orb_matrices, repeat=2):
            i1, i2 = o1[0], o2[0]
            C1, C2 = o1[1], o2[1]
            # h = ao_kinetic + ao_potential, mints in RI-basisset (indexing Kl)
            tmp_h = t_Kl + v_Kl
            tmp_h = numpy.einsum("ji, kl, jk -> il", C1, C2, tmp_h, optimize="greedy")  # ~ C1^T tmp_h C2
            for p in get_irange(i1):
                t = ind(i1, p)
                for q in get_irange(i2):
                    u = ind(i2, q)
                    h_elems[p, q] = tmp_h[t, u]
        h = NBodyTensor(elems=h_elems, active_indices=self.active, size_full=self.n_ri)

        return h, g, r

    def compute(self) -> float:
        """
        Computes universal explicitly correlated correction based on parameters of class instance
        Builds tensors either by a naive active space of the CABS+-approach

        Returns
        -------
            the explicitly correlated correction
        """
        # Prepare tensors
        if self.cabs_type.lower() == "active":
            print("Set up universal f12-correction using a HF-RI-basis, with an active space as OBS.")
            h, g, r = self.setup_tensors_active()
        elif self.cabs_type.lower() == "cabs+":
            print("Set up universal f12-correction using the CABS+ approach.")
            try:
                h, g, r = self.setup_tensors_psi4_cabsplus()
            except:
                raise TequilaException("Something went wrong. Probably the psi4 version you have installed\
                                        does not support the CABS-functionality.\
                                        See preamble to setup_tensors_psi4_cabsplus.")
        else:
            raise TequilaException("No cabs_method to specify integrals provided.")
        fock = super().build_fock_operator(h=h, g=g)

        # Compute correction using base class method
        correction = super().compute_correction(g=g, r=r, fock=fock)

        return correction
