Reduced Density Matrices in Tequila
===================================

This notebook serves as a tutorial to the computation and usage of the
one- and two-particle reduced density matrices.

.. code:: ipython3

    import tequila as tq
    import numpy

The 1- and 2-RDM
----------------

First, look at the definition of the reduced density matrices (RDM) for
some state :math:`|\psi\rangle`:

1-RDM: :math:`\gamma^p_q\equiv\langle\psi|a^pa_q|\psi\rangle`

2-RDM :math:`\gamma^{pq}_{rs}\equiv\langle\psi|a^pa^qa_sa_r|\psi\rangle`
(we mainly use the standard physics ordering for the second-quantized
operators, i.e. :math:`p,r` go with particle 1 and :math:`q,s` with
particle 2)

The operators :math:`a^p=a_p^\dagger` and :math:`a_p` denote the
standard fermionic creation and annihilation operators.

Since we work on a quantum computer, :math:`|\psi\rangle` is represented
by some unitary transformation :math:`U`:
:math:`|\psi\rangle=U|0\rangle^{\otimesN_q}`, using :math:`N_q` qubits.
This corresponds to :math:`N_q` spin-orbitals in Jordan-Wigner encoding.
Obtaining the RDMs from a quantum computer is most intuitive when using
the Jordan-Wigner transformation, since the results directly correspond
to the ones computed classically in second quantized form.

It is worth mentioning that since we only consider real orbitals in
chemistry applications, the implementation also expects only real-valued
RDM's. The well-known anticommutation relations yield a series of
symmetry properties for the reduced density matrices, which can be taken
into consideration to reduce the computational cost:

.. raw:: latex

   \begin{align} \gamma^p_q &= \gamma^q_p \\ \gamma^{pq}_{rs} &=-\gamma^{qp}_{rs}=-\gamma^{pq}_{sr}=\gamma^{qp}_{sr}=\gamma^{rs}_{pq}\end{align}

In chemistry applications, solving the electronic structure problem
involves the electronic Hamiltonian (here in Born-Oppenheimer
approximation)

.. math::  H_{el} = h_0 + \sum_{pq} h^q_p a^p_q + \frac{1}{2}\sum_{pqrs} h^{rs}_{pq} a^{pq}_{rs}

with the one- and two-body integrals :math:`h^q_p, h^{rs}_{pq}` that
turn out to be independent of spin.

Therefore, we introduce the spin-free RDMs :math:`\Gamma^P_Q` and
:math:`\Gamma^{PQ}_{RS}`, obtained by spin-summation (we write molecular
orbitals in uppercase letters :math:`P,Q,\ldots\in\{1,\ldots,N_p\}` in
opposite to spin-orbitals :math:`p,q,\ldots\in\{1,\ldots,N_q\}`):

.. raw:: latex

   \begin{align} \Gamma^P_Q &= \sum_{\sigma \in \{\alpha, \beta\}} \gamma^{p\sigma}_{q\sigma} = \langle \psi |\sum_{\sigma} a^{p\sigma} a_{q\sigma} | \psi\rangle \\
   \Gamma^{PQ}_{RS} &= \sum_{\sigma,\tau \in \{\alpha, \beta\}} \gamma^{p\sigma q\tau}_{r\sigma s\tau} = \langle \psi | \sum_{\sigma,\tau}  a^{p\sigma} a^{q\tau} a_{s\tau} a_{r\sigma} | \psi \rangle.  \end{align}

Note, that by making use of linearity, we obtain the second equality in
the two expressions above. Performing the summation before evaluating
the expected value means less expected values and a considerable
reduction in computational cost (only :math:`N_p=\frac{N_q}{2}`
molecular orbitals vs. :math:`N_q` spin-orbitals).

Due to the orthogonality of the spin states, the symmetries for the
spin-free 2-RDM are slightly less than for the spin-orbital RDM:

.. raw:: latex

   \begin{align} \Gamma^P_Q &= \Gamma^Q_P\\
    \Gamma^{PQ}_{RS} &= \Gamma^{QP}_{SR} = \Gamma^{RS}_{PQ} \end{align}

.. code:: ipython3

    # As an example, let's use the Helium atom in a minimal basis
    mol = tq.chemistry.Molecule(geometry='He 0.0 0.0 0.0', basis_set='6-31g')
    
    # We want to get the 1- and 2-RDM for the (approximate) ground state of Helium
    # For that, we (i) need to set up a unitary transformation U(angles)
    #             (ii) determine a set of angles using VQE s.th. U(angles) |0> = |psi>, where H|psi> = E_0|psi>
    #            (iii) compute the RDMs using compute_rdms
    
    # (i) Set up a circuit
    # This can be done either using the make_uccsd-method (see Chemistry-tutorial) or by a hand-written circuit
    # We use a hand-written circuit here
    U = tq.gates.X(target=0)
    U += tq.gates.X(target=1)
    U += tq.gates.Ry(target=3, control=0, angle='a1')
    U += tq.gates.X(target=0)
    U += tq.gates.X(target=1, control=3)
    U += tq.gates.Ry(target=2, control=1, angle='a2')
    U += tq.gates.X(target=1)
    U += tq.gates.Ry(target=2, control=1, angle='a3')
    U += tq.gates.X(target=1)
    U += tq.gates.X(target=2)
    U += tq.gates.X(target=0, control=2)
    U += tq.gates.X(target=2)
    
    # (ii) Run VQE
    H = mol.make_hamiltonian()
    O = tq.objective.objective.ExpectationValue(H=H, U=U)
    result = tq.minimize(objective=O, method='bfgs')


.. code:: ipython3

    # (iii) Using the optimal parameters out of VQE, we know have a circuit U_opt |0> ~ U|0> = |psi> 
    mol.compute_rdms(U=U, variables=result.angles, spin_free=True, get_rdm1=True, get_rdm2=True)
    rdm1_spinfree, rdm2_spinfree = mol.rdm1, mol.rdm2
    print('\nThe spin-free matrices:')
    print('1-RDM:\n' + str(rdm1_spinfree))
    print('2-RDM:\n' + str(rdm2_spinfree))
    
    
    # Let's also get the spin-orbital rdm2
    # We can select to only determine one of either matrix, but if both are needed at some point, it is 
    # more efficient to compute both within one call of compute_rdms
    print('\nThe spin-ful matrices:')
    mol.compute_rdms(U=U, variables=result.angles, spin_free=False, get_rdm1=False, get_rdm2=True)
    rdm1_spin, rdm2_spin = mol.rdm1, mol.rdm2
    print('1-RDM is None now: ' + str(rdm1_spin))
    print('2-RDM has been determined:\n' + str(rdm2_spin))
    
    # We can compute the 1-rdm still at a later point
    mol.compute_rdms(U=U, variables=result.angles, spin_free=False, get_rdm1=True, get_rdm2=False)
    rdm1_spin = mol.rdm1
    print('1-RDM is also here now:\n' + str(rdm1_spin))

.. code:: ipython3

    # To check consistency with the spin-free rdms, we can do spin-summation afterwards 
    # (again, if only the spin-free version is of interest, it is cheaper to get it right from compute_rdms) 
    rdm1_spinsum, rdm2_spinsum = mol.rdm_spinsum(sum_rdm1=True, sum_rdm2=True)
    print('\nConsistency of spin summation:')
    print('1-RDM: ' + str(numpy.allclose(rdm1_spinsum, rdm1_spinfree, atol=1e-10)))
    print('2-RDM: ' + str(numpy.allclose(rdm2_spinsum, rdm2_spinfree, atol=1e-10)))

.. code:: ipython3

    # We can also compute the RDMs using the psi4-interface.
    # Then, psi4 is called to perform a CI-calculation, while collecting the 1- and 2-RDM
    # Let's use full CI here, but other CI flavors work as well
    mol.compute_rdms(psi4_method='fci')
    rdm1_psi4, rdm2_psi4 = mol.rdm1, mol.rdm2
    print('\nPsi4-RDMs:')
    print('1-RDM:\n' + str(rdm1_psi4))
    print('2-RDM:\n' + str(rdm2_psi4))
    
    # Comparing the results to the VQE-matrices, we observe a close resemblance,
    # also suggested by the obtained energies
    fci_energy = mol.logs['fci'].variables['FCI TOTAL ENERGY']
    vqe_energy = result.energy
    print('\nFCI energy: ' + str(fci_energy))
    print('VQE-Energy: ' + str(vqe_energy))

Consistency checks
------------------

At this point, we can make a few consistency checks.

We can validate the trace condition for the 1- and 2-RDM:

.. raw:: latex

   \begin{align}\mathrm{tr}(\mathbf{\Gamma}_m)&=N!/(N-m)!\\ \mathrm{tr} (\mathbf{\Gamma}_1) &= \sum_P \Gamma^P_P = N \\
    \mathrm{tr} (\mathbf{\Gamma}_2) &= \sum_{PQ} \Gamma^{PQ}_{PQ} = N(N-1), \end{align}

:math:`N` describes the number of particles involved, i.e. in our case
using a minimal basis this corresponds to :math:`N_p` above. For the
Helium atom in Born-Oppenheimer approximation, :math:`N_p=2`. In the
literature, one can also find the :math:`m`-particle reduced density
matrices normalized by a factor :math:`1/m!`, which in that case would
be inherited by the trace conditions.

Also, the (in our case, as we use the wavefunction from VQE,
ground-state) energy can be computed by

.. raw:: latex

   \begin{equation} E = \langle H_{el} \rangle = h_0 + \sum_{PQ} h^Q_P \Gamma^P_Q + \frac{1}{2}\sum_{PQRS} h^{RS}_{PQ} \Gamma^{PQ}_{RS}, \end{equation}

where :math:`h_0` denotes the nuclear repulsion energy, which is 0 for
Helium anyways.

Note, that the expressions above also hold true for the spin-RDMs, given
that the one- and two-body integrals are available in spin-orbital
basis.

.. code:: ipython3

    # Computation of consistency checks
    #todo: normalization of rdm2 *= 1/2
    # Trace
    tr1_spin = numpy.einsum('pp', rdm1_spin, optimize='greedy')
    tr1_spinfree = numpy.einsum('pp', rdm1_spinfree, optimize='greedy')
    
    tr2_spin = numpy.einsum('pqpq', rdm2_spin, optimize='greedy')
    tr2_spinfree = numpy.einsum('pqpq', rdm2_spinfree, optimize='greedy')
    
    print("1-RDM: N_true = 2, N_spin = " + str(tr1_spin) + ", N_spinfree = " + str(tr1_spinfree)+".")
    print("2-RDM: N*(N-1)_true = 2, spin = " + str(tr2_spin) + ", spinfree = " + str(tr2_spinfree)+".")
    
    # Energy
    # Get molecular integrals
    h0 = mol.molecule.nuclear_repulsion
    print("h0 is zero: " + str(h0))
    h1 = mol.molecule.one_body_integrals
    h2 = mol.molecule.two_body_integrals
    # Reorder two-body-integrals according to physics convention
    h2 = tq.chemistry.qc_base.NBodyTensor(elems=h2, scheme='openfermion')
    h2.reorder(to='phys')
    h2 = h2.elems
    # Compute energy
    rdm_energy = numpy.einsum('qp, pq', h1, rdm1_spinfree, optimize='greedy') + 1/2*numpy.einsum('rspq, pqrs', h2, rdm2_spinfree, optimize='greedy')
    print('\nVQE-Energy is:      ' + str(vqe_energy))
    print('RDM-energy matches: ' + str(rdm_energy))


References
----------

... for the definition of the reduced density matrices, spin-free
formulation, symmetries: 1. Kutzelnigg, W., Shamasundar, K. R. &
Mukherjee, D. Spinfree formulation of reduced density matrices, density
cumulants and generalised normal ordering. Mol. Phys. 108, 433–451
(2010). 2. Helgaker, T., Jørgensen, P. & Olsen, J. Molecular
Electronic-Structure Theory (John Wiley & Sons, Ltd, 2000).

Possible applications
---------------------

So far, the content of this notebook is comparably trivial, and misses
some interesting applications. An interesting possilibity on how to make
use of the RDM's obtained by a quantum computer is given by a technique
that has been named quantum subspace expansion, which e.g. can be used
to approximate excited states [3], decode quantum errors [4] or improve
the accuracy of results [5]. References herefore: 3. McClean, J. R.,
Kimchi-Schwartz, M. E., Carter, J. & De Jong, W. A. Hybrid
quantum-classical hierarchy for mitigation of decoherence and
determination of excited states. Phys. Rev. A 95, 1–10 (2017). 4.
McClean, J. R., Jiang, Z., Rubin, N. C., Babbush, R. & Neven, H.
Decoding quantum errors with subspace expansions. Nat. Commun. 11, 1–9
(2020). 5. Takeshita, T. et al. Increasing the Representation Accuracy
of Quantum Simulations of Chemistry without Extra Quantum Resources.
Phys. Rev. X 10, 11004 (2020).

Everybody is invited to enrich this notebook by implementing one of the
techniques mentioned, or some other application of the 1- and 2-RDM!
