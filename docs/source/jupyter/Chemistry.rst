Chemistry basics with Tequila
=============================

| Here we show the basics of the tequila chemistry module.
| In order for this to work you need to have psi4 installed in the same
  python environment as tequila.
| If you are in a conda environment installing psi4 is easy:
  ``conda install psi4 -c psi4``
| But better check the `psi4 website <http://www.psicode.org/>`__ for up
  to date instructions.

In some cases problems with the environment were observed which are
suspected to originate from conflicts between psi4 and tequila
dependcies. Usually the safest way is to install psi4 **first** and then
install tequila

Basic functionality is currently also provided with PySCF which might be
easier to install.

At the moment we only support closed-shell molecules

This tutorial will give an overview over:

-  Initialization of molecules within tequila
-  Usage of different qubit encodings from openermion (JW, BK, BKSF,
   Tapered-BK)
-  Using basic functionality of Psi4 with tequila
-  Setting up active spaces
-  Constructing UCC based quantum circuits with tequila

**There might be issues with psi4 and jupyter, currently the workarround
is to reload the kernel before a cell gets executed, or run as a regular
python script**

.. code:: ipython3

    import tequila as tq

Initialize Molecules
--------------------

Molecules can be initialized by passing their geometries as string or
the name of a ``xyz`` file.

.. code:: ipython3

    import tequila as tq
    molecule = tq.chemistry.Molecule(geometry = "H 0.0 0.0 0.0\nLi 0.0 0.0 1.6", basis_set="sto-3g")
    print(molecule)
    
    # lets also print some information about the orbitals
    # we need it later
    
    print("The Orbitals are:")
    for orbital in molecule.orbitals:
        print(orbital)

| You can initialize a tequila ``QubitHamiltonian`` from a molecule with
  ``make_hamiltonian``. The standard transformation is the
  ``jordan-wigner`` transformation.
| You can use other transformations by initializing the molecule with
  the ``transformation`` keyword.

.. code:: ipython3

    import tequila as tq
    H = molecule.make_hamiltonian()
    # the LiH Hamiltonian is already quite large, better not print the full thing
    print("Hamiltonian has {} terms".format(len(H)))

.. code:: ipython3

    molecule = tq.chemistry.Molecule(geometry = "H 0.0 0.0 0.0\nLi 0.0 0.0 1.6", basis_set="sto-3g", transformation="bravyi-kitaev")
    H = molecule.make_hamiltonian()
    print("Hamiltonian has {} terms".format(len(H)))

Using different Qubit Encodings of OpenFermion
----------------------------------------------

The different qubit encodings of openfermion can be applied by passing
the keyword ``transformation`` to the molecule and setting it to the
name of the corresponding openfermion function.

Some of those transformation might require additional keywords.
Following ``psi4`` conventions those should be given to the ``Molecule``
initialization with the prefix ``transformation__``. For most of them,
``tequila`` is however able to assign the keys automatically.

In the following we provide some examples using various transformations
from openfermion

.. code:: ipython3

    import tequila as tq
    import numpy
    geomstring = "H 0.0 0.0 0.0\nH 0.0 0.0 0.7"
    basis_set = "sto-3g"
    
    # Jordan-Wigner (this is the default)
    mol = tq.chemistry.Molecule(geometry=geomstring, basis_set=basis_set, transformation="jordan_wigner")
    H = mol.make_hamiltonian()
    print("Jordan-Wigner\n", H)
    eigenValues = numpy.linalg.eigvalsh(H.to_matrix())
    print("lowest energy = ", eigenValues[0])
    
    # Bravyi-Kitaev
    mol = tq.chemistry.Molecule(geometry=geomstring, basis_set=basis_set, transformation="bravyi_kitaev")
    H = mol.make_hamiltonian()
    print("Bravyi-Kitaev\n", H)
    eigenValues = numpy.linalg.eigvalsh(H.to_matrix())
    print("lowest energy = ", eigenValues[0])
    
    # symmetry_conserving_bravyi_kitaev
    # this transformation will taper off two qubits of the Hamiltonian
    # it needs additional information on the number of spin-orbitals and the active_fermions/electrons in the system
    mol = tq.chemistry.Molecule(geometry=geomstring, basis_set=basis_set,
                                  transformation="symmetry_conserving_bravyi_kitaev")
    H = mol.make_hamiltonian()
    print("Symmetry conserving Bravyi-Kitaev\n", H)
    eigenValues = numpy.linalg.eigvalsh(H.to_matrix())
    print("lowest energy = ", eigenValues[0])
    
    # Symmetry 

Setting active spaces
---------------------

| You can define active spaces on your molcule by passing down a
  dictionary of active orbitals.
| The orbitals are grouped into the irreducible representation of the
  underlying symmetry group (see the printout of ``print(molecule)``
  above).

Lets take the LiH molecule from above but initialize it with an active
space containing the second two A1 orbitals (meaning the first 0A1
orbital is frozen) and the B1 orbital

.. code:: ipython3

    import tequila as tq
    active_orbitals = {"A1":[1,2], "B1":[0]}
    molecule = tq.chemistry.Molecule(geometry = "H 0.0 0.0 0.0\nLi 0.0 0.0 1.6", basis_set="sto-3g", active_orbitals=active_orbitals)
    H = molecule.make_hamiltonian()
    print("Hamiltonian has {} terms".format(len(H)))

Lets make the active space even smaller, so that we can print out stuff
in this tutorial

.. code:: ipython3

    import tequila as tq
    active_orbitals = {"A1":[1], "B1":[0]}
    molecule = tq.chemistry.Molecule(geometry = "H 0.0 0.0 0.0\nLi 0.0 0.0 1.6", basis_set="sto-3g", active_orbitals=active_orbitals)
    H = molecule.make_hamiltonian()
    print("Hamiltonian has {} terms".format(len(H)))
    print(H)

Computing classical methods with Psi4
-------------------------------------

We can use psi4 to compute the energies (and sometimes other quantities)
with the ``compute_energy`` function. Here are some examples. Note that
the energies are computed within the active space if one is set.

Note also that not all active spaces can be represented by psi4 which
will mean you can/should not use the classical psi4 methods with those
(a warning will be printed). You will still get the right active space
hamiltonian however.

Active spaces which will not work for psi4 methds are the ones where the
orbitals of individual irreps are not in one block ( e.g.
{``"A1":[1,3]``} )

.. code:: ipython3

    # YOU MIGHT HAVE TO RESTART THE JUPYTER KERNEL
    import tequila as tq
    active_orbitals = {"A1":[1], "B1":[0], "B2":[0]}
    molecule = tq.chemistry.Molecule(geometry = "H 0.0 0.0 0.0\nLi 0.0 0.0 1.6", basis_set="sto-3g", active_orbitals=active_orbitals)
    
    mp2 = molecule.compute_energy(method="mp2")
    
    # Note there are known issues for some methods when the active space as frozen virtuals as is the case here
    # detci based methods are fine again 
    fci = molecule.compute_energy(method="fci")
    
    # for most coupled-cluster like models you can compute amplitudes
    # Amplitudes are computed in c1 and in the full space, this is why the active space troubles from above usually don't hold
    # Note that amplitudes are in closed-shell
    amplitudes = molecule.compute_amplitudes("mp2")
    
    # you can export a parameter dictionary which holds the indices of the amplitude as keys and values as values
    # for this small active space that is only one amplitude for mp2
    variables = amplitudes.make_parameter_dictionary()
    print(variables)
    
    # similar for ccsd since the singles are 0 due to symmetry (that changes if you change the active space)
    amplitudes = molecule.compute_amplitudes("ccsd")
    variables = amplitudes.make_parameter_dictionary()
    print(variables)

Hello World "H2" optimization with LiH in an active space
---------------------------------------------------------

Lets do a small hand-constructed VQE like it would be done for the
Hydrogen molecule in STO-3G, just that we use our active space LiH
molecule from the cell above. For consistency reasons we initialize
everything again.

Check the ``BasicUsage`` and ``SciPyOptimizers`` tutorial notebooks for
more information about then

.. code:: ipython3

    import tequila as tq
    # define the active space
    active_orbitals = {"A1":[1], "B1":[0]}
    
    # define the molecule
    molecule = tq.chemistry.Molecule(geometry = "H 0.0 0.0 0.0\nLi 0.0 0.0 1.6", basis_set="sto-3g", active_orbitals=active_orbitals)
    
    # make the hamiltonian
    H = molecule.make_hamiltonian()
    
    # define a hand designed circuit
    U = tq.gates.Ry(angle="a", target=0) + tq.gates.X(target=[2,3])
    U += tq.gates.X(target=1, control=0)
    U += tq.gates.X(target=2, control=0)
    U += tq.gates.X(target=3, control=1)
    
    # define the expectationvalue
    E = tq.ExpectationValue(H=H, U=U)
    
    # optimize
    result = tq.minimize(objective=E, method="BFGS", initial_values={k:0.0 for k in E.extract_variables()})
    
    # compute a reference value with psi4
    fci = molecule.compute_energy(method="fci")
    
    print("VQE : {:+2.8}f".format(result.energy))
    print("FCI : {:+2.8}f".format(fci))


.. code:: ipython3

    # some more information from the optimization
    result.history.plot("energies", baselines={"fci":fci})

Unitary Coupled-Cluster Style Construction
------------------------------------------

| Here we show how to initialize in the style of unitary
  coupled-cluster.
| In this example we are gonna compute the ``mp2`` amplitudes and build
  a UCC type circuit from them.
| Here we use the cc2 amplitudes only to define an order on the
  trotterized gates and remove small amplitudes as classical
  prescreening.

We use again an active space to make the computation fast.

First we start with the manual construction and then show how to use in
build convenience functions of tequila to.

| An important function is the ``make_excitation_generator`` function of
  the molecule.
| This initializes a ``QubitHamiltonian`` which can be used to define a
  unitary gate which acts as excitation operation of electrons.

.. math::

   \displaystyle
   U(\theta) = e^{-i\frac{\theta}{2} G_{iajbkc\dots}}

In fermionic language the generator is defined as

.. math::


   G_{ia,jb,jc,\dots} =  i ( a^\dagger_a a_i a^\dagger_b a_j a^\dagger_c a_k \dots - h.c. )

The ``make_excitation_generator`` function gives back this generator in
the qubit representation (depends on the chosen ``transformation`` of
the molecule) and takes the indices as list of tuples

.. math::


   \text{make_excitation_generator(indices=[(a,i),(b,j),...])} = G_{ia,jb,jc,\dots}

Manual Construction
~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # YOU MIGHT HAVE TO RESTART THE JUPYTER KERNEL
    import tequila as tq
    threshold = 1.e-6
    
    # define the active space
    active_orbitals = {"A1":[1], "B1":[0]}
    
    # define the molecule
    molecule = tq.chemistry.Molecule(geometry = "H 0.0 0.0 0.0\nLi 0.0 0.0 1.6", basis_set="sto-3g", active_orbitals=active_orbitals)
    
    # make the hamiltonian
    H = molecule.make_hamiltonian()
    
    # compute classical amplitudes
    amplitudes = molecule.compute_amplitudes(method="mp2")
    
    # in this example there is only one closed-shell MP2 amplitude, therefore manual construction is reasonable in this tutorial
    # first we make a dictionary out of the non-zero MP2 amplitudes
    ampdict = amplitudes.make_parameter_dictionary(threshold=threshold)
    print(ampdict)
    # lets get the indices of the only amplitude which is there manually
    indices = list(ampdict.keys())[0]
    
    # the (1, 0, 1, 0) index in closed shell leads to the (2, 0, 3, 1) and (3, 1, 2, 0) excitations on the qubits
    # but first we need to initialize the hartree fock state
    U = molecule.prepare_reference()
    
    # now add the two 2-electron excitations 
    # for this we define the generators and build trotterized gates with them
    # note that the two generators are actually the same
    # we sum them up since we want to parametrize them with the same variable which we will call "a"
    generator = molecule.make_excitation_generator(indices=[(3, 1),(2, 0)]) + molecule.make_excitation_generator(indices=[(3, 1),(2, 0)])
    U += tq.gates.Trotterized(generators=[generator], angles=["a"], steps=1)
    
    # define the expectationvalue
    E = tq.ExpectationValue(H=H, U=U)
    
    # optimize
    result = tq.minimize(objective=E, method="BFGS", initial_values={k:0.0 for k in E.extract_variables()})
    
    # compute a reference value with psi4
    fci = molecule.compute_energy(method="fci")
    print("VQE : {:+2.8}f".format(result.energy))
    print("FCI : {:+2.8}f".format(fci))
    


Automatic Construction
~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # YOU MIGHT HAVE TO RESTART THE JUPYTER KERNEL
    import tequila as tq
    threshold = 1.e-6
    
    # define the active space
    active_orbitals = {"A1":[1], "B1":[0]}
    
    # define the molecule
    molecule = tq.chemistry.Molecule(geometry = "H 0.0 0.0 0.0\nLi 0.0 0.0 1.6", basis_set="sto-3g", active_orbitals=active_orbitals)
    
    # make the hamiltonian
    H = molecule.make_hamiltonian()
    
    # make the UCCSD ansatz (note that this will be without singles since it starts from mp2)
    U = molecule.make_uccsd_ansatz(initial_amplitudes="mp2", threshold=threshold, trotter_steps=1)
    
    # define the expectationvalue
    E = tq.ExpectationValue(H=H, U=U)
    
    # optimize
    result = tq.minimize(objective=E, method="BFGS", initial_values={k:0.0 for k in E.extract_variables()})
    
    # compute a reference value with psi4
    fci = molecule.compute_energy(method="fci")
    
    print("VQE : {:+2.8}f".format(result.energy))
    print("FCI : {:+2.8}f".format(fci))
        
        

Pi System of Benzene
~~~~~~~~~~~~~~~~~~~~

Lets repeat the last cell with the pi system of the benzene molecule

.. code:: ipython3

    # YOU MIGHT HAVE TO RESTART THE JUPYTER KERNEL
    import tequila as tq
    threshold = 1.e-6
    active = {"B1u": [0], "B3g": [0, 1], "B2g": [0], "Au": [0], "b1u": [1]}
    molecule = tq.quantumchemistry.Molecule(geometry="data/benzene.xyz", basis_set='sto-3g', active_orbitals=active)
    H = molecule.make_hamiltonian()
    
    # make the UCCSD ansatz
    U = molecule.make_uccsd_ansatz(initial_amplitudes="cc2", threshold=threshold, trotter_steps=1)
    
    # define the expectationvalue
    E = tq.ExpectationValue(H=H, U=U)
    
    # compute reference energies
    fci = molecule.compute_energy("fci")
    cisd = molecule.compute_energy("detci", options={"detci__ex_level": 2})
    
    # optimize
    # Scipy either `eps` (version <1.5) or 'finite_diff_rel_step' (verion > 1.5)
    # for the 2-point stencil
    result = tq.minimize(objective=E, method="BFGS", gradient="2-point", method_options={"eps":1.e-4, "finite_diff_rel_step":1.e-4, "gtol": 1.e-3}, initial_values={k:0.0 for k in E.extract_variables()})
    
    print("VQE : {:+2.8}f".format(result.energy))
    print("CISD: {:+2.8}f".format(cisd))
    print("FCI : {:+2.8}f".format(fci))

.. code:: ipython3

    result.history.plot("energies", baselines={"fci":fci, "cisd": cisd}, filename="benzene_result_bfgs")

Noisy optimization of an active space molecule with tapered qubit embeding
--------------------------------------------------------------------------

This example shows the combination of several features of tequila in a
few lines

-  automatic handling of active spaces
-  consitent usage of qubit encodings (here the
   symmetry\_conserving\_bravyi\_kitaev encoding from openfermion which
   reduces the number of qubits by 2)
-  custom circuit construction
-  unitary cluster circuits
-  optimization of measurements (here the Hamiltonian will be grouped
   into two commuting groups, this can be seen by the optimizer output
   which holds two expectation values). See the
   `MeasurementGroups <https://github.com/aspuru-guzik-group/tequila/blob/master/tutorials/MeasurementGroups.ipynb>`__
   tutorial for more background information.
-  interface to different quantum backends (you will need qiskit to run
   this cell)

.. code:: ipython3

    import tequila as tq
    # define the active space
    active_orbitals = {"A1":[1], "B1":[0]}
    samples = 1000000
    backend = "qiskit"
    device = "fake_rome"
    # define the molecule
    molecule = tq.chemistry.Molecule(geometry = "H 0.0 0.0 0.0\nLi 0.0 0.0 1.6",
                                     basis_set="sto-3g",
                                     active_orbitals=active_orbitals,
                                     transformation="symmetry_conserving_bravyi_kitaev")
    
    fci = molecule.compute_energy("fci")
    
    H = molecule.make_hamiltonian()
    
    # Toy circuit (no deeper meaning)
    U = tq.gates.Ry(angle="a", target=0)
    U += tq.gates.X(target=1, control=0)
    E = tq.ExpectationValue(H=H, U=U, optimize_measurements=True)
    
    vqe = tq.minimize(method="cobyla", objective=E, initial_values=0.0)
    noisy_vqe = tq.minimize(method="cobyla", objective=E, samples=samples, backend=backend, device=device, initial_values=0.0)
     
    # The same with UpCCGSD
    UpCCGSD = molecule.make_upccgsd_ansatz(include_singles=False)
    E2 = tq.ExpectationValue(H=H, U=UpCCGSD, optimize_measurements=True)
    ucc = tq.minimize(method="cobyla", objective=E2, initial_values=0.0)
    noisy_ucc = tq.minimize(method="cobyla", objective=E2, samples=samples,  backend=backend, device=device, initial_values=0.0)
    
    print("VQE         = {:2.8f}".format(min(vqe.history.energies)))
    print("VQE (noisy) = {:2.8f}".format(min(noisy_vqe.history.energies)))
    print("UCC         = {:2.8f}".format(min(ucc.history.energies)))
    print("UCC (noisy) = {:2.8f}".format(min(noisy_ucc.history.energies)))


.. code:: ipython3

    # repeat printout
    print("VQE         = {:2.3f}".format(min(vqe.history.energies)))
    print("VQE (noisy) = {:2.3f}".format(min(noisy_vqe.history.energies)))
    print("UCC         = {:2.3f}".format(min(ucc.history.energies)))
    print("UCC (noisy) = {:2.3f}".format(min(noisy_ucc.history.energies)))


