import tequila as tq
import numpy

###############################################################################
# This Script was used to create the Hamiltonians with the .npy integral files#
# Leaving this in here for transparancy reasons                               #
###############################################################################

# nucelar distance in angstrom
R = 1.4

for transformation in ["jordan_wigner", "bravyi_kitaev", "symmetry_conserving_bravyi_kitaev"]:
    print("\n\n\n")
    #filename for the Hamiltonian
    filename="h2_{}_4so_{}".format(R, transformation)
    
    geomstring = "H 0.0 0.0 0.0\nH 0.0 0.0 {}".format(R)
    
    # you need to have psi4 for the next lines to work
    # but nuclear repulsion can also be set to 0.0 for a VQE
    # or in this case just be computed manually
    # leaving that piece of code in since it might be useful to have
    try:
        molx = tq.chemistry.Molecule(geometry=geomstring, basis_set="sto-3g")
        nuc = molx.molecule.nuclear_repulsion
    except:
        angstrom_to_bohr=1.8897
        nuc = 1.0/(angstrom_to_bor*R)
    
    # load integrals from file
    obi = numpy.load('one_body_integrals.npy')
    tbi = numpy.load('two_body_integrals.npy')
    
    # openfermion conventions
    # integrals are in Mulliken notation: 
    # g_{pqrs} = (pq|rs) = p(1)q(1) 1/r12 r(2)s(2)
    # need to transform to openfermion conventions
    # g_{pqrs} = p(1)s(1) 1/r12 q(2)r(2)
    tbi = numpy.einsum("psqr", tbi, optimize='optimize')
    
    data = {
        "one_body_integrals": obi,
        "two_body_integrals": tbi,
        "nuclear_repulsion": nuc,
        "n_orbitals": 2
    }
    
    # make the tequila molecule
    # basis_set is just a dummy in that case
    mol = tq.chemistry.Molecule(geometry=geomstring,
            backend="base",
            basis_set='mra',
            transformation=transformation,
            **data)
    
    H = mol.make_hamiltonian()
    
    # for the outputfile
    val, vec = numpy.linalg.eigh(H.to_matrix())
    print("created Hamiltonian with {} encoding:\n".format(transformation), H)
    print("{} spin-orbitals to {} qubits".format(mol.n_orbitals*2, H.n_qubits))
    print("ground state energy: {:2.8f}".format(val[0]))
    print("ground state is    : ", tq.QubitWaveFunction(vec[:,0]))
    


    # store in openfermion format
    Hof = H.to_openfermion()
    import openfermion as of
    of.utils.save_operator(Hof, file_name=filename, data_directory=".")
    # can be loaded with
    loaded = of.utils.load_operator(file_name=filename, data_directory=".")
    
    # reload into tequila
    H2 = tq.QubitHamiltonian.from_openfermion(loaded)
    assert(H == H2)


