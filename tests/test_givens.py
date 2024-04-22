import numpy
import tequila as tq
import tequila.quantumchemistry.qc_base as qc
import tequila.tools.random_generators as rg
import random

transformations = ["JordanWigner", "ReorderedJordanWigner", "BravyiKitaev", "BravyiKitaevTree"]
def test_givens_on_molecule():
    # random size and transformation
    size = random.randint(2, 10)
    transformation = random.choice(transformations)

    # dummy one-electron integrals
    h = numpy.ones(shape=[size,size])
    # dummy two-electron integrals
    g = numpy.ones(shape=[size, size, size, size])

    U = rg.generate_random_unitary(size)

    # transformed integrals
    th = (U.T.dot(h)).dot(U)
    tg = numpy.einsum("ijkx, xl -> ijkl", g, U, optimize='greedy')
    tg = numpy.einsum("ijxl, xk -> ijkl", tg, U, optimize='greedy')
    tg = numpy.einsum("ixkl, xj -> ijkl", tg, U, optimize='greedy')
    tg = numpy.einsum("xjkl, xi -> ijkl", tg, U, optimize='greedy')

    # original molecule/H
    mol = tq.Molecule(geometry="He 0.0 0.0 0.0", nuclear_repulsion=0.0, one_body_integrals=h, two_body_integrals=g, basis_set="dummy", transformation=transformation)
    H = mol.make_hamiltonian()
    # transformed molecule/H
    tmol = tq.Molecule(geometry="He 0.0 0.0 0.0", nuclear_repulsion=0.0, one_body_integrals=th, two_body_integrals=tg,basis_set="dummy", transformation=transformation)
    tH = tmol.make_hamiltonian()

    # transformation in qubit space (this corresponds to the U above)
    UR = mol.get_givens_circuit(U) # Works!

    # test circuit
    circuit = rg.make_random_circuit(size)

    # create expectation values and see if they are the same
    E1 = tq.ExpectationValue(U=circuit, H=tH)
    E2 = tq.ExpectationValue(U=circuit + UR, H=H)

    result1 = tq.simulate(E1)
    result2 = tq.simulate(E2)
    
    assert numpy.isclose(result1, result2)

def test_givens_decomposition():
    # random unitary of random size
    size = random.randint(2, 10)
    unitary = rg.generate_random_unitary(size)

    # decompose givens
    theta_list, phi_list = qc.get_givens_decomposition(unitary)

    # reconstruct original unitary from givens
    reconstructed_matrix = qc.reconstruct_matrix_from_givens(unitary.shape[0], theta_list, phi_list)
    
    assert numpy.allclose(unitary, reconstructed_matrix)
