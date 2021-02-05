import tequila as tq
import numpy


def test_simple_example():
    H = tq.paulis.X(0)
    Upre = tq.gates.X(0)
    Upost = tq.gates.Y(0)
    generators = [tq.paulis.Y(0), tq.paulis.Z(0), tq.paulis.X(0)]
    operator_pool = tq.adapt.AdaptPoolBase(generators=generators)
    solver = tq.adapt.Adapt(H=H, operator_pool=operator_pool, Upre=Upre, UPost=Upost)
    result = solver()
    assert numpy.isclose(result.energy, -1.0, atol=1.e-4)


def make_test_molecule():
    one = numpy.array([[-1.94102524, -0.31651552],
                       [-0.31651552, -0.0887454]])
    two = numpy.array([[[[1.02689005, 0.31648659],
                         [0.31648659, 0.22767214]],

                        [[0.31648659, 0.22767214],
                         [0.85813498, 0.25556095]]],

                       [[[0.31648659, 0.85813498],
                         [0.22767214, 0.25556095]],

                        [[0.22767214, 0.25556095],
                         [0.25556095, 0.76637672]]]])

    return tq.Molecule(geometry="He 0.0 0.0 0.0", backend="base", one_body_integrals=one, two_body_integrals=two)


def test_molecular_example():
    mol = make_test_molecule()
    Upost = mol.make_excitation_gate(angle="a", indices=[(0, 2)])
    Upost += mol.make_excitation_gate(angle="a", indices=[(1, 3)])
    operator_pool = tq.adapt.MolecularPool(molecule=mol, indices="UpCCSD")
    solver = tq.adapt.Adapt(H=mol.make_hamiltonian(), Upre=mol.prepare_reference(),Upost=Upost, operator_pool=operator_pool)
    result = solver(operator_pool=operator_pool, label=0)
    energy = numpy.linalg.eigvalsh(mol.make_hamiltonian().to_matrix())[0]
    assert numpy.isclose(result.energy, energy, atol=1.e-4)

def test_molecular_excited_example():
    mol = make_test_molecule()

    H = mol.make_hamiltonian()
    eigenvalues, eigenvectors = numpy.linalg.eigh(H.to_matrix())
    n_qubits = H.n_qubits
    reference_basis_state = 2 ** (n_qubits - 1) + 2 ** (n_qubits - 2)
    energies = []
    for i in range(len(eigenvalues)):
        if not numpy.isclose(eigenvectors[:, i][reference_basis_state], 0.0, atol=1.e-4):
            energies.append(eigenvalues[i])

    operator_pool = tq.adapt.MolecularPool(molecule=mol, indices="UpCCSD")

    circuits = []
    variables = {}
    for state in range(3):
        Upre = mol.prepare_reference()
        objective_factory = tq.adapt.ObjectiveFactorySequentialExcitedState(Upre=mol.prepare_reference(),
                                                                            H=mol.make_hamiltonian(), circuits=circuits,
                                                                            factors=[100.0] * len(circuits))
        solver = tq.adapt.Adapt(objective_factory=objective_factory,
                                Upre=mol.prepare_reference(),
                                operator_pool=operator_pool)
        result = solver(operator_pool=operator_pool, label=state, static_variables=variables)
        U = Upre + result.U
        circuits.append(U)
        variables = {**variables, **result.variables}
        assert numpy.isclose(result.energy, energies[state], atol=1.e-4)