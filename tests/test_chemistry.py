import pytest
import tequila.quantumchemistry as qc
import numpy
import os, glob

import tequila.quantumchemistry.chemistry_tools
import tequila.simulators.simulator_api
from tequila.objective import ExpectationValue
from tequila.quantumchemistry.encodings import known_encodings
from tequila.simulators.simulator_api import simulate

HAS_PYSCF = "pyscf" in qc.INSTALLED_QCHEMISTRY_BACKENDS
HAS_PSI4 = "psi4" in qc.INSTALLED_QCHEMISTRY_BACKENDS

import tequila as tq

# Get QC backends for parametrized testing
import select_backends

backends = select_backends.get()


def teardown_function(function):
    [os.remove(x) for x in glob.glob("data/*.pickle")]
    [os.remove(x) for x in glob.glob("data/*.out")]
    [os.remove(x) for x in glob.glob("data/*.hdf5")]
    [os.remove(x) for x in glob.glob("*.clean")]
    [os.remove(x) for x in glob.glob("data/*.npy")]
    [os.remove(x) for x in glob.glob("*.npy")]
    [os.remove(x) for x in glob.glob("qvm.log")]
    [os.remove(x) for x in glob.glob("*.dat")]

@pytest.mark.skipif(condition=not HAS_PSI4 and not HAS_PYSCF, reason="you don't have psi4 or pyscf")
def test_UR_and_UC():
    mol = tq.Molecule(geometry="h 0.0 0.0 0.0\nH 0.0 0.0 1.5", basis_set="sto-3g")
    mol = mol.use_native_orbitals()
    H = mol.make_hamiltonian()
    U = mol.prepare_reference()
    U+= mol.UC(0,1)
    U+= mol.UR(0,1)
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(E)
    fci = mol.compute_energy("fci")
    assert numpy.isclose(result.energy,fci)


@pytest.mark.parametrize("trafo", list(known_encodings().keys()))
def test_base(trafo):
    obt = numpy.asarray([[-1.94102524, -0.31651552], [-0.31651552, -0.0887454]])
    tbt = numpy.asarray(
        [[[[1.02689005, 0.31648659], [0.31648659, 0.22767214]], [[0.31648659, 0.22767214], [0.85813498, 0.25556095]]],
         [[[0.31648659, 0.85813498], [0.22767214, 0.25556095]], [[0.22767214, 0.25556095], [0.25556095, 0.76637672]]]])
    np = 0.0
    n = 2
    molecule = tq.chemistry.Molecule(backend="base", geometry="he 0.0 0.0 0.0", basis_set="whatever",
                                     transformation=trafo, one_body_integrals=obt, two_body_integrals=tbt,
                                     nuclear_repulsion=np, n_orbitals=2)
    H = molecule.make_hamiltonian()
    eigvals = numpy.linalg.eigvalsh(H.to_matrix())
    assert numpy.isclose(eigvals[0], -2.87016214e+00)
    if "trafo" in ["JordanWigner", "BravyiKitaev", "bravyi_kitaev_fast",
                   "BravyiKitaevTree"]:  # others change spectrum outside of the groundstate
        assert numpy.isclose(eigvals[-1], 7.10921141e-01)
        assert len(eigvals) == 16


@pytest.mark.skipif(condition=not HAS_PSI4 and not HAS_PYSCF, reason="you don't have psi4 or pyscf")
@pytest.mark.parametrize("trafo", ["JordanWigner", "BravyiKitaev", "BravyiKitaevTree"])
def test_prepare_reference(trafo):
    geometry = "Li 0.0 0.0 0.0\nH 0.0 0.0 1.5"
    basis_set = "sto-3g"
    mol = tq.Molecule(geometry=geometry, basis_set=basis_set, transformation=trafo)
    H = mol.make_hamiltonian()
    U = mol.prepare_reference()
    E = tq.ExpectationValue(H=H, U=U)
    energy = tq.simulate(E)
    hf_energy = mol.compute_energy("hf")
    assert numpy.isclose(energy, hf_energy, atol=1.e-4)
    mol = tq.Molecule(geometry=geometry, basis_set=basis_set, transformation="reordered" + trafo)
    H = mol.make_hamiltonian()
    U = mol.prepare_reference()
    E = tq.ExpectationValue(H=H, U=U)
    energy2 = tq.simulate(E)
    assert numpy.isclose(energy, energy2, atol=1.e-4)

@pytest.mark.skipif(condition=not HAS_PSI4 and not HAS_PYSCF, reason="you don't have psi4 or pyscf")
def test_orbital_types():
    geometry = "H 0.0 0.0 0.0\nH 0.0 0.0 2.0\nH 0.0 0.0 4.0\nH 0.0 0.0 6.0"

    mol1 = tq.Molecule(geometry=geometry, basis_set="sto-3g")
    mol2 = tq.Molecule(geometry=geometry, basis_set="sto-3g", orbital_type="hf")
    mol3 = tq.Molecule(geometry=geometry, basis_set="sto-3g", orbital_type="native")
    
    energy = mol1.compute_energy("fci")
    for mol in [mol1, mol2, mol3]:
        fci = mol.compute_energy("fci")
        e0 = numpy.linalg.eigvalsh(mol.make_hamiltonian().to_matrix())[0]
        assert numpy.isclose(fci, energy)
        assert numpy.isclose(e0, energy)
    
    # test initialization
    geometry = "Be 0.0 0.0 0.0\nH 0.0 0.0 2.0\nH 0.0 0.0 1.5"
    
    mol1 = tq.Molecule(geometry=geometry, basis_set="sto-3g")
    mol2 = tq.Molecule(geometry=geometry, basis_set="sto-3g", orbital_type="hf")
    mol3 = tq.Molecule(geometry=geometry, basis_set="sto-3g", orbital_type="native")


@pytest.mark.skipif(condition=not HAS_PSI4, reason="you don't have psi4")
@pytest.mark.parametrize("trafo_args", [{"transformation": "JordanWigner"}, {"transformation": "BravyiKitaev"},
                                        {"transformation": "bravyi_kitaev_fast"},
                                        {"transformation": "TaperedBravyiKitaev",
                                         "transformation__active_orbitals": 4, "transformation__active_fermions": 2}])
def test_transformations(trafo_args):
    geomstring = "H 0.0 0.0 0.0\nH 0.0 0.0 0.7"
    molecule = tq.chemistry.Molecule(geometry=geomstring, basis_set="sto-3g", **trafo_args)
    gs = numpy.linalg.eigvalsh(molecule.make_hamiltonian().to_matrix())[0]
    assert numpy.isclose(gs, -1.1361894540879054)


@pytest.mark.dependencies
def test_dependencies():
    for key in qc.SUPPORTED_QCHEMISTRY_BACKENDS:
        assert key in qc.INSTALLED_QCHEMISTRY_BACKENDS.keys()


@pytest.mark.skipif(condition=not HAS_PSI4 and not HAS_PYSCF, reason="no quantum chemistry backends installed")
def test_interface():
    molecule = tq.chemistry.Molecule(basis_set='sto-3g', geometry="data/h2.xyz", transformation="JordanWigner")


@pytest.mark.skipif(condition=not HAS_PSI4, reason="you don't have psi4")
def test_h2_hamiltonian_psi4():
    do_test_h2_hamiltonian(qc_interface=qc.QuantumChemistryPsi4)

@pytest.mark.skipif(condition=not HAS_PYSCF, reason="you don't have pyscf")
def test_h2_hamiltonian_psi4():
    do_test_h2_hamiltonian(qc_interface=qc.QuantumChemistryPySCF)

def do_test_h2_hamiltonian(qc_interface):
    parameters = tequila.quantumchemistry.chemistry_tools.ParametersQC(geometry="data/h2.xyz", basis_set="sto-3g")
    H = qc_interface(parameters=parameters).make_hamiltonian().to_matrix()
    vals = numpy.linalg.eigvalsh(H)
    assert (numpy.isclose(vals[0], -1.1368354639104123, atol=1.e-4))
    assert (numpy.isclose(vals[1], -0.52718972, atol=1.e-4))
    assert (numpy.isclose(vals[2], -0.52718972, atol=1.e-4))
    assert (numpy.isclose(vals[-1], 0.9871391, atol=1.e-4))


@pytest.mark.skipif(condition=not HAS_PSI4, reason="you don't have psi4")
@pytest.mark.parametrize("trafo", ["JordanWigner", "BravyiKitaev",
                                   "BravyiKitaevTree"])  # bravyi_kitaev_fast not yet supported for ucc
@pytest.mark.parametrize("backend", backends)
def test_ucc_psi4(trafo, backend):
    if backend == "symbolic":
        pytest.skip("skipping for symbolic simulator  ... way too slow")
    parameters_qc = tequila.quantumchemistry.chemistry_tools.ParametersQC(geometry="data/h2.xyz", basis_set="sto-3g")
    do_test_ucc(qc_interface=qc.QuantumChemistryPsi4, parameters=parameters_qc, result=-1.1368354639104123, trafo=trafo,
                backend=backend)


@pytest.mark.skipif(condition=not HAS_PSI4, reason="you don't have psi4")
def test_ucc_singles_psi4():
    parameters_qc = tequila.quantumchemistry.chemistry_tools.ParametersQC(geometry="data/h2.xyz", basis_set="6-31G")
    # default backend is fine
    # will not converge if singles are not added
    do_test_ucc(qc_interface=qc.QuantumChemistryPsi4, parameters=parameters_qc, result=-1.15016, trafo="JordanWigner",
                backend=None)


def do_test_ucc(qc_interface, parameters, result, trafo, backend="qulacs"):
    # check examples for comments
    psi4_interface = qc_interface(parameters=parameters, transformation=trafo)

    hqc = psi4_interface.make_hamiltonian()
    amplitudes = psi4_interface.compute_ccsd_amplitudes()
    U = psi4_interface.make_uccsd_ansatz(trotter_steps=1, initial_amplitudes=amplitudes, include_reference_ansatz=True)
    variables = amplitudes.make_parameter_dictionary()
    H = psi4_interface.make_hamiltonian()
    ex = ExpectationValue(U=U, H=H)
    energy = simulate(ex, variables=variables, backend=backend)
    assert (numpy.isclose(energy, result, atol=1.e-3))


@pytest.mark.skipif(condition=not HAS_PSI4, reason="you don't have psi4")
def test_mp2_psi4():
    # the number might be wrong ... its definetely not what psi4 produces
    # however, no reason to expect projected MP2 is the same as UCC with MP2 amplitudes
    parameters_qc = tequila.quantumchemistry.chemistry_tools.ParametersQC(geometry="data/h2.xyz", basis_set="sto-3g")
    do_test_mp2(qc_interface=qc.QuantumChemistryPsi4, parameters=parameters_qc, result=-1.1344497203826904)


def do_test_mp2(qc_interface, parameters, result):
    # check examples for comments
    psi4_interface = qc_interface(parameters=parameters)
    hqc = psi4_interface.make_hamiltonian()

    amplitudes = psi4_interface.compute_mp2_amplitudes()
    variables = amplitudes.make_parameter_dictionary()

    U = psi4_interface.make_uccsd_ansatz(trotter_steps=1, initial_amplitudes=amplitudes,
                                         include_reference_ansatz=True)
    H = psi4_interface.make_hamiltonian()
    O = ExpectationValue(U=U, H=H)

    energy = simulate(objective=O, variables=variables)
    assert (numpy.isclose(energy, result))


@pytest.mark.skipif(condition=not HAS_PSI4, reason="you don't have psi4")
@pytest.mark.parametrize("method", ["cc2", "ccsd", "cc3"])
def test_amplitudes_psi4(method):
    results = {"mp2": -1.1279946983462537, "cc2": -1.1344484090805054, "ccsd": None, "cc3": None}
    # the number might be wrong ... its definitely not what psi4 produces
    # however, no reason to expect projected MP2 is the same as UCC with MP2 amplitudes
    parameters_qc = tequila.quantumchemistry.chemistry_tools.ParametersQC(geometry="data/h2.xyz", basis_set="sto-3g")
    do_test_amplitudes(method=method, qc_interface=qc.QuantumChemistryPsi4, parameters=parameters_qc,
                       result=results[method])


def do_test_amplitudes(method, qc_interface, parameters, result):
    # check examples for comments
    psi4_interface = qc_interface(parameters=parameters)
    hqc = psi4_interface.make_hamiltonian()
    if result is None:
        result = psi4_interface.compute_energy(method=method)
    amplitudes = psi4_interface.compute_amplitudes(method=method)
    variables = amplitudes.make_parameter_dictionary()

    U = psi4_interface.make_uccsd_ansatz(trotter_steps=1, initial_amplitudes=amplitudes,
                                         include_reference_ansatz=True)
    H = psi4_interface.make_hamiltonian()
    O = ExpectationValue(U=U, H=H)

    energy = simulate(objective=O, variables=variables)
    assert (numpy.isclose(energy, result))


@pytest.mark.skipif(condition=not HAS_PSI4, reason="psi4 not found")
@pytest.mark.parametrize("method", ["mp2", "mp3", "mp4", "cc2", "cc3", "ccsd", "ccsd(t)", "cisd", "cisdt"])
def test_energies_psi4(method):
    # mp3 needs C1 symmetry
    parameters_qc = tequila.quantumchemistry.chemistry_tools.ParametersQC(geometry="data/h2.xyz", basis_set="6-31g")
    if method in ["mp3", "mp4"]:
        psi4_interface = qc.QuantumChemistryPsi4(parameters=parameters_qc, point_group="c1")
    else:
        psi4_interface = qc.QuantumChemistryPsi4(parameters=parameters_qc)

    result = psi4_interface.compute_energy(method=method)
    assert result is not None


@pytest.mark.skipif(condition=not HAS_PSI4, reason="psi4 not found")
def test_restart_psi4():
    h2 = tq.chemistry.Molecule(geometry="data/h2.xyz", basis_set="6-31g")
    wfn = h2.logs['hf'].wfn
    h2x = tq.chemistry.Molecule(geometry="data/h2x.xyz", basis_set="6-31g", guess_wfn=wfn)
    wfnx = h2x.logs['hf'].wfn
    # new psi4 version changed printout
    # can currently only test if it does not crash (no guarantee that it actually read in)
    # with open(h2x.logs['hf'].filename, "r") as f:
    #     found = False
    #     for line in f:
    #         if "Reading orbitals from file 180" in line:
    #             found = True
    #             break
    #     assert found

    wfnx.to_file("data/test_wfn.npy")
    h2 = tq.chemistry.Molecule(geometry="data/h2.xyz", basis_set="6-31g", name="data/andreasdorn",
                               guess_wfn="data/test_wfn.npy")
    # with open(h2.logs['hf'].filename, "r") as f:
    #     found = False
    #     for line in f:
    #         if "Reading orbitals from file 180" in line:
    #             found = True
    #             break
    #     assert found


@pytest.mark.skipif(condition=not HAS_PSI4, reason="psi4 not found")
@pytest.mark.parametrize("active", [{"A1": [2, 3]}, {"B2": [0], "B1": [0]}, {"A1": [0, 1, 2, 3]}, {"B1": [0]}])
def test_psi4_active_spaces(active):
    mol = tq.chemistry.Molecule(geometry="data/h2o.xyz", basis_set="sto-3g", active_orbitals=active)
    H = mol.make_hamiltonian()
    Uhf = mol.prepare_reference()
    hf = tequila.simulators.simulator_api.simulate(tq.ExpectationValue(U=Uhf, H=H))
    assert (tq.numpy.isclose(hf, mol.energies["hf"], atol=1.e-4))
    qubits = 2 * sum([len(v) for v in active.values()])
    assert (H.n_qubits == qubits)


@pytest.mark.skipif(condition=not HAS_PSI4, reason="psi4 not found")
def test_rdms_psi4():
    rdm1_ref = numpy.array([[1.97710662, 0.0], [0.0, 0.02289338]])
    rdm2_ref = numpy.array([[[[1.97710662, 0.0], [0.0, -0.21275021]], [[0.0, 0.0], [0.0, 0.0]]],
                            [[[0.0, 0.0], [0.0, 0.0]], [[-0.21275021, 0.0], [0.0, 0.02289338]]]])
    mol = qc.Molecule(geometry="data/h2.xyz", basis_set="sto-3g", backend="psi4", transformation="jordan-wigner")
    # Check matrices by psi4
    mol.compute_rdms(U=None, psi4_method="detci", psi4_options={"detci__ex_level": 2,
                                                                "detci__opdm": True, "detci__tpdm": True})
    rdm1, rdm2 = mol.rdm1, mol.rdm2
    assert (numpy.allclose(rdm1, rdm1_ref, atol=1e-8))
    assert (numpy.allclose(rdm2, rdm2_ref, atol=1e-8))


@pytest.mark.skipif(condition=not HAS_PSI4 and not HAS_PYSCF, reason="psi4/pyscf not found")
@pytest.mark.parametrize("geometry", ["H 0.0 0.0 0.0\nH 0.0 0.0 0.7"])
@pytest.mark.parametrize("trafo", tq.quantumchemistry.encodings.known_encodings())
def test_upccgsd(geometry, trafo):
    molecule = tq.chemistry.Molecule(geometry=geometry, basis_set="sto-3g", transformation=trafo)
    if not molecule.supports_ucc():
        return
    energy = do_test_upccgsd(molecule)
    fci = molecule.compute_energy("fci")
    assert numpy.isclose(fci, energy, atol=1.e-3)
    energy2 = do_test_upccgsd(molecule, label="asd", order=2)
    assert numpy.isclose(fci, energy2, atol=1.e-3)


@pytest.mark.skipif(condition=not HAS_PSI4 and not HAS_PYSCF, reason="psi4 or pyscf not found")
def test_upccgsd_singles():
    molecule = tq.chemistry.Molecule(geometry="H 0.0 0.0 0.0\nH 0.0 0.0 0.7", basis_set="6-31G")
    H = molecule.make_hamiltonian()
    energy1 = numpy.linalg.eigvalsh(H.to_matrix())[0]
    energy2 = do_test_upccgsd(molecule)
    fci = molecule.compute_energy("fci")
    assert numpy.isclose(fci, energy1, atol=1.e-3)
    assert numpy.isclose(fci, energy2, atol=1.e-3)


def do_test_upccgsd(molecule, *args, **kwargs):
    U = molecule.make_upccgsd_ansatz(*args, **kwargs)
    H = molecule.make_hamiltonian()
    E = tq.ExpectationValue(U=U, H=H)

    result = tq.minimize(objective=E, initial_values=0.0, gradient="2-point", method="bfgs",
                         method_options={"finite_diff_rel_step": 1.e-4, "eps": 1.e-4})

    # test variable map in action
    variables = result.variables
    vm = {k: str(k) + "X" for k in E.extract_variables()}
    variables2 = {vm[k]: v for k, v in variables.items()}
    E2 = E.map_variables(vm)
    print(E.extract_variables())
    print(E2.extract_variables())
    energy1 = tq.simulate(E, variables=variables)
    energy2 = tq.simulate(E2, variables=variables2)
    assert numpy.isclose(energy1,energy2,atol=1.e-4)

    return result.energy


@pytest.mark.parametrize("backend", tq.simulators.simulator_api.INSTALLED_SIMULATORS.keys())
@pytest.mark.skipif(condition=not HAS_PSI4 and not HAS_PYSCF, reason="psi4/pyscf not found")
def test_hamiltonian_reduction(backend):
    mol = tq.chemistry.Molecule(geometry="H 0.0 0.0 0.0\nH 0.0 0.0 0.7", basis_set="6-31G")
    hf = mol.compute_energy("hf")
    U = mol.prepare_reference()
    H = mol.make_hamiltonian()
    E = tq.simulate(tq.ExpectationValue(H=H, U=U), backend=backend)
    assert numpy.isclose(E, hf, atol=1.e-4)
    for q in range(8):
        U2 = U + tq.gates.X(target=q) + tq.gates.Y(target=q) + tq.gates.Y(target=q) + tq.gates.X(target=q)
        E = tq.simulate(tq.ExpectationValue(H=H, U=U2), backend=backend)
        assert numpy.isclose(E, hf, atol=1.e-4)


@pytest.mark.skipif(condition=not HAS_PSI4 and not HAS_PYSCF, reason="psi4/pyscf not found")
@pytest.mark.parametrize("assume_real", [True, False])
@pytest.mark.parametrize("trafo", ["jordan_wigner", "bravyi_kitaev", "tapered_bravyi_kitaev"])
def test_fermionic_gates(assume_real, trafo):
    mol = tq.chemistry.Molecule(geometry="H 0.0 0.0 0.7\nLi 0.0 0.0 0.0", basis_set="sto-3g")
    U1 = mol.prepare_reference()
    U2 = mol.prepare_reference()
    variable_count = {}
    for i in [0, 1, 0]:
        for a in numpy.random.randint(2, 5, 3):
            idx = [(2 * i, 2 * a), (2 * i + 1, 2 * a + 1)]
            U1 += mol.make_excitation_gate(indices=idx, angle=(i, a), assume_real=assume_real)
            g = mol.make_excitation_generator(indices=idx)
            U2 += tq.gates.Trotterized(generator=g, angle=(i, a), steps=1)
            if (i, a) in variable_count:
                variable_count[(i, a)] += 1
            else:
                variable_count[(i, a)] = 1

    a = numpy.random.choice(U1.extract_variables(), 1)[0]

    H = mol.make_hamiltonian()
    E = tq.ExpectationValue(H=H, U=U1)
    dE = tq.grad(E, a)
    if not assume_real:
        assert dE.count_expectationvalues() == 4 * variable_count[a.name]
    else:
        assert dE.count_expectationvalues() == 2 * variable_count[a.name]

    E2 = tq.ExpectationValue(H=H, U=U2)
    dE2 = tq.grad(E2, a)

    variables = {k: numpy.random.uniform(0.0, 2.0 * numpy.pi, 1)[0] for k in E.extract_variables()}
    test1 = tq.simulate(E, variables=variables)
    test1x = tq.simulate(E2, variables=variables)
    test2 = tq.simulate(dE, variables=variables)
    test2x = tq.simulate(dE2, variables=variables)

    assert numpy.isclose(test1, test1x, atol=1.e-6)
    assert numpy.isclose(test2, test2x, atol=1.e-6)


@pytest.mark.skipif(condition=not HAS_PSI4 and not HAS_PYSCF, reason="psi4/pyscf not found")
@pytest.mark.parametrize("trafo", ["JordanWigner", "BravyiKitaev", "BravyiKitaevTree", "ReorderedJordanWigner",
                                   "ReorderedBravyiKitaev"])
def test_hcb(trafo):
    geomstring = "Be 0.0 0.0 0.0\n H 0.0 0.0 1.6\n H 0.0 0.0 -1.6"
    mol1 = tq.Molecule(geometry=geomstring, active_orbitals=[1, 2, 3, 4, 5, 6], basis_set="sto-3g",
                       transformation="ReorderedJordanWigner")
    H = mol1.make_hardcore_boson_hamiltonian()
    U = mol1.make_upccgsd_ansatz(name="HCB-UpCCGD")

    E = tq.ExpectationValue(H=H, U=U)
    energy1 = tq.minimize(E).energy
    assert numpy.isclose(energy1, -15.527740838656282, atol=1.e-3)

    mol2 = tq.Molecule(geometry=geomstring, active_orbitals=[1, 2, 3, 4, 5, 6], basis_set="sto-3g",
                       transformation=trafo)
    H = mol2.make_hamiltonian()
    U = mol2.make_upccgsd_ansatz(name="UpCCGD", hcb_optimization=False)
    # U = mol2.prepare_reference() + mol2.make_excitation_gate(indices=[(0,4),(1,5)], angle="a")# tq.gates.QubitExcitation(angle="a", target=[0,2,6,8])
    print(U)
    E = tq.ExpectationValue(H=H, U=U)
    energy2 = tq.minimize(E).energy

    assert numpy.isclose(energy1, energy2)


# cross testing
@pytest.mark.skipif(condition=not HAS_PYSCF, reason="pyscf not found")
@pytest.mark.skipif(condition=not HAS_PSI4, reason="psi4 not found")
@pytest.mark.parametrize("method", ["hf", "mp2", "ccsd", "ccsd(t)", "fci"])
@pytest.mark.parametrize("geometry", ["h 0.0 0.0 0.0\nh 0.0 0.0 0.7", "li 0.0 0.0 0.0\nh 0.0 0.0 1.5",
                                      "Be 0.0 0.0 0.0\nH 0.0 0.0 1.5\nH 0.0 0.0 -1.5"])
@pytest.mark.parametrize("basis_set", ["sto-3g"])
def test_pyscf_methods(method, geometry, basis_set):
    mol = tq.Molecule(geometry=geometry, basis_set=basis_set, backend="psi4")

    e1 = mol.compute_energy(method=method)
    mol = tq.MoleculeFromTequila(mol=mol, backend="pyscf")

    e2 = mol.compute_energy(method)
    assert numpy.isclose(e1, e2, atol=1.e-4)

    mol = tq.Molecule(geometry=geometry, basis_set=basis_set, backend="pyscf")
    e3 = mol.compute_energy(method)
    assert numpy.isclose(e1, e3, atol=1.e-4)


@pytest.mark.skipif(condition=not HAS_PYSCF, reason="pyscf not found")
def test_orbital_optimization():
    from tequila.quantumchemistry import optimize_orbitals
    mol = tq.Molecule(geometry="Li 0.0 0.0 0.0\nH 0.0 0.0 3.0", basis_set="STO-3G")

    circuit = mol.make_upccgsd_ansatz(name="UpCCGD")
    mol2 = optimize_orbitals(molecule=mol, circuit=circuit).molecule
    mol2 = optimize_orbitals(molecule=mol2, circuit=circuit).molecule
    mol2 = optimize_orbitals(molecule=mol2, circuit=circuit).molecule
    H = mol2.make_hamiltonian()
    E = tq.ExpectationValue(H=H, U=circuit)
    result = tq.minimize(E, print_level=2)
    assert numpy.isclose(-7.79860454, result.energy, atol=1.e-3)


@pytest.mark.skipif(condition=not HAS_PYSCF, reason="pyscf not found")
def test_orbital_transformation():
    mol0 = tq.Molecule(geometry="Li 0.0 0.0 0.0\nH 0.0 0.0 0.75", basis_set="STO-3G", frozen_core=False)
    mol0.print_basis_info()
    mol1 = mol0.orthonormalize_basis_orbitals()

    fci = mol0.compute_energy("fci")
    assert numpy.isclose(mol1.compute_energy("fci"), fci, atol=1.e-4)

    U = tq.gates.X([2]) + tq.gates.Ry(angle="a", target=4) + tq.gates.CNOT(4, 2) + tq.gates.X([0, 1])
    U += tq.gates.CNOT(2, 3)
    U += tq.gates.CNOT(4, 5)
    E0 = tq.ExpectationValue(H=mol0.make_hamiltonian(), U=U)

    hf = tq.simulate(E0, variables={"a": 0.0})

    guess = numpy.eye(mol0.n_orbitals)
    guess[1] = [0.001, 1.0, 1.0, 0.0, 0.0, 1.0]
    guess[2] = [0.001, 1.0, 1.0, 0.0, 0.0, -1.0]
    guess[5] = [0.000, 1.0, -1.0, 0.0, 0.0, 0.0]
    opt = tq.chemistry.optimize_orbitals(circuit=U, molecule=mol1, initial_guess=guess, silent=True)
    print(opt.mo_coeff)

    mol2 = mol1.transform_orbitals(opt.mo_coeff)
    assert numpy.isclose(mol2.compute_energy("fci"), fci, atol=1.e-4)
    E2 = tq.ExpectationValue(H=mol2.make_hamiltonian(), U=U)
    hf2 = tq.simulate(E2, variables={"a": 0.0})
    assert numpy.isclose(hf2, hf, atol=1.e-4)

    mol3 = opt.molecule
    assert numpy.isclose(mol3.compute_energy("fci"), fci, atol=1.e-4)
    E3 = tq.ExpectationValue(H=mol3.make_hamiltonian(), U=U)
    hf3 = tq.simulate(E3, variables={"a": 0.0})
    assert numpy.isclose(hf3, hf, atol=1.e-4)


@pytest.mark.skipif(condition=not HAS_PYSCF, reason="pyscf not found")
@pytest.mark.skipif(condition=not HAS_PSI4, reason="psi4 not found")
def test_crosscheck_mp2():
    # Be has issues with degeneracies and ordering (t1-t2 is not necessarily 0)
    mol1 = tq.Molecule(geometry="he 0.0 0.0 0.0", basis_set="6-31G", backend="psi4")
    mol2 = tq.Molecule(geometry="he 0.0 0.0 0.0", basis_set="6-31G", backend="pyscf")
    t2_1, e1 = mol1.compute_mp2_amplitudes(return_energy=True)
    t2_2, e2 = mol2.compute_mp2_amplitudes(return_energy=True)
    assert numpy.isclose(e1, e2, atol=1.e-4)
    assert numpy.isclose(numpy.linalg.norm(t2_1.tIjAb - t2_2.tIjAb), 0.0, atol=1.e-4)


@pytest.mark.skipif(condition=not HAS_PYSCF, reason="pyscf not found")
@pytest.mark.skipif(condition=not HAS_PSI4, reason="psi4 not found")
def test_crosscheck_cis():
    mol1 = tq.Molecule(geometry="he 0.0 0.0 0.0", basis_set="6-31G", backend="psi4")
    mol2 = tq.Molecule(geometry="he 0.0 0.0 0.0", basis_set="6-31G", backend="pyscf")
    r_1 = mol1.compute_cis_amplitudes()
    r_2 = mol2.compute_cis_amplitudes()
    for i in range(len(r_1.omegas)):
        assert numpy.isclose(r_1.omegas[i], r_2.omegas[i], atol=1.e-4)
        x = r_1.amplitudes[i].tIA.T.dot(r_1.amplitudes[i].tIA)
        y = r_2.amplitudes[i].tIA.T.dot(r_2.amplitudes[i].tIA)
        assert numpy.isclose(numpy.linalg.norm(x - y), 0.0, atol=1.e-4)


@pytest.mark.skipif(condition=not HAS_PYSCF, reason="pyscf not found")
@pytest.mark.skipif(condition=not HAS_PSI4, reason="psi4 not found")
def test_crosscheck_cis_mp2_large():
    # only energies (degeneracies: orbitals change places)
    mol1 = tq.Molecule(geometry="be 0.0 0.0 0.0", basis_set="6-31G", backend="psi4")
    mol2 = tq.Molecule(geometry="be 0.0 0.0 0.0", basis_set="6-31G", backend="pyscf")
    r_1 = mol1.compute_cis_amplitudes()
    r_2 = mol2.compute_cis_amplitudes()
    for i in range(len(r_1.omegas)):
        assert numpy.isclose(r_1.omegas[i], r_2.omegas[i], atol=1.e-4)
    x, e_1 = mol1.compute_mp2_amplitudes(return_energy=True)
    x, e_2 = mol2.compute_mp2_amplitudes(return_energy=True)
    assert e_1 is not None
    assert e_2 is not None
    assert numpy.isclose(e_1, e_2, atol=1.e-4)


@pytest.mark.skipif(condition=not HAS_PSI4 and not HAS_PYSCF, reason="psi4/pyscf not found")
def test_spa_ansatz_be():
    print("\n\n")
    print(HAS_PSI4)
    print(HAS_PYSCF)
    print(not HAS_PSI4 and not HAS_PYSCF)
    edges = [(0,), (1, 2, 3, 4)]
    # doing without frozen-core to test explicitly if single-orbital pairs work
    mol = tq.Molecule(geometry="be 0.0 0.0 0.0", basis_set="sto-3g", transformation="BravyiKitaev", frozen_core=False)
    H = mol.make_hamiltonian()
    U = mol.make_ansatz(name="SPA", edges=edges)
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(E, silent=True)
    energy = result.energy

    mol = tq.Molecule(geometry="be 0.0 0.0 0.0", basis_set="sto-3g", frozen_core=False)
    H = mol.make_hamiltonian()
    U0 = mol.make_ansatz(name="SPA", edges=edges)
    U1 = mol.make_ansatz(name="SPA", ladder=False, edges=edges)
    U2 = mol.make_ansatz(name="SPA", optimize=False, edges=edges)
    U3 = mol.make_ansatz(name="SPA", optimize=False, ladder=False, edges=edges)

    for U in [U0, U1, U2, U3]:
        E = tq.ExpectationValue(H=H, U=U)
        result = tq.minimize(E, silent=True)
        assert numpy.isclose(energy, result.energy)
        print("a")

#@pytest.mark.skipif(condition=not HAS_PSI4 and not HAS_PYSCF, reason="psi4/pyscf not found")
@pytest.mark.parametrize("name", ["SPA", "HCB-SPA"])
@pytest.mark.parametrize("optimize", [True, False])
@pytest.mark.parametrize("geometry", ["H 0.0 0.0 0.0\nH 0.0 0.0 4.5", "Li 0.0 0.0 0.0\nH 0.0 0.0 3.0", "Be 0.0 0.0 0.0\nH 0.0 0.0 3.0\nH 0.0 0.0 -3.0"])
@pytest.mark.parametrize("transformation", tq.quantumchemistry.encodings.known_encodings())
@pytest.mark.skipif(condition=not HAS_PSI4 and not HAS_PYSCF, reason="psi4/pyscf not found")
def test_spa_consistency(geometry, name, optimize, transformation):
    mol = tq.Molecule(geometry=geometry, basis_set="sto-3g",
                      transformation=transformation).use_native_orbitals()

    if mol.transformation.hcb_to_me() is None:
        return
    # doesn't need to make physical sense for the test
    edges = []
    i = 0
    for i in range(mol.n_electrons//2):
        a = i+mol.n_electrons//2
        edge = (i,a)
        edges.append(edge)
    U = mol.make_ansatz("SPA", edges=edges, optimize=optimize)
    variables = {k: 1.0 for k in U.extract_variables()}
    wfn = (tq.simulate(U, variables=variables))

    UX = mol.make_ansatz(name=name, edges=edges, optimize=optimize)
    wfnx = (tq.simulate(UX, variables=variables))
    print(wfnx)
    if "HCB" in name:
        UX += mol.hcb_to_me()
    wfnx = (tq.simulate(UX, variables=variables))
    F = abs(wfn.inner(wfnx)) ** 2
    if not numpy.isclose(F, 1.0):
        print(wfn)
        print(wfnx)
        print(UX)

    assert numpy.isclose(F, 1.0)

@pytest.mark.skipif(condition=not HAS_PSI4 and not HAS_PYSCF, reason="psi4/pyscf not found")
def test_variable_consistency():
    geometry = "H 0.0 0.0 0.0\nH 0.0 0.0 1.0\nH 0.0 0.0 2.0\nH 0.0 0.0 3.0"
    mol = tq.Molecule(geometry=geometry, basis_set="sto-3g", transformation="ReorderedJordanWigner").use_native_orbitals()
    U = mol.make_ansatz("SPA",edges=[(0,1),(2,3)])
    variables = {k:1.0 for k in U.extract_variables()}

    H = mol.make_hamiltonian()
    E = tq.ExpectationValue(H=H, U=U)
    E1 = tq.simulate(E, variables=variables)

    U = mol.make_ansatz("HCB-SPA",edges=[(0,1),(2,3)])
    H = mol.make_hardcore_boson_hamiltonian(condensed=False)
    E = tq.ExpectationValue(H=H, U=U)
    E2 = tq.simulate(E, variables=variables)


    mol = tq.Molecule(geometry=geometry, basis_set="sto-3g", transformation="JordanWigner").use_native_orbitals()
    U = mol.make_ansatz("SPA",edges=[(0,1),(2,3)])
    H = mol.make_hamiltonian(condensed=False)
    E = tq.ExpectationValue(H=H, U=U)
    E3 = tq.simulate(E, variables=variables)

    U = mol.make_ansatz("HCB-SPA",edges=[(0,1),(2,3)])
    H = mol.make_hardcore_boson_hamiltonian(condensed=False)
    E = tq.ExpectationValue(H=H, U=U)
    E4 = tq.simulate(E, variables=variables)

    assert numpy.isclose(E1,E2,atol=1.e-4)
    assert numpy.isclose(E1,E3,atol=1.e-4)
    assert numpy.isclose(E1,E4,atol=1.e-4)
    assert numpy.isclose(E2,E3,atol=1.e-4)
    assert numpy.isclose(E2,E4,atol=1.e-4)

@pytest.mark.skipif(condition=not HAS_PSI4 and not HAS_PYSCF, reason="you don't have psi4 or pyscf")
@pytest.mark.parametrize("geometry", ["H 0.0 0.0 0.0\nH 0.0 0.0 4.5", "Li 0.0 0.0 0.0\nH 0.0 0.0 3.0", "Be 0.0 0.0 0.0\nH 0.0 0.0 3.0\nH 0.0 0.0 -3.0"])
@pytest.mark.parametrize("optimize", [True, False])
def test_hcb_rdms(geometry, optimize):

    mol1 = tq.Molecule(geometry=geometry, basis_set="sto-3g", transformation="ReorderedJordanWigner")
    H = mol1.make_hamiltonian()
    HCB = mol1.make_hardcore_boson_hamiltonian()

    if mol1.n_electrons == 4:
        edges=[(0, 2, 5), (1, 3, 4)]
    elif mol1.n_electrons == 2:
        edges=[(0, 1)]
    else:
        raise Exception("test only created for n_electrons = 2,4 you gave {}".format(mol1.n_electrons))


    U1_1 = mol1.make_ansatz("SPA", edges=edges, optimize=optimize)
    U1_2 = mol1.make_ansatz("HCB-SPA", edges=edges, optimize=optimize)
    E1 = tq.ExpectationValue(H=H, U=U1_1)
    E2 = tq.ExpectationValue(H=HCB, U=U1_2)

    variables = {k: 1.0 for k in U1_1.extract_variables()}
    energy1 = tq.simulate(E1, variables=variables)
    energy2 = tq.simulate(E2, variables=variables)

    xrdm1_1, xrdm1_2 = mol1.compute_rdms(U1_1, variables=variables, use_hcb=False)
    yrdm1_1, yrdm1_2 = mol1.compute_rdms(U1_2, variables=variables, use_hcb=True)

    c,h1,h2 = mol1.get_integrals()
    # this is the default for RDMs
    # integrals are given in openfermion ordering
    h2 = h2.reorder(to="phys").elems

    energy3 = numpy.einsum('qp, pq', h1,xrdm1_1, optimize='greedy')
    energy3+= 1 / 2 * numpy.einsum('rspq, pqrs', h2,xrdm1_2, optimize='greedy')
    energy3+= c


    d1_1 = tq.numpy.linalg.norm(xrdm1_1 - yrdm1_1)
    d1_2 = tq.numpy.linalg.norm(xrdm1_2 - yrdm1_2)

    assert numpy.isclose(d1_1, 0., atol=1.e-4)
    assert numpy.isclose(d1_2, 0., atol=1.e-4)

    assert numpy.isclose(energy1, energy2, atol=1.e-4)
    assert numpy.isclose(energy1, energy3, atol=1.e-4)


@pytest.mark.skipif(condition=not HAS_PYSCF, reason="you don't have pyscf")
@pytest.mark.parametrize("geometry", ["H 0.0 0.0 0.0\nH 0.0 0.0 4.5", "Li 0.0 0.0 0.0\nH 0.0 0.0 3.0", "Be 0.0 0.0 0.0\nH 0.0 0.0 3.0\nH 0.0 0.0 -3.0"])
def test_orbital_optimization_hcb(geometry):

    mol = tq.Molecule(geometry=geometry, basis_set="sto-3g")


    if mol.n_electrons == 4:
        edges=[(0, 2, 5), (1, 3, 4)]
    elif mol.n_electrons == 2:
        edges=[(0, 1)]
    else:
        raise Exception("test only created for n_electrons = 2,4 you gave {}".format(mol.n_electrons))


    U1 = mol.make_ansatz(name="HCB-SPA", edges=edges)
    import time
    start = time.time()
    opt1 = tq.chemistry.optimize_orbitals(circuit=U1, molecule=mol, silent=True, use_hcb=True)
    time1 = time.time()-start
    U2 = mol.make_ansatz(name="SPA", edges=edges)
    start = time.time()
    opt2 = tq.chemistry.optimize_orbitals(circuit=U2, molecule=mol, silent=True)
    time2 = time.time()-start

    assert numpy.isclose(opt1.energy,opt2.energy,atol=1.e-5)
    assert time1 < time2
    assert (numpy.isclose(opt1.mo_coeff,opt2.mo_coeff, atol=1.e-5)).all()
