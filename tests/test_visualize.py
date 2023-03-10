import pytest
from tequila.circuit.visualize import visualize
import sys
import tequila
import openfermion
import matplotlib.pyplot as plt

if 'networkx' in sys.modules:
    NETWORKX_EXIST = True
else:
    NETWORKX_EXIST = False

if 'matplotlib' in sys.modules:
    MATPLOTLIB_EXIST = True
else:
    MATPLOTLIB_EXIST = False

if 'openfermion' in sys.modules:
    OPENFERMION_EXIST = True
else:
    OPENFERMION_EXIST = False


@pytest.mark.skipif(condition=not NETWORKX_EXIST, reason="You don't have networkx")
@pytest.mark.skipif(condition=not MATPLOTLIB_EXIST, reason="You don't have matplotlib")
@pytest.mark.skipif(condition=not OPENFERMION_EXIST, reason="You don't have openfermion")
@pytest.mark.parametrize("qh", tequila.QubitHamiltonian(openfermion.QubitOperator("X0 X5 Y3", 1)))
def test_visualize_with_qubit_hamiltonian(qh):
    fig = visualize(qh)
    assert plt.fignum_exists(fig.number)


@pytest.mark.skipif(condition=not NETWORKX_EXIST, reason="You don't have networkx")
@pytest.mark.skipif(condition=not MATPLOTLIB_EXIST, reason="You don't have matplotlib")
@pytest.mark.skipif(condition=not OPENFERMION_EXIST, reason="You don't have openfermion")
@pytest.mark.parametrize(
    "qh", tequila.QubitHamiltonian(openfermion.QubitOperator("X0 X5 Y3", 1))
          + tequila.QubitHamiltonian(openfermion.QubitOperator("X0 Y3", 1)))
def test_visualize_with_invalid_qubit_hamiltonian(qh):
    with pytest.raises(Exception) as e_info:
        visualize(qh)


@pytest.mark.skipif(condition=not NETWORKX_EXIST, reason="You don't have networkx")
@pytest.mark.skipif(condition=not MATPLOTLIB_EXIST, reason="You don't have matplotlib")
@pytest.mark.skipif(condition=not OPENFERMION_EXIST, reason="You don't have openfermion")
@pytest.mark.parametrize("qh, circuit", [tequila.QubitHamiltonian(openfermion.QubitOperator("X0 X5 Y3", 1)), tequila.gates.X(0) + tequila.gates.CNOT(0, 5) + tequila.gates.Y(3)])
def test_visualize_with_circui(qh, circuit):
    fig = visualize(qh, circuit=circuit)
    assert plt.fignum_exists(fig.number)


@pytest.mark.skipif(condition=not NETWORKX_EXIST, reason="You don't have networkx")
@pytest.mark.skipif(condition=not MATPLOTLIB_EXIST, reason="You don't have matplotlib")
@pytest.mark.skipif(condition=not OPENFERMION_EXIST, reason="You don't have openfermion")
@pytest.mark.parametrize("qh, connectivity", [tequila.QubitHamiltonian(openfermion.QubitOperator("X0 X5 Y3", 1)), [(0, 0), (0, 5), (3, 3)]])
def test_visualize_with_connectivity(qh, connectivity):
    fig = visualize(qh, connectivity=connectivity)
    assert plt.fignum_exists(fig.number)


@pytest.mark.skipif(condition=not NETWORKX_EXIST, reason="You don't have networkx")
@pytest.mark.skipif(condition=not MATPLOTLIB_EXIST, reason="You don't have matplotlib")
@pytest.mark.skipif(condition=not OPENFERMION_EXIST, reason="You don't have openfermion")
@pytest.mark.parametrize("qh, file", [tequila.QubitHamiltonian(openfermion.QubitOperator("X0 X5 Y3", 1)), "test_file1"])
def test_visualize_with_file(qh, file):
    fig = visualize(qh, file_path=file)
    assert plt.fignum_exists(fig.number)


@pytest.mark.skipif(condition=not NETWORKX_EXIST, reason="You don't have networkx")
@pytest.mark.skipif(condition=not MATPLOTLIB_EXIST, reason="You don't have matplotlib")
@pytest.mark.skipif(condition=not OPENFERMION_EXIST, reason="You don't have openfermion")
@pytest.mark.parametrize("qh, circuit, file", [tequila.QubitHamiltonian(openfermion.QubitOperator("X0 X5 Y3", 1)), tequila.gates.X(0) + tequila.gates.CNOT(0, 5) + tequila.gates.Y(3), "test_file2"])
def test_visualize_with_circuit_and_file(qh, circuit, file):
    fig = visualize(qh, circuit=circuit, file_path=file)
    assert plt.fignum_exists(fig)


@pytest.mark.skipif(condition=not NETWORKX_EXIST, reason="You don't have networkx")
@pytest.mark.skipif(condition=not MATPLOTLIB_EXIST, reason="You don't have matplotlib")
@pytest.mark.skipif(condition=not OPENFERMION_EXIST, reason="You don't have openfermion")
@pytest.mark.parametrize("qh, connectivity, file", [tequila.QubitHamiltonian(openfermion.QubitOperator("X0 X5 Y3", 1)), [(0, 0), (0, 5), (3, 3)], "test_file3"])
def test_visualize_with_connectivity(qh, connectivity, file):
    fig = visualize(qh, connectivity=connectivity, file_path=file)
    assert plt.fignum_exists(fig.number)
