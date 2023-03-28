import pytest
from tequila.circuit.visualize import visualize
import sys
import tequila
import os

if 'networkx' in sys.modules:
    NETWORKX_EXIST = True
else:
    NETWORKX_EXIST = False

if 'matplotlib' in sys.modules:
    MATPLOTLIB_EXIST = True
    import matplotlib.pyplot as plt
else:
    MATPLOTLIB_EXIST = False
    import matplotlib.pyplot as plt


@pytest.mark.skipif(condition=not NETWORKX_EXIST, reason="You don't have networkx")
@pytest.mark.skipif(condition=not MATPLOTLIB_EXIST, reason="You don't have matplotlib")
@pytest.mark.parametrize("qh", [tequila.QubitHamiltonian("X(0)X(5)Y(3)")])
def test_visualize_with_qubit_hamiltonian(qh):
    fig = visualize(qh)
    assert plt.fignum_exists(fig.number)


@pytest.mark.skipif(condition=not NETWORKX_EXIST, reason="You don't have networkx")
@pytest.mark.skipif(condition=not MATPLOTLIB_EXIST, reason="You don't have matplotlib")
@pytest.mark.parametrize(
    "qh", [tequila.QubitHamiltonian("X(0)X(5)Y(3)")
          + tequila.QubitHamiltonian("X(0)Y(3)")])
def test_visualize_with_invalid_qubit_hamiltonian(qh):
    with pytest.raises(Exception) as e_info:
        visualize(qh)

@pytest.mark.skipif(condition=not NETWORKX_EXIST, reason="You don't have networkx")
@pytest.mark.skipif(condition=not MATPLOTLIB_EXIST, reason="You don't have matplotlib")
@pytest.mark.parametrize("qh, circuit", [(tequila.QubitHamiltonian("X(0)X(5)Y(3)"), tequila.gates.X(0) + tequila.gates.CNOT(0, 5) + tequila.gates.Y(3))])
def test_visualize_with_circuit(qh, circuit):
    fig = visualize(qh, circuit=circuit)
    assert plt.fignum_exists(fig.number)


@pytest.mark.skipif(condition=not NETWORKX_EXIST, reason="You don't have networkx")
@pytest.mark.skipif(condition=not MATPLOTLIB_EXIST, reason="You don't have matplotlib")
@pytest.mark.parametrize("qh, file", [(tequila.QubitHamiltonian("X(0)X(5)Y(3)"), "test_file1")])
def test_visualize_with_filepath(qh, file):
    fig = visualize(qh, file_path=file)
    assert plt.fignum_exists(fig.number)
    if os.path.isfile(file+".png"):
        os.remove(file+".png")


@pytest.mark.skipif(condition=not NETWORKX_EXIST, reason="You don't have networkx")
@pytest.mark.skipif(condition=not MATPLOTLIB_EXIST, reason="You don't have matplotlib")
@pytest.mark.parametrize("qh, circuit, file", [(tequila.QubitHamiltonian("X(0)X(5)Y(3)"), tequila.gates.X(0) + tequila.gates.CNOT(0, 5) + tequila.gates.Y(3), "test_file2")])
def test_visualize_with_circuit_and_file(qh, circuit, file):
    fig = visualize(qh, circuit=circuit, file_path=file)
    assert os.path.isfile(file+".png") and not plt.fignum_exists(fig)
    if os.path.isfile(file+".png"):
        os.remove(file+".png")


@pytest.mark.skipif(condition=not NETWORKX_EXIST, reason="You don't have networkx")
@pytest.mark.skipif(condition=not MATPLOTLIB_EXIST, reason="You don't have matplotlib")
@pytest.mark.parametrize("qh, connectivity, file", [(tequila.QubitHamiltonian("X(0)X(5)Y(3)"), [(0, 0), (0, 5), (3, 3)], "test_file3")])
def test_visualize_with_connectivity(qh, connectivity, file):
    fig = visualize(qh, connectivity=connectivity, file_path=file)
    assert plt.fignum_exists(fig.number)
    if os.path.isfile(file+".png"):
        os.remove(file+".png")
