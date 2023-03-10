import importlib.util

import tequila
from typing import Optional, List
import matplotlib.pyplot as plt
import matplotlib.figure as figure
import networkx


def visualize(qubit_hamiltonian: tequila.QubitHamiltonian,
              circuit: Optional[tequila.QCircuit] = None,
              connectivity: Optional[List[tuple]] = None,
              file_path: Optional[str] = None) -> figure.Figure:
    """
    Precondition:
    The maximum number of qubits is 10 at the moment (Feb24, 2023)

    Post condition:
    The graph of the qubit_hamiltonian is displayed
    or is stored in file_path

    One thing to note is that if you are using command-line interface,
    the plot might not be successfully shown

    , so it's better idea to save it as an image.
    === parameters ===
    qubit_hamiltonian: A QubitHamiltonian representation of pauli operators
    circuit: A QCircuit that corresponds to the
    connectivity: networkx.Graph.edges that show the connectivity of qubits
    A nested list should be provided for this argument.
    file_name: str format of file name
    to which the plotted graph will be exported. The file will be a png format

    === return ===
    None

    === sample usages ===
    >>> visualize(
    >>> tequila.QubitHamiltonian(openfermion.QubitOperator("X0 X5 Y3", 1)))
    *** A graph with nodes 0 and 5 having colour red
    and node 3 having colour green ***
    >>> visualize(
    >>> tequila.QubitHamiltonian(openfermion.QubitOperator("X0 X5 Y3", 1)),
    >>> circuit=gates.X(0) + gates.CNOT(0, 5) + gates.Y(3))
    *** A graph with nodes 0 and 5 having color red and
    node 3 having colour green with edge 0 and 5 exists ***
    >>> visualize(
    >>> tequila.QubitHamiltonian(openfermion.QubitOperator("X0 X5 Y3", 1)),
    >>> connectivity=[(0, 0), (0, 5), (3, 3)])
    *** A graph with nodes 0 and 5 having color red and
    node 3 having colour green with edge 0 and 5 exists ***
    >>> visualize(
    >>> tequila.QubitHamiltonian(openfermion.QubitOperator("X0 X5 Y3", 1)),
    >>> connectivity=[(0, 0), (0, 5), (3, 3)],
    >>> file_name="test_system")
    *** Exported an image of a graph with nodes 0 and 5 having color red and
    node 3 having colour green with edge 0 and 5 exists to test_system.png ***
    """
    if importlib.util.find_spec("networkx") is None:
        raise tequila.TequilaException("networkx module is not in the system")
    if importlib.util.find_spec("matplotlib") is None:
        raise tequila.TequilaException("matplotlib module is not in the system")

    if(len(qubit_hamiltonian.qubit_operator.terms)) != 1:
        raise tequila.TequilaException("The input qubit_operator"
                                       " should have length 1")

    qh = next(iter(qubit_hamiltonian.qubit_operator.terms))
    graph = networkx.Graph()
    graph, pos = _draw_basics(graph)
    if circuit is None and connectivity is None:
        graph, pos = _visualize_helper(qh, graph, pos)

    elif connectivity is None:
        graph, pos = _visualize_helper(qh, graph, pos, list(circuit.to_networkx().edges))

    elif circuit is None:
        graph, pos = _visualize_helper(qh, graph, pos, connectivity)

    if file_path is None:
        return plt.figure()
    if file_path is not None:
        plt.savefig(file_path+".png", format="PNG")
        return plt.figure()


def _draw_basics(graph):
    """
    A helper function for visualize() function.
    This function sets up the basic graph to be used.
    """
    length = 10
    for site in range(length):
        graph.add_node(site)
    pos = networkx.spring_layout(graph, seed=3113794652)
    networkx.draw_networkx_nodes(
        graph, pos,  node_color="#FFFFFF", edgecolors="#000000")
    return graph, pos


def _visualize_helper(qh_list: List[tequila.QubitHamiltonian],
                      graph, pos, connection: Optional[list] = None):
    """
    A helper function for visualize() function.
    This function visualize the graph based on the QubitHamiltonian and connection
    """
    for pair in qh_list:
        if pair[1] == 'X':
            networkx.draw_networkx_nodes(
                graph, pos,
                nodelist=[pair[0]], node_color="tab:red")
        elif pair[1] == 'Y':
            networkx.draw_networkx_nodes(
                graph, pos,
                nodelist=[pair[0]], node_color="tab:green")
        elif pair[1] == 'Z':
            networkx.draw_networkx_nodes(
                graph, pos,
                nodelist=[pair[0]], node_color="tab:blue")
    if connection is not None:
        for edge in connection:
            if edge[0] != edge[1]:
                networkx.draw_networkx_edges(
                    graph, pos, edgelist=[edge], width=5)
    networkx.draw_networkx_labels(
        graph, pos,
        labels={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9})
    return graph, pos
