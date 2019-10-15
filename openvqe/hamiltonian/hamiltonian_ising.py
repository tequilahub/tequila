"""
(Quantum) Ising Hamiltonian (with) without transversal Field
"""
from openvqe.hamiltonian import QubitHamiltonian, paulis
import networkx as nx
import matplotlib.pyplot as plt

PX = paulis.X
PY = paulis.Y
PZ = paulis.Z


class IsingHamiltonian(QubitHamiltonian):

    def __init__(self, connections=None, g=None):
        """
        :param n_qubits: number of sites
        :param g: strenth of the transversal field, default is None
        """
        self._n_qubits = None
        self._g = g
        if connections is None:
            self._connections = None
        else:
            self.initialize_from_graph(connections=connections, g=g)

    @staticmethod
    def make_spin_chain(n_qubits, g=None, start=0, periodic_bc=True):
        """
        Use as constructor
        :return: Ising Hamiltonian for a spin chain from i=start to i=start+n_qubits-1
        """
        connections = {}
        for i in range(start, start + n_qubits - 1):
            connections[i] = [i + 1]
        if periodic_bc:
            connections[start + n_qubits - 1] = [start]

        return IsingHamiltonian(connections=connections, g=g)

    @staticmethod
    def make_spin_grid(n_x, n_y, g=None, periodic_x=True, periodic_y=True, map=None):

        if g is not None:
            raise NotImplementedError("g!=None for 2D grid not yet defined")

        def default_map(x, y):
            return x*n_x + y

        def get_connections(x,y):
            x_connections = []
            y_connections = []
            x2 = (x+1)%n_x
            y2 = (y+1)%n_y
            if x<n_x-1 or periodic_x:
                x_connections += [map(x2,y)]
            if y<n_y-1 or periodic_y:
                y_connections += [map(x,y2)]
            return x_connections + y_connections

        if map is None:
            map = default_map

        connections = {}
        for x in range(n_x):
            for y in range(n_y):
                connections[map(x, y)] = get_connections(x,y)

        return IsingHamiltonian(connections=connections, g=g)

    def initialize_from_graph(self, connections=None, g=None):
        """
        :param connections: Adictionary with: keys = nodes/vertices, values: List of other nodes (edges)
        :param g: interaction strength
        :return: Ising Hamiltonian with interactions between all connections in the graph
        """

        self._n_qubits = max(connections.keys()) + 1
        self._g = g
        self._connections = connections

        H = 0 * QubitHamiltonian()

        for node, edges in connections.items():
            for edge in edges:
                H += 0.5 * PZ(qubit=node) * PZ(qubit=edge)

        if g is not None and g is not 0:
            for i in connections.keys():
                H += g * PX(qubit=i)

        self.hamiltonian = H

        return self

    @property
    def g(self):
        return self._g

    @g.setter
    def g(self, other):
        self._g = other
        self.initialize_from_graph(connections=self.connections, g=other)
        return self

    @property
    def connections(self):
        return self._connections

    @connections.setter
    def connections(self, other):
        self._connections = other
        self.initialize_from_graph(connections=other, g=self.g)
        return self

    @property
    def n_qubits(self):
        return self._n_qubits

    def plot_connections(self, title):
        G = nx.Graph()
        for node, edges in self.connections.items():
            G.add_node(node)
            for edge in edges:
                G.add_edge(*(node,edge))

        plt.title(title)
        nx.draw(G, with_labels=True, font_weight='bold', title=title)
        plt.show()



if __name__ == "__main__":
    print("Making periodic Ising chain starting from 1 and with lenth 4:")
    H = IsingHamiltonian.make_spin_chain(n_qubits=4, start=1)
    print("connection graph:", H.connections)
    print(H)
    print("\n")

    print("Making periodic Ising chain with transversal field of strength 1 starting from 1 and with lenth 4:")
    H = IsingHamiltonian.make_spin_chain(n_qubits=4, start=1, g=1)
    print("connection graph:", H.connections)
    print(H)
    print("\n")

    print("Making non-periodic Ising chain starting from 1 and with lenth 4:")
    H = IsingHamiltonian.make_spin_chain(n_qubits=4, start=1, periodic_bc=False)
    print("connection graph:", H.connections)
    print(H)
    print("\n")

    print("Making non-periodic Ising chain with transversal field of strength 1 starting from 1 and with lenth 4:")
    H = IsingHamiltonian.make_spin_chain(n_qubits=4, start=1, g=1, periodic_bc=False)
    print("connection graph:", H.connections)
    print(H)
    print("\n")

    print("Making 2D Ising grid, periodic in x direction:")
    H = IsingHamiltonian.make_spin_grid(n_x=3, n_y=3, g=None, periodic_x=True, periodic_y=False)
    print("connection graph:", H.connections)
    print(H)
    print("\n")
    H.plot_connections(title="2D Ising periodic in X")

    print("Making 2D Ising grid, full periodic")
    H = IsingHamiltonian.make_spin_grid(n_x=3, n_y=3, g=None, periodic_x=True, periodic_y=True)
    print("connection graph:", H.connections)
    print(H)
    print("\n")
    H.plot_connections(title="2D Ising non-periodic")

    print("Making 2D Ising grid, full periodic")
    H = IsingHamiltonian.make_spin_grid(n_x=3, n_y=3, g=None, periodic_x=False, periodic_y=False)
    print("connection graph:", H.connections)
    print(H)
    print("\n")
    H.plot_connections(title="2D Ising non-periodic")
