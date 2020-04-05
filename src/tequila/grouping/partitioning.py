import numpy as np
from tequila.grouping import BinaryHamiltonian
from tequila import TequilaException
#from tequila.hamiltonian import QubitHamiltonian, PauliString

class AugmentedBinaryHamiltonian(BinaryHamiltonian):
    def __init__(self, binary_terms):
        pass

    def anti_commutativity_matrix(self):
        n = self.n_qubit
        matrix = np.array(self.get_binary())
        gram = np.block([[np.zeros((n,n)), np.eye(n)], [np.eye(n), np.zeros((n,n))]])
        return matrix @ gram @ matrix.T % 2

    def commuting_groups(self, method='LF'):
        coefficients = self.get_coeff()
        words = self.get_binary()
        n = self.n_term
        cg = self.anti_commutativity_matrix()

        def largest_first():
            rows = cg.sum(axis=0)
            ind = np.argsort(rows)[::-1]
            m = cg[ind,:][:,ind]
            colors = dict()
            c = np.zeros(n, dtype=int)
            k = 0 #color
            for i in range(n):
                neighbors = np.argwhere(m[i,:])
                colors_available = set(np.arange(1, k+1)) - set(c[[x[0] for x in neighbors]])
                term = (words[ind[i]], coefficients[ind[i]])
                if not colors_available:
                    k += 1
                    c[i] = k
                    colors[c[i]] = [term]
                else:
                    c[i] = min(list(colors_available))
                    colors[c[i]].append(term)
            return colors

        def recursive_largest_first():
            colors = dict()
            c = np.zeros(n, dtype=int)
            # so, the preliminary work is done
            uncolored = set(np.arange(n))
            colored = set()
            k = 0
            while uncolored:
                decode = np.array(list(uncolored))
                k += 1
                m = cg[:, decode][decode, :]
                v = np.argmax(m.sum(axis=1))
                colored_sub = {v}
                uncolored_sub = set(np.arange(len(decode))) - {v}
                n0 = n_0(m, colored_sub)#vertices that are not adjacent to any colored vertices
                n1 = uncolored_sub - n0
                while n0:
                    m_uncolored = m[:,list(n1)][list(n0),:]
                    v = list(n0)[np.argmax(m_uncolored.sum(axis=1))]
                    colored_sub.add(v) #stable
                    uncolored_sub -= {v} #stable
                    n0 = n_0(m, colored_sub)
                    n1 = uncolored_sub - n0 #stable
                indices = decode[list(colored_sub)]
                c[indices] = k  # stable
                colors[k] = [(words[i], coefficients[i]) for i in indices] # stable
                colored |= set(indices)
                uncolored = set(np.arange(n)) - colored
            return colors

        def n_0(m, colored):
            m_colored = m[list(colored)]
            l = m_colored[-1]
            for i in range(len(m_colored)-1):
                l += m_colored[i]
            white_neighbors = np.argwhere(np.logical_not(l))
            return set([x[0] for x in white_neighbors]) - colored

        if method == 'LF':
            colors = largest_first()
        elif method == 'RLF':
            colors = recursive_largest_first()
        else:
            raise TequilaException(f"There is no algorithm {method}")
        return [BinaryHamiltonian(value) for key, value in colors.items()]