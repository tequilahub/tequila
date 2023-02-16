import numpy as np
import tequila as tq

from tequila.apps.qcc.qcc import IterativeQCC


molecule = tq.chemistry.Molecule(geometry="H 0.0 0.0 0.0\nLi 0.0 0.0 1.5", basis_set="sto-3g")

H = molecule.make_hamiltonian()

E_fci = molecule.compute_energy("fci")
print('FCI energy: {}'.format(E_fci))

iQCC = IterativeQCC(molecule)

for i in range(10):
    iQCC.do_iteration(n_gen=1)
    print('iQCC energy at iter {}: {}'.format(i, iQCC.energy))

print('\nEnergy errors for the iQCC iterations:')
for i in range(10):
    print('iter {}: {}'.format(i, iQCC.iteration_energies[i] - E_fci))
