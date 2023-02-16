import numpy as np
import tequila as tq

from tequila.apps.qcc.qcc import IterativeQCC


import math
import tequila as tq

R = 1.0
angle = math.radians(107.6 / 2)
x = R * math.sin(angle)
y = R * math.cos(angle)

xyz = ''.join(["O 0.0 0.0 0.0\n",
               "H {}  {}  0.0\n".format(-x, y),
               "H {}  {}  0.0".format(x, y)
              ]
             )

basis = '6-31g'
active = {'B1':[0,1], 'A1':[2,3]}

h2o = tq.quantumchemistry.Molecule(geometry=xyz, basis_set = basis, active_orbitals = active)

energy_fci = h2o.compute_energy('fci') #compute the FCI energy for error analysis later
print('FCI energy: {}'.format(energy_fci))

iqcc_solver = IterativeQCC(h2o)

n_iter = 3

for _ in range(n_iter):
    iqcc_solver.do_iteration(n_gen=6)

print('\nL1 norm of gradients at exit: {}'.format(iqcc_solver.grad_norm()))

print('Energy errors (Hartree) relative to the FCI value for the iQCC iterations:')
for idx in range(n_iter):
    print('iter {}: {}'.format(idx, iqcc_solver.energies[idx] - energy_fci))

print(iqcc_solver.energies)
print(iqcc_solver.generators)
print(iqcc_solver.n_terms)
