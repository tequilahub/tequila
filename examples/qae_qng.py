import tequila as tq
from time import time
import numpy as onp
from tequila.optimizers import optimizer_scipy
from tequila.quantumchemistry import Molecule

def write_xyz(lol,filename):
    with open(filename,'w') as f:
        f.write(str(len(lol))+'\n')
        f.write('demo xyz file maker all distances angstrom\n')
        for block in lol:
            b_string=''
            for i,e in enumerate(block):
                b_string += str(e)
                if i !=3:
                    b_string += ' '
            b_string += '\n'
            f.write(b_string)

sizes = onp.arange(0.3,1.2,0.05)
geos  = []
for val in sizes:
    lol=[]
    lol.append(['H',0.0,0.0,0.0])
    lol.append(['H',0.0,0.0,val])
    geos.append(lol)

filenames=[]
for i,val in enumerate(sizes):
    string='data/h2_%s.xyz'%(str(val))
    filenames.append(string)
    write_xyz(geos[i],string)

def make_state_prep_circuit(geomfile):
    return Molecule(geometry=geomfile,basis_set='sto-3g',transformation='Jordan-Wigner').make_uccsd_ansatz(trotter_steps=1,initial_amplitudes="mp2",parametrized=False)

trash_qubits=[2,3]
H= 0.5 * (tq.paulis.I(1) + tq.paulis.Z(1))
for q in trash_qubits:
    H *= 0.5 * (tq.paulis.I(q) + tq.paulis.Z(q))
s=time()
state_prep_circuits = [make_state_prep_circuit(file) for file in filenames]
e=time()
print('quantum chemistry finished in ',str(e-s), ' seconds.')
variational = tq.gates.Rx(angle="q", target=0) \
                + tq.gates.Rx(angle="b", target=1) \
                + tq.gates.CNOT(target=3, control=1) \
                + tq.gates.CNOT(target=2, control=0) \
                + tq.gates.CNOT(target=1, control=0)

objective = tq.ExpectationValue(H=H, U=state_prep_circuits[0] + variational)/len(filenames)
for u in state_prep_circuits[1:]:
    objective+=tq.ExpectationValue(H=H, U=u + variational)/len(filenames)
initial_values = {"q": 2.0, "b": 2.0}
s=time()
result = optimizer_scipy.minimize(objective=objective,qng=True,initial_values=initial_values)
e=time()
print('optimizing took ',e-s, ' seconds.')
result.history.plot()


result.history.plot(property='gradients', key='a')  # if no key is given it will plot all of them
result.history.plot(property='gradients', key='n')
result.history.plot(property='angles', key='a')
result.history.plot(property='angles', key='b')
# combine plots
result.history.plot(property='angles', key=['a', 'b'])
result.history.plot(property=['angles', 'energies'], key=['a', 'b'])

# plot other results
print("final angles are:\n", result.angles)
print("final energy is :\n", result.energy)
print("iterations :", result.history.iterations)
# some intuitive ways to deal with the history
# evolution of angle 'a'
#all_angles_a = result.history.extract_angles(key='v_0')