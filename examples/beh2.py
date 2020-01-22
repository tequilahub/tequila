import tequila as tq
import numpy as np
import pickle

"""
Demo for the optimization of BeH2/Sto-3g
Or
It is really that hard to beat simple MP2
"""

molecule = tq.chemistry.Molecule(basis_set="sto-3g", geometry="data/beh2.xyz", backend="psi4")
print(molecule)

mp2 = molecule.compute_mp2_amplitudes()
print(mp2.export_parameter_dictionary())
print("RHF energy = ", molecule.molecule.hf_energy)
print("MP2 energy = ", molecule.molecule.mp2_energy)

ccsd = molecule.compute_ccsd_amplitudes()

# sort MP2 amplitudes
tmp2 = dict(sorted(mp2.export_parameter_dictionary().items(), key=lambda x: x[1]))
tccsd = dict(sorted(ccsd.export_parameter_dictionary().items(), key=lambda x: x[1]))
print("Sorted Mp2 Amplitudes\n", tmp2)
print("Sorted CCSD Amplitudes\n", tccsd)

# group MP2 amplitudes of same size together
t_groups = []
current_group = None
for k, v in tmp2.items():
    if current_group is None:
        current_group = [k]
        current_val = v
    elif np.isclose(current_val, v):
        current_group.append(k)
    else:
        t_groups.append(current_group)
        current_group = [k]
        current_val = v
t_groups.append(current_group)

print("created {} groups".format(len(t_groups)))

H = molecule.make_hamiltonian()
U = molecule.prepare_reference()
variables = {}

hf_energy = -15.560122075419443
history = tq.optimizers.OptimizerHistory()
for iteration, t_group in enumerate(t_groups):
    for t in t_group:
        op = molecule.make_excitation_operator(indices=[(t.name[0], t.name[1]), (t.name[2], t.name[3])])
        U += tq.gates.Trotterized(generators=[op], angles=[2.0*t], steps=1)
        variables[t] = 0.0

    O = tq.ExpectationValue(H=H, U=U)
    print("Starting Macro-Iteration {}".format(iteration))
    print(O)
    print("{} active qubits: {}".format(len(U.qubits), U.qubits))
    options = {'gtol': 1e-02, 'maxiter': 3, 'disp': True}
    result = tq.optimizer_scipy.minimize(objective=O, method_options=options, method='bfgs', backend="qulacs", initial_values=variables, tol=1.e-3, silent=False)
    print("Macro-Iteration ", iteration, ":")
    print("energy {energy:6.4f}".format(energy=result.energy))
    print("angles: ", result.angles)
    variables = result.angles
    history += result.history
    if iteration == 1:
        break

print("Computations Ended:")
print("RHF  energy = {:8.5f} ", molecule.molecule.hf_energy)
print("MP2  energy = {:8.5f} ", molecule.molecule.mp2_energy)
print("CCSD energy = {:8.5f} ", -15.594728547924952)
print("FCI  energy = {:8.5f} ", -15.59512472620611)
print("VQE  energy = {:8.5f} ",  result.energy + hf_energy)
print(history.energies)

with open("history.pickle", "wb") as f:
    pickle.dump(result.history, f, pickle.HIGHEST_PROTOCOL)

history.plot("energies", filename="energies.pdf")
history.plot("gradients", filename="gradients.pdf")
history.plot("angles", filename="angles.pdf")


