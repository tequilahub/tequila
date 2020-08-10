
import tequila as tq
import numpy
import openfermion as of

R = 2.0
transformation = "symmetry_conserving_bravyi_kitaev"

# lets load the Hamiltonian
H_of=of.utils.load_operator(file_name="h2_{}_4so_{}".format(R,transformation), data_directory=".")
H = tq.QubitHamiltonian.from_openfermion(H_of)

# Since the Hamiltonian is small
# lets get the true ground state
val, vec = numpy.linalg.eigh(H.to_matrix())
energy = val[0]
wfn = tq.QubitWaveFunction(vec[:,0])

# lets create a toy circuit
# needs to be adapted if the transformation is changed
U = tq.gates.Rx(target=0, angle=("x", 0))
U += tq.gates.Rx(target=1, angle=("x",1))
U += tq.gates.Phase(target=1, control=0, phi="phase")
U += tq.gates.Rx(target=0, angle=("x",2))
U += tq.gates.Rx(target=1, angle=("x",3))
U += tq.gates.Phase(target=1, control=0, phi="phase")

print("VQE Ansatz is:\n", U)

# measurement optimization does not affect simulation without finite samples
# but might be useful in the future
# more information in the tequila tutorials on github under MeasurementGroups
# leaving that here, let me know if you have questions
E = tq.ExpectationValue(U=U, H=H, optimize_measurements=False)

# optimize the toy circuit
# further options for minimze: samples=..., backend=..., noise=...
# see tequila tutorials on github for more information
result = tq.minimize(method="nelder-mead", objective=E)
# simulate the optimized wavefunction
sim_wfn = tq.simulate(U, variables=result.angles)
print("VQE result {:2.8f} :".format(result.energy))
print("True result {:2.8f}:".format(energy))
print("True wfn  :", wfn)
print("Ansatz wfn:", sim_wfn)
print("fidelity: {:2.8f}".format(numpy.abs(sim_wfn.inner(wfn))**2))
# fidelity as quantum circuit
P = tq.paulis.Projector(wfn=wfn)
F = tq.ExpectationValue(H=P, U=U)
# with this options its perfect simulation, so the same result as above
fidelity = tq.simulate(F, variables=result.angles)
print("measured fidelity: {:2.8f}".format(fidelity))




