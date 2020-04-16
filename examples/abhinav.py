import tequila as tq


# get rid of the jax GPU/CPU warnings
import warnings

import tequila.simulators.simulator_api

warnings.filterwarnings("ignore", module="jax")

# if you want to transform them, you need to explicitly declare variables
z = tq.Variable("z")

U  = tq.gates.Rx(target=0, angle=z.apply(tq.numpy.arcsin))
U += tq.gates.Rz(target=0, angle=z.apply(tq.numpy.arccos))
U += tq.gates.Ry(target=0, angle="theta_1")
U += tq.gates.Rx(target=1, angle=z.apply(tq.numpy.arcsin))
U += tq.gates.Rz(target=1, angle=tq.Variable("z").apply(tq.numpy.arccos)) # just to give an alternative
U += tq.gates.Ry(target=1, angle="theta_2")
U += tq.gates.ExpPauli(paulistring="X(0)X(1)", angle="theta_3")
# we use same convention as for rotations
# so the last gate is: exp(-i*angle/2 X(0)X(1))
# if you want to get rid of this you can do
# U += tq.gates.ExpPauli(paulistring="X(0)X(1)", angle=-2.0*tq.Variable("theta_3"))
# or
# U += tq.gates.ExpPauli(paulistring=tq.PauliString.from_string("X(0)X(1)", coeff=-2.0), angle="theta_3")

# in case you forgot the names of the variables in your circuit:
var_names = U.extract_variables()
print("variables are: ", var_names)

# Measurement Hamiltonian
H = tq.paulis.I() - tq.paulis.Z(0)

# Form an expectation value
E = tq.ExpectationValue(U=U, H=H)
#
# lets run it
values = {"z":1.0, "theta_1":1.0, "theta_2": 1.0, "theta_3": 1.0}
result = tequila.simulators.simulator_api.simulate(E, samples=100, variables=values)

print(result)

# if you don't like expectationvalues (can't optimize then though)
distribution = tequila.simulators.simulator_api.simulate(U + tq.gates.Measurement(0), samples=100, variables=values)
# prints out like wavefunction (same module actually)
print(distribution)
# access result like dictionaries
print(distribution.items())

# compute gradients
dE_z = tq.grad(E, "z")
print(dE_z)
# evaluate
# should give nan, since we set the value for z to 1 where sin^-1 is not differentiable
result = tequila.simulators.simulator_api.simulate(dE_z, samples=100, variables=values)
print(result)

# lets use a sane value
values["z"] = 0.5
result = tequila.simulators.simulator_api.simulate(dE_z, samples=100, variables=values)
print(result)

# lets do something else in the end
objective = (E**2-1).apply(tq.numpy.exp)
result = tequila.simulators.simulator_api.simulate(objective, samples=100, variables=values)
print(result)
result = tequila.simulators.simulator_api.simulate(objective, samples=None, variables=values)
print(result)

# just to show how the optimizer works (guess this is not what you actually want to optimize)
values = {"z": 0.5, "theta_1": 1.0, "theta_2": 1.0, "theta_3": 1.0}
result = tq.optimizer_scipy.minimize(objective=E, initial_values=values, method="bfgs", silent=False)
result.history.plot("energies")
result.history.plot("angles")




