from tequila.circuit.gradient import grad
from tequila import Variable, gates, paulis, simulators, Objective
from numpy import pi

print("Weights are inconsistent")

a = Variable(name='a', value=pi/2)
U = gates.X(target=1) + gates.Ry(target=0, angle=a)
test = grad(U)['a']
print(test)

a = Variable(name='a', value=pi/2)
U = gates.X(target=1) + gates.Ry(target=0, angle=-a)
test = grad(U)['a']
print(test)

H = paulis.X(0)
a = Variable(name='a', value=0.0)
b = Variable(name='b', value=0.0)
U = gates.Ry(target=0, angle=a/2) + gates.Ry(target=0, angle=b/2)
U.update_variables({'a': 0.0})

E = simulators.SimulatorQulacs().simulate_expectationvalue(E=Objective(unitaries=U, observable=H))
dO = grad(Objective(unitaries=U, observable=H))['a']
dE = simulators.SimulatorQulacs().simulate_expectationvalue(E=dO)
print("E=", E)
print("dE=", dE)

E = simulators.SimulatorCirq().measure_objective(objective=Objective(unitaries=U, observable=H), samples=1000)
dO = grad(Objective(unitaries=U, observable=H))['a']
dE = simulators.SimulatorCirq().measure_objective(objective=dO, samples=1000)
print("E=", E)
print("dE=", dE)
print("dO:", dO)
