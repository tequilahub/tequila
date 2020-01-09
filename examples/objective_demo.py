import tequila as tq
from jax import numpy as numpy
from tequila.circuit.gradient import grad
from matplotlib import pyplot as plt

def my_trafo(E):
    return E**2

simulator = tq.simulators.SimulatorQulacs()

# create a simple expectationvalue (this is actually a wrapped objective)
U = tq.gates.Ry(target=0, angle=tq.Variable(name="a", value=2.0))
H = tq.paulis.X(qubit=0)
E = tq.Objective.ExpectationValue(H=H, U=U)

# three ways of creating an objective out of it
F1 = E**2
F2 = tq.Objective(args=E.args, transformation=my_trafo)
F3 = tq.Objective(args=E.args, transformation=lambda E: E**2)

print(simulator(E))
print(simulator(F1))
print(simulator(F2))
print(simulator(F3))

# compute the gradients w.r.t a variable
dE = grad(obj=E, variable="a")
print(simulator(dE))

dF1 = grad(obj=F1, variable="a")
print(simulator(dF1))

dF2 = grad(obj=F2, variable="a")
print(simulator(dF2))


def my_test(objective, title="title", steps=25):
    print("test ", objective)
    gradient = grad(objective, variable="a")
    print("gradient ", gradient)
    O = []
    dO = []
    for step in range(steps):
        var = 0.0 + step / steps * 2 * numpy.pi
        objective.update_variables({"a": var})
        gradient.update_variables({"a": var})
        O.append(tq.simulators.SimulatorQulacs()(objective))
        dO.append(tq.simulators.SimulatorQulacs()(gradient))

    fig = plt.figure()
    plt.plot(O, label="O")
    plt.plot(dO, label="dO")
    plt.title(title)
    plt.legend()
    plt.show()

my_test(objective=E, title="F = E")  # grad has three expectationvalues but df/dE = const 1, can we identify constant functions in Jax to avoid that? For pure expectationvalues there is an easy solution, not included here to demonstrate this
my_test(objective=E ** 2, title="F = E**2")  # grad has three expectationvalues, and actually three are needed
my_test(objective=E * E, title="F = E*E")  # grad has 8 expectationvalues -> currently a bad way of initializing if the expectationvalues are the same
my_test(objective=2.0 * E, title="F = 2E")  # grad has three expectationvalues -> same as at first
my_test(objective=E + E, title="F = E + E")  # grad has 8 expectationvalues -> currently bad way of initializing if the epectationvalues are the same

my_test(objective=2.0 * E + E ** 2, title="F= 2.0*E + E**2")
my_test(objective=tq.Objective(args=E.args, transformation=lambda E0: 2.0 * E0 + E0 ** 2),
        title="F=2.0*E + E**2")

N = tq.Objective.ExpectationValue(H=None, U=U)
print(tq.simulators.SimulatorQulacs()(N))
my_test(objective=E * N ** (-1), title="F= <U|H|U>/<U|U>")
my_test(objective=N ** (-1) * E, title="F= <U|H|U>/<U|U>")
my_test(objective=tq.Objective(args=E.args + N.args,
                               transformation=lambda E0, E1: numpy.true_divide(E0, E1)), title="F= <U|H|U>/<U|U>")
my_test(objective=tq.Objective(args=E.args + N.args,
                               transformation=lambda E0, E1: E0/E1), title="F= <U|H|U>/<U|U>")

# something weird(er) in the end
my_test(objective=tq.Objective(args=E.args, transformation=lambda E: numpy.exp(-E ** 2)), title="F= exp(-E**2)", steps=100)
my_test(objective=numpy.e ** (-E ** 2), title="F= exp(-E**2)", steps=100)

U2 = tq.gates.Rx(target=0, angle=tq.Variable(name="a", value=2.0))
H2 = tq.paulis.Y(qubit=0)
E2 = tq.Objective.ExpectationValue(H=H2, U=U2)

my_test(objective=(numpy.e ** (-E ** 2) + E2 / 2 + 1.234) ** 2, title="F= whatever", steps=100)