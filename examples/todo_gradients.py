from openvqe.circuit import gates, Variable
from openvqe.simulator.simulator_cirq import SimulatorCirq
from openvqe.simulator.simulator_qiskit import SimulatorQiskit
from openvqe.simulator.simulator_qulacs import SimulatorQulacs
from openvqe.hamiltonian import paulis
from openvqe.objective import Objective
from openvqe.circuit.gradient import grad
from numpy import pi
from matplotlib import pyplot as plt

"""
This works fine
"""
steps = 100
start = 0.0
end = 2*pi
H = paulis.X(0)
E =[]
dE = []
simulator = SimulatorCirq()
for s in range(steps):
    angle = Variable(name="angle", value=start + s/steps*(end-start))
    U = gates.Ry(target=0, angle=angle)
    O = Objective(observable=H, unitaries=U)
    E.append(simulator.simulate_objective(O))
    for k,v in grad(O).items():
        dE.append(simulator.simulate_objective(v))

plt.title("Should be Fine")
plt.plot(E, label="E")
plt.plot(dE, label="dE")
plt.legend()
plt.show()

"""
This Fails
"""
steps = 100
start = 0.0
end = 2*pi
H = paulis.X(0)
E =[]
dE = []
simulator = SimulatorCirq()
for s in range(steps):
    angle = Variable(name="angle", value=start + s/steps*(end-start))
    U = gates.Ry(target=0, angle=-angle) # here is the sign
    O = Objective(observable=H, unitaries=U)
    E.append(simulator.simulate_objective(O))
    for k,v in grad(O).items():
        dE.append(simulator.simulate_objective(v))

plt.title("Fails")
plt.plot(E, label="E")
plt.plot(dE, label="dE")
plt.legend()
plt.show()
