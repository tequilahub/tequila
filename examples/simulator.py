from tequila import simulators, gates, paulis, Objective,ExpectationValue

Simulator = simulators.pick_simulator(samples=100, demand_full_wfn=True)
simulator = Simulator()

U = gates.X(0) + gates.Ry(target=1, angle=2.0) + gates.CNOT(0, 1)
H = paulis.X(0) + paulis.Y(1)
O = Objective(expectationvalues=[ExpectationValue(U=U,H=H)])

U += gates.Measurement([0,1])
wfn = simulator(U, samples=100).counts
E = simulator(O, samples=60000)

print("wfn=", wfn)
print("E=", E)
