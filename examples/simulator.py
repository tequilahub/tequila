from tequila import simulate, gates, paulis, ExpectationValue




U = gates.X(0) + gates.Ry(target=1, angle=2.0) + gates.CNOT(0, 1)
H = paulis.X(0) + paulis.Y(1)
O = ExpectationValue(U=U, H=H)
U += gates.Measurement([0,1])
wfn = simulate(U, samples=None)
E = simulate(O, samples=60000)

print("wfn=", wfn)
print("E=", E)
