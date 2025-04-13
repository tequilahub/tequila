import tequila as tq


U = tq.gates.Ry("a",0)
U+= tq.gates.Ry("b",6)
H = tq.paulis.X(0)
H+= tq.paulis.X(6)
E = tq.ExpectationValue(H=H,U=U)
E1 = tq.compile(E,backend="spex")
E2 = tq.compile(E,backend="qulacs")
for a in [1.5,2.0,1.0]:
    print(a)
    print(E1({"a":a,"b":a}))
    print(E2({"a":a,"b":a}),"\n")