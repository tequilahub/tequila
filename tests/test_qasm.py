from tequila.circuit.qasm import export_open_qasm, import_open_qasm, import_open_qasm_from_file
from tequila.circuit.gates import *
from tequila.simulators.simulator_api import simulate
import numpy
import os
import pytest


@pytest.mark.parametrize(
    "zx_calculus",
    [
        False,
        True,
    ]
)
def test_export_import_qasm_simple(zx_calculus):
    tequila_circuit = H(target=1) + \
                      X(target=1) + \
                      Y(target=0) + \
                      Z(target=2) + \
                      CX(target=3, control=0) + \
                      CY(target=4, control=2) + \
                      CZ(target=5, control=1) + \
                      CNOT(target=3, control=0) + \
                      SWAP(first=0, second=3) + \
                      S(target=1, control=0) + \
                      T(target=1, control=2)

    qasm_code_simple = export_open_qasm(tequila_circuit, zx_calculus=zx_calculus)
    imported_circuit = import_open_qasm(qasm_code=qasm_code_simple)

    wfn1 = simulate(tequila_circuit, backend="symbolic")
    wfn2 = simulate(imported_circuit, backend="symbolic")

    assert (numpy.isclose(wfn1.inner(wfn2), 1.0))


@pytest.mark.parametrize(
    "zx_calculus,variabs",
    [
        (False, [2.8, 5.6, 7.6, 1.8, 4.98, 2.35, 3.12, 6.79]),
        (True,  [1.5, 3.7, 9.2, 3.1, 7.62, 1.98, 8.56, 2.97]),
        (False, [0, 0, 0, 0, 0, 0, 0, 0]),
        (True,  [0, 0, 0, 0, 0, 0, 0, 0]),
        (False, [numpy.pi/12, numpy.pi, numpy.pi/3, numpy.pi/6, numpy.pi*0.95, numpy.pi/2, numpy.pi*2.3, numpy.pi/7]),
        (True,  [numpy.pi/12, numpy.pi, numpy.pi/3, numpy.pi/6, numpy.pi*0.95, numpy.pi/2, numpy.pi*2.3, numpy.pi/7])
    ]
)
def test_export_import_qasm(zx_calculus, variabs):

    variables = {"ang1": variabs[0],
                 "ang2": variabs[1],
                 "ang3": variabs[2],
                 "ang4": variabs[3],
                 "ang5": variabs[4],
                 "ang6": variabs[5],
                 "ang7": variabs[6],
                 "ang8": variabs[7]}

    tequila_circuit = H(target=[0, 1]) + \
                      H(target=0, control=1) + \
                      X(target=1) + \
                      Y(target=0) + \
                      Z(target=2) + \
                      CX(target=3, control=0) + \
                      CY(target=4, control=2) + \
                      CZ(target=5, control=1) + \
                      CNOT(target=3, control=0) + \
                      SWAP(first=0, second=3) + \
                      Rx(target=1, angle="ang1") + \
                      Ry(target=0, angle="ang2") + \
                      Rz(target=2, angle="ang3") + \
                      CRx(target=5, control=8, angle="ang4") + \
                      CRy(target=6, control=9, angle="ang5") + \
                      CRz(target=7, control=0, angle="ang6") + \
                      Phase(control=0, target=1, phi="ang1") + \
                      S(target=1, control=0) + \
                      T(target=1, control=2) + \
                      Rp(paulistring="Y(1)", angle="ang7") + \
                      ExpPauli(paulistring="Z(1)X(2)", control=0, angle="ang8") + \
                      Toffoli(target=2, first=4, second=3)

    qasm_code = export_open_qasm(tequila_circuit, variables=variables, zx_calculus=zx_calculus)
    imported_circuit = import_open_qasm(qasm_code=qasm_code)

    wfn1 = simulate(tequila_circuit, backend="symbolic", variables=variables)
    wfn2 = simulate(imported_circuit, backend="symbolic")

    assert (numpy.isclose(wfn1.inner(wfn2), 1.0))


@pytest.mark.parametrize(
    "zx_calculus,target1,control1,target2,control2",
    [
        (False, 1,            None, 0,      1),
        (True,  1,            None, 0,      1),
        (False, [0, 1],       None, 0,      1),
        (True,  [0, 1],       None, 0,      1),
        (False, [0, 1, 2, 3], 4,    2,      1),
        (True,  [0, 1, 2, 3], 4,    2,      1),
        (False, [1, 2],       4,    [0, 3], 1),
        (True,  [1, 2],       4,    [0, 3], 1)
    ]
)
def test_export_import_qasm_h_ch_gate(zx_calculus, target1, control1, target2, control2):

    tequila_circuit = H(target=target1, control=control1) + H(target=target2, control=control2)

    qasm_code = export_open_qasm(tequila_circuit, zx_calculus=zx_calculus)
    imported_circuit = import_open_qasm(qasm_code=qasm_code)

    wfn1 = simulate(tequila_circuit, backend="symbolic")
    wfn2 = simulate(imported_circuit, backend="symbolic")

    assert (numpy.isclose(wfn1.inner(wfn2), 1.0))


@pytest.mark.parametrize(
    "zx_calculus,string1,string2,angles,steps",
    [
        (False, "1.0 X(1)Z(2) - 0.5 Z(3)X(4)", None,                          [numpy.pi/12],   1),
        (True,  "1.0 X(1)Z(2) - 0.5 Z(3)X(4)", None,                          [numpy.pi/12],   1),
        (False, "1.0 X(1)Z(2) + 0.5 Z(3)X(4)", "1.0 Y(1)X(2) - 0.9 X(2)Z(3)", [5.6, numpy.pi], 1),
        (True,  "1.0 X(1)Z(2) + 0.5 Z(3)X(4)", "1.0 Y(1)X(2) - 0.9 X(2)Z(3)", [5.6, numpy.pi], 1),
        (False, "1.5 Z(2)Z(4) + 0.8 Y(3)X(4)", None,                          [numpy.pi],      1),
        (True,  "1.5 Z(2)Z(4) + 0.8 Y(3)X(4)", None,                          [numpy.pi],      2)
    ]
)
def test_export_import_qasm_trotterized_gate(zx_calculus, string1, string2, angles, steps):

    variables = {"ang1": angles[0]} if string2 is None else {"ang1": angles[0], "ang2": angles[1]}

    g1 = QubitHamiltonian.from_string(string1)
    g2 = None if string2 is None else QubitHamiltonian.from_string(string2)
    tequila_circuit = Trotterized(generators=[g1] if string2 is None else [g1, g2],
                                  angles=["ang1"] if string2 is None else ["ang1", "ang2"],
                                  steps=steps)

    qasm_code = export_open_qasm(tequila_circuit, variables=variables, zx_calculus=zx_calculus)
    imported_circuit = import_open_qasm(qasm_code=qasm_code)

    wfn1 = simulate(tequila_circuit, backend="symbolic", variables=variables)
    wfn2 = simulate(imported_circuit, backend="symbolic")

    assert (numpy.isclose(wfn1.inner(wfn2), 1.0))


@pytest.mark.parametrize(
    "zx_calculus,variabs",
    [
        (False, [2.8, 5.6, 7.6, 1.8, 4.98]),
        (True,  [9.6, 4.1, 6.3, 2.5, 5.62]),
        (False, [numpy.pi*3, numpy.pi, numpy.pi/3, numpy.pi, numpy.pi*0.95]),
        (True,  [numpy.pi*2, numpy.pi, numpy.pi/4, numpy.pi, numpy.pi*0.68])
    ]
)
def test_export_import_qasm_file(zx_calculus, variabs):

    variables = {"ang1": variabs[0],
                 "ang2": variabs[1],
                 "ang3": variabs[2],
                 "ang4": variabs[3],
                 "ang5": variabs[4]}

    tequila_circuit = H(target=[0, 1]) + \
                      Y(target=0) + \
                      Z(target=2) + \
                      CY(target=4, control=2) + \
                      SWAP(first=0, second=3) + \
                      Ry(target=0, angle="ang1") + \
                      Rz(target=2, angle="ang2") + \
                      CRx(target=5, control=8, angle="ang3") + \
                      CRy(target=6, control=9, angle="ang4") + \
                      S(target=1, control=0) + \
                      T(target=1, control=2) + \
                      ExpPauli(paulistring="Y(1)Z(3)", control=0, angle="ang5") + \
                      Toffoli(target=2, first=4, second=3)

    file_name = "test_file_qasm.txt"

    qasm_code = export_open_qasm(tequila_circuit, variables=variables, zx_calculus=zx_calculus,
                                 filename=file_name)
    imported_circuit = import_open_qasm(qasm_code=qasm_code)
    imported_circuit_from_file = import_open_qasm_from_file(filename=file_name)

    wfn1 = simulate(tequila_circuit, backend="symbolic", variables=variables)
    wfn2 = simulate(imported_circuit, backend="symbolic")
    wfn3 = simulate(imported_circuit_from_file, backend="symbolic")

    # remove file
    if os.path.exists(file_name):
        os.remove(file_name)

    assert (numpy.isclose(wfn1.inner(wfn2), 1.0))
    assert (numpy.isclose(wfn1.inner(wfn3), 1.0))
    assert (numpy.isclose(wfn2.inner(wfn3), 1.0))


def test_import_qasm_with_custom_gates():

    openqasmcode = "OPENQASM 2.0;\n" \
                   "include \"qelib1.inc\";\n" \
                   "gate mycustom a,b,c\n" \
                   "{\n" \
                   "cx c,b;\n" \
                   "cx c,a;\n" \
                   "}\n" \
                   "qreg q1[3];\n" \
                   "qreg q2[4];\n" \
                   "creg c[3];\n" \
                   "y q1[1];\n" \
                   "z q2[2];\n" \
                   "mycustom q1[0],q2[0],q1[2];\n" \
                   "h q2[1];\n" \
                   "mycustom q2[3],q1[1],q2[2];\n" \
                   "y q2[1];\n"

    imported_circuit = import_open_qasm(qasm_code=openqasmcode)

    # openqasm   -> tequila qbits
    # qreg q1[3] -> 0, 1, 2
    # qreg q2[4] -> 3, 4, 5, 6

    tequila_circuit = Y(target=1) + Z(target=5) + \
                      CX(target=3, control=2) + CX(target=3, control=0) + \
                      H(target=4) + \
                      CX(target=1, control=5) + CX(target=6, control=5) + \
                      Y(target=4)

    wfn1 = simulate(tequila_circuit, backend="symbolic")
    wfn2 = simulate(imported_circuit, backend="symbolic")

    assert (numpy.isclose(wfn1.inner(wfn2), 1.0))

