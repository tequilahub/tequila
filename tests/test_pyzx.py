from tequila.circuit.pyzx import convert_to_pyzx, convert_from_pyzx
from tequila.circuit.gates import *
from tequila.simulators.simulator_api import simulate
from tequila import TequilaException
import numpy
import pytest
HAS_PYZX = True
try:
    import pyzx
    HAS_PYZX = True
except ImportError:
    HAS_PYZX = False


@pytest.mark.skipif(not HAS_PYZX,
                    reason="Pyzx package not installed, test_convert_to_from_pyzx_simple not executed")
@pytest.mark.parametrize(
    "tequila_circuit",
    [
        (X(target=3) + Y(target=2) + Z(target=1)),
        (Rx(target=1, control=0, angle=5.67) + Ry(target=2, angle=0.98) + Rz(target=3, angle=1.67)),
        (H(target=1) + H(target=1, control=0) + X(target=1) + Y(target=0) + Z(target=2) +
         CX(target=3, control=0) + CY(target=4, control=2) + CZ(target=5, control=1) +
         CNOT(target=3, control=0) + SWAP(first=0, second=3) +
         S(target=1, control=0) + T(target=1, control=2))
    ]
)
def test_convert_to_from_pyzx_simple(tequila_circuit):

    pyzx_circuit = convert_to_pyzx(tequila_circuit)
    converted_circuit = convert_from_pyzx(pyzx_circuit)

    wfn1 = simulate(tequila_circuit, backend="symbolic")
    wfn2 = simulate(converted_circuit, backend="symbolic")

    assert (numpy.isclose(wfn1.inner(wfn2), 1.0))


@pytest.mark.skipif(not HAS_PYZX,
                    reason="Pyzx package not installed, test_convert_to_from_pyzx not executed")
@pytest.mark.parametrize(
    "variabs",
    [
        ([2.8, 5.6, 7.6, 1.8, 4.98, 2.35, 3.12, 6.79, 0.12]),
        ([1.5, 3.7, 9.2, 3.1, 7.62, 1.98, 8.56, 2.97, 1.34]),
        ([0, 0, 0, 0, 0, 0, 0, 0, 0]),
        ([numpy.pi/12, numpy.pi, numpy.pi/3, numpy.pi/6, numpy.pi*0.95, numpy.pi/2, numpy.pi*2.3, numpy.pi/7, 0.56])

    ]
)
def test_convert_to_from_pyzx(variabs):

    variables = {"ang1": variabs[0],
                 "ang2": variabs[1],
                 "ang3": variabs[2],
                 "ang4": variabs[3],
                 "ang5": variabs[4],
                 "ang6": variabs[5],
                 "ang7": variabs[6],
                 "ang8": variabs[7],
                 "ang9": variabs[8]}

    tequila_circuit = H(target=[0, 1], control=2) + \
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
                      Phase(control=0, target=1, phi="ang7") + \
                      S(target=1, control=0) + \
                      T(target=1, control=2) + \
                      Rp(paulistring="Y(1)", angle="ang8") + \
                      ExpPauli(paulistring="Z(1)X(2)", control=0, angle="ang9") + \
                      Toffoli(target=2, first=4, second=3)

    pyzx_circuit = convert_to_pyzx(tequila_circuit, variables=variables)
    converted_circuit = convert_from_pyzx(pyzx_circuit)

    wfn1 = simulate(tequila_circuit, backend="symbolic", variables=variables)
    wfn2 = simulate(converted_circuit, backend="symbolic")

    assert (numpy.isclose(wfn1.inner(wfn2), 1.0))


@pytest.mark.skipif(not HAS_PYZX,
                    reason="Pyzx package not installed, test_convert_to_from_pyzx_trotterized_gate not executed")
@pytest.mark.parametrize(
    "string1,string2,angles,steps",
    [
        ("1.0 X(1)Z(2) - 0.5 Z(3)X(4)", None,                          [numpy.pi/12],   1),
        ("1.0 X(1)Z(2) + 0.5 Z(3)X(4)", "1.0 Y(1)X(2) - 0.9 X(2)Z(3)", [5.6, numpy.pi], 1),
        ("1.5 Z(2)Z(4) + 0.8 Y(3)X(4)", None,                          [numpy.pi],      1)
    ]
)
def test_convert_to_from_pyzx_trotterized_gate(string1, string2, angles, steps):

    variables = {"ang1": angles[0]} if string2 is None else {"ang1": angles[0], "ang2": angles[1]}

    g1 = QubitHamiltonian.from_string(string1)
    g2 = None if string2 is None else QubitHamiltonian.from_string(string2)
    tequila_circuit = Trotterized(generators=[g1] if string2 is None else [g1, g2],
                                  angles=["ang1"] if string2 is None else ["ang1", "ang2"],
                                  steps=steps)

    pyzx_circuit = convert_to_pyzx(tequila_circuit, variables=variables)
    converted_circuit = convert_from_pyzx(pyzx_circuit)

    wfn1 = simulate(tequila_circuit, backend="symbolic", variables=variables)
    wfn2 = simulate(converted_circuit, backend="symbolic")

    assert (numpy.isclose(wfn1.inner(wfn2), 1.0))


@pytest.mark.skipif(not HAS_PYZX,
                    reason="Pyzx package not installed, test_convert_from_pyzx_exception not executed")
def test_convert_from_pyzx_exception():

    pyzx_circuit = pyzx.circuit.Circuit(qubit_amount=1)
    tequila_circuit = QCircuit()

    with pytest.raises(expected_exception=TequilaException,
                       match="Circuit provided must be of type pyzx.circuit.Circuit"):
        convert_from_pyzx(tequila_circuit)

    assert (isinstance(convert_from_pyzx(pyzx_circuit), QCircuit))


@pytest.mark.skipif(not HAS_PYZX,
                    reason="Pyzx package not installed, test_convert_to_from_pyzx_optimizing_circuit not executed")
@pytest.mark.parametrize(
    "tequila_circuit,t_reduce",
    [
        (X(target=3) + Y(target=2) + Z(target=1), True),
        (Rx(target=1, control=0, angle=5.67) + Ry(target=2, angle=0.98) + Rz(target=3, angle=1.67), False),
        (H(target=1) + H(target=1, control=0) + X(target=1) + Y(target=0) + Z(target=2) +
         CX(target=3, control=0) + CY(target=4, control=2) + CZ(target=5, control=1) +
         CNOT(target=3, control=0) + SWAP(first=0, second=3) +
         S(target=1, control=0) + T(target=1, control=2), True)
    ]
)
def test_convert_to_from_pyzx_optimizing_circuit(tequila_circuit, t_reduce):

    pyzx_circuit = convert_to_pyzx(tequila_circuit)

    pyzx_graph = pyzx_circuit.to_graph()

    if t_reduce:
        pyzx.teleport_reduce(pyzx_graph)
        pyzx_circuit_opt = pyzx.Circuit.from_graph(pyzx_graph)
    else:
        pyzx.full_reduce(pyzx_graph)
        pyzx_graph.normalize()
        pyzx_circuit_opt = pyzx.extract_circuit(pyzx_graph.copy())

    # compare_tensors returns True if pyzx_circuit and pyzx_circuit_opt
    # implement the same circuit (up to global phase)
    assert (pyzx.compare_tensors(pyzx_circuit, pyzx_circuit_opt))

    # verify_equality return True if full_reduce() is able to reduce the
    # composition of the circuits to the identity
    assert (pyzx_circuit.verify_equality(pyzx_circuit_opt))

    converted_circuit = convert_from_pyzx(pyzx_circuit_opt)

    wfn1 = simulate(tequila_circuit, backend="symbolic")
    wfn2 = simulate(converted_circuit, backend="symbolic")

    assert (numpy.isclose(wfn1.inner(wfn2), 1.0))

