import tequila as tq
import numpy


def test_simple_example():
    H = tq.paulis.X(0)
    Upre = tq.gates.X(0)
    Upost = tq.gates.Y(0)
    generators = [tq.paulis.Y(0), tq.paulis.Z(0), tq.paulis.X(0)]
    operator_pool = tq.adapt.AdaptPoolBase(generators=generators)
    solver = tq.adapt.Adapt(H=H, operator_pool=operator_pool, Upre=Upre, UPost=Upost)
    result = solver()
    assert numpy.isclose(result.energy, -1.0, atol=1.0e-4)
