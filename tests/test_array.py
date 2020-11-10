import tequila as tq
import pytest, numpy

# Backends
import select_backends
backends = select_backends.get()


@pytest.mark.parametrize("backend", backends)
def test_array_computation(backend):
    U = tq.gates.X(target=0)
    hamiltonians = [tq.paulis.I(), tq.paulis.X(0), tq.paulis.Y(0), tq.paulis.Z(0)]
    E = tq.ExpectationValue(H=hamiltonians, U=U, shape=[4])
    result = tq.simulate(E, backend=backend)
    assert all(result == numpy.asarray([1.0, 0.0, 0.0, -1.0]))

@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("shape", [(4,),(2,2)])
def test_array_shape(backend, shape):
    expected = numpy.asarray([1.0, 0.0, 0.0, -1.0]).reshape(shape)
    U = tq.gates.X(target=0)
    hamiltonians = [tq.paulis.I(), tq.paulis.X(0), tq.paulis.Y(0), tq.paulis.Z(0)]
    E = tq.ExpectationValue(H=hamiltonians, U=U, shape=shape)
    result = tq.simulate(E, backend=backend)
    assert (result == expected).all()

@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("shape", [(2,2)])
def test_array_contraction(backend, shape):
    def contraction(array):
        bra = numpy.asarray([1.0, 1.0, 1.0, -1.0]).reshape(shape)
        return numpy.trace(bra.dot(array))

    expected = numpy.asarray([1.0, 0.0, 0.0, -1.0]).reshape(shape)
    U = tq.gates.X(target=0)
    hamiltonians = [tq.paulis.I(), tq.paulis.X(0), tq.paulis.Y(0), tq.paulis.Z(0)]
    E = tq.ExpectationValue(H=hamiltonians, U=U, shape=shape, contraction=contraction)
    result = tq.simulate(E, backend=backend)
    assert result == contraction(expected)
