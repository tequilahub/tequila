import pytest
from tequila import numpy as np
from tequila.circuit.gradient import grad
from tequila import Objective, Variable, VectorObjective
import operator


def test_nesting():
    a = Variable(name='a')
    variables = {a: 3.0}
    b = a + 2 - 2
    c = (b * 5) / 5
    d = -(-c)
    e = d ** 0.5
    f = e ** 2

    assert np.isclose(a(variables), f(variables))


def test_gradient():
    a = Variable(name='a')
    variables = {a: 3.0}
    b = a + 2 - 2
    c = (b * 5) / 5
    d = -(-c)

    assert grad(d, a)(variables) == 1.0


def test_equality():
    a = Variable('a')
    b = Variable('a.')
    assert a != b


# test backward compatibility (VectorObjective is now a special case of QTensor)
def test_transform_update():
    a = Variable('a')
    b = Variable('a.')
    t = VectorObjective(transformations=[operator.add], argsets=[[a, b]])
    variables = {a: 8, b: 1, a: 9, "c": 17}
    assert np.isclose(float(t(variables)), 10.0)


@pytest.mark.parametrize('gradvar', ['a', 'b', 'c', 'd', 'e', 'f'])
def test_exotic_gradients(gradvar):
    # a and b will fail for autograd not with jax
    a = Variable('a')
    b = Variable('b')
    c = Variable('c')
    d = Variable('d')
    e = Variable('e')
    f = Variable('f')
    variables = {a: 2.0, b: 3.0, c: 4.0, d: 5.0, e: 6.0, f: 7.0}

    t = c * a ** b + b / c - Objective(args=[c], transformation=np.cos) + f / (d * e) + a * Objective(args=[d],
                                                                                                      transformation=np.exp) / (
                    f + b) + Objective(args=[e], transformation=np.tanh) + Objective(args=[f],
                                                                                             transformation=np.sinc)
    g = grad(t, gradvar)
    if gradvar == 'a':
        assert np.isclose(g(variables) , c(variables) * b(variables) * (a(variables) ** (b(variables) - 1.)) + np.exp(d(variables)) / (f(variables) + b(variables)))
    if gradvar == 'b':
        assert np.isclose(g(variables) , (c(variables) * a(variables) ** b(variables)) * np.log(a(variables)) + 1. / c(variables) - a(variables) * np.exp(d(variables)) / (f(variables) + b(variables)) ** 2.0)
    if gradvar == 'c':
        assert np.isclose(g(variables) , a(variables) ** b(variables) - b(variables) / c(variables) ** 2. + np.sin(c(variables)))
    if gradvar == 'd':
        assert np.isclose(g(variables) , -f(variables) / (np.square(d(variables)) * e(variables)) + a(variables) * np.exp(d(variables)) / (f(variables) + b(variables)))
    if gradvar == 'e':
        assert np.isclose(g(variables), 2. / (1. + np.cosh(2 * e(variables))) - f(variables) / (d(variables) * e(variables) ** 2.))
    if gradvar == 'f':
        assert np.isclose(g(variables) , 1. / (d(variables) * e(variables)) - a(variables) * np.exp(d(variables)) / (f(variables) + b(variables)) ** 2. + np.cos(np.pi * f(variables)) / f(variables) - np.sin(np.pi * f(variables)) / (np.pi * f(variables) ** 2.))
