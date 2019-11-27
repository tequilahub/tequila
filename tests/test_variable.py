import numpy as np
from tequila.circuit.gradient import __weight_chain
from tequila import Variable
from tequila.circuit.variable import Transform
import operator

def test_nesting():
	a=Variable(name='',value=3)
	b=a+2-2
	c=(b*5)/5
	d=-(-c)
	e=d**0.5
	f=e**2

	assert np.isclose(a(),f())

def test_gradient():
	a=Variable(name='',value=3)
	b=a+2-2
	c=(b*5)/5
	d=-(-c)

	assert __weight_chain(d,a)==1.0

def test_equality():
	a=Variable('a',7)
	b=Variable('a.',7)
	assert a!=b


def test_var_update():
	a=Variable('a',7)
	a.update({'a':8})
	assert np.isclose(a.value,8.0)
	a.update({'b':3})
	assert np.isclose(a.value,8.0)

def test_transform_update():
	a=Variable('a',7)
	b=Variable('a.',23)
	t=Transform(func=operator.add,args=[a,b])
	d={'a':8,'a.':1,'a':9,'c':17}
	t.update(d)
	assert np.isclose(float(t),10.0)


