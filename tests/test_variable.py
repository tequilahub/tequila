import pytest
from jax import numpy as np
from tequila.circuit.gradient import grad
from tequila.objective.objective import Objective
from tequila.circuit.variable import Variable
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

	assert grad(d, a)() == 1.0

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
	t=Objective(transformation=operator.add,args=[a,b])
	d={'a':8,'a.':1,'a':9,'c':17}
	t.update_variables(d)
	assert np.isclose(float(t()),10.0)

@pytest.mark.parametrize('gradvar',['a','b','c','d','e','f'])
def test_exotic_gradients(gradvar):
	a=Variable('a',2)
	b=Variable('b',3)
	c=Variable('c',4)
	d=Variable('d',5)
	e=Variable('e',6)
	f=Variable('f',7)
	t= c*a**b +b/c -Objective(args=[c],transformation=np.cos) +f/(d*e) +a*Objective(args=[d],transformation=np.exp)/(f+b) \
	   +Objective(args=[e],transformation=np.tanh) + Objective(args=[f],transformation=np.sinc)
	g=grad(t, gradvar)
	if gradvar is 'a':
		assert g() == c()*b()*(a()**(b()-1.)) +np.exp(d())/(f()+b())
	if gradvar is 'b':
		assert g() == (c()*a()**b())*np.log(a()) +1./c() -a()*np.exp(d())/(f()+b())**2.
	if gradvar is 'c':
		assert g() == a()**b() -  b()/ c()**2. + np.sin(c())
	if gradvar is 'd':
		assert g() == -f()/(np.square(d())*e()) + a()*np.exp(d())/(f()+b())
	if gradvar is 'e':
		assert np.isclose(g(),2./(1.+np.cosh(2*e())) - f()/(d()*e()**2.))
	if gradvar is 'f':
		assert g() == 1./(d()*e()) -a()*np.exp(d())/(f()+b())**2. +np.cos(np.pi*f())/f() -np.sin(np.pi*f())/(np.pi*f()**2.)


