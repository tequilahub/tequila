from openvqe import OpenVQEException
from openvqe.circuit.circuit import QCircuit
from openvqe import numpy as np
from openvqe.circuit.gates import Rx, H, X
from openvqe.circuit._gates_impl import RotationGateImpl, QGateImpl, PowerGateImpl, MeasurementImpl
from openvqe.circuit.variable import Variable,Transform,Add,Sub,Div,Mul,Pow,Sqr,has_variable
from openvqe.circuit.gradient import grad, tgrad, weight_chain
from openvqe import copy


def test_nesting():
	a=Variable(name='',value=3)
	b=a+2-2
	c=(b*5)/5
	d=-(-c)
	e=Transform(Sqr,[d])
	f=e**2

	assert np.isclose(a(),f())

def test_gradient():
	a=Variable(name='',value=3)
	b=a+2-2
	c=(b*5)/5
	d=-(-c)

	assert weight_chain(d,a)==1.0

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
	t=Transform(func=Add,args=[a,b])
	t.update([{'a':8,'a.':1,'a':9,'c':17}])
	assert np.isclose(float(t),10.0)


