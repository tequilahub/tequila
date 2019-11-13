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

def test_seperability():
	a=Variable(7)
	b=Variable(7)
	assert a!=b



