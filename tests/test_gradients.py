from openvqe.circuit import *
from openvqe.objective import Objective
from openvqe.circuit.gradient import grad
from openvqe.circuit.gates import X,Y,Z, Ry, Rx, Rz
from openvqe.hamiltonian import PX, PY, PZ
from openvqe.circuit._gates_impl import RotationGateImpl
from openvqe.hamiltonian import HamiltonianBase
from openfermion import QubitOperator
from openvqe.simulator.simulator_cirq import SimulatorCirq
import numpy as np
from numpy import pi
if __name__ == "__main__":

	rx_g=[]
	ry_g=[]
	rz_g=[]

	rx_e=[]
	ry_e=[]
	rz_e=[]


	crx_g=[]
	cry_g=[]
	crz_g=[]

	crx_e=[]
	cry_e=[]
	crz_e=[]


	t=[]
	for i in range(65):
		theta=i*pi/16
		t.append(theta)
		a= Rx(target=0, angle=theta, phase=1.0, frozen=False)
		b= Ry(target=0, angle=theta, phase=1.0, frozen=False)
		c= Rz(target=0, angle=theta, phase=1.0, frozen=False)

		circuit_a=QCircuit.wrap_gate(a)
		circuit_b=QCircuit.wrap_gate(b)
		circuit_c=QCircuit.wrap_gate(c)
		O_a=Objective(observable=PZ(qubit=0),unitaries=circuit_a)
		O_b=Objective(observable=PX(qubit=0),unitaries=circuit_b)
		O_c=Objective(observable=PY(qubit=0),unitaries=circuit_c)

		e_a= SimulatorCirq().expectation_value(objective=O_a)
		e_b= SimulatorCirq().expectation_value(objective=O_b)
		e_c= SimulatorCirq().expectation_value(objective=O_c)
		g_a= SimulatorCirq().expectation_value(objective=grad(O_a)[0])
		g_b= SimulatorCirq().expectation_value(objective=grad(O_b)[0])
		g_c= SimulatorCirq().expectation_value(objective=grad(O_c)[0])

		rx_e.append(e_a)
		ry_e.append(e_b)
		rz_e.append(e_c)

		rx_g.append(g_a)
		ry_g.append(g_b)
		rz_g.append(g_c)

	from matplotlib import pyplot as plt
	plt.title("Testing Gradient of Rx with observable Z")
	plt.xlabel("Theta")
	plt.plot(t, rx_e, label="E")
	plt.plot(t, rx_g, label="dE/dt")
	plt.legend()
	plt.show()

	plt.title("Testing Gradient of Ry with observable X")
	plt.xlabel("Theta")
	plt.plot(t, ry_e, label="E")
	plt.plot(t, ry_g, label="dE/dt")
	plt.legend()
	plt.show()

	plt.title("Testing Gradient of Rz with observable Y")
	plt.xlabel("Theta")
	plt.plot(t, rz_e, label="E")
	plt.plot(t, rz_g, label="dE/dt")
	plt.legend()
	plt.show()


	t=[]
	for i in range(65):
		theta=i*pi/16
		t.append(theta)
		a= Rx(target=1,control=0, angle=theta, phase=1.0, frozen=False)
		b= Ry(target=1,control=0, angle=theta, phase=1.0, frozen=False)
		c= Rz(target=1,control=0, angle=theta, phase=1.0, frozen=False)

		circuit_a=QCircuit.wrap_gate(X(target=0))*QCircuit.wrap_gate(a)
		circuit_b=QCircuit.wrap_gate(X(target=0))*QCircuit.wrap_gate(b)
		circuit_c=QCircuit.wrap_gate(X(target=0))*QCircuit.wrap_gate(c)
		O_a=Objective(observable=PZ(qubit=1),unitaries=circuit_a)
		O_b=Objective(observable=PX(qubit=1),unitaries=circuit_b)
		O_c=Objective(observable=PY(qubit=1),unitaries=circuit_c)

		e_a= SimulatorCirq().expectation_value(objective=O_a)
		e_b= SimulatorCirq().expectation_value(objective=O_b)
		e_c= SimulatorCirq().expectation_value(objective=O_c)
		g_a= SimulatorCirq().expectation_value(objective=grad(O_a)[0])
		g_b= SimulatorCirq().expectation_value(objective=grad(O_b)[0])
		g_c= SimulatorCirq().expectation_value(objective=grad(O_c)[0])

		crx_e.append(e_a)
		cry_e.append(e_b)
		crz_e.append(e_c)

		crx_g.append(g_a)
		cry_g.append(g_b)
		crz_g.append(g_c)

	plt.title("Testing Gradient of CRx with observable Z")
	plt.xlabel("Theta")
	plt.plot(t, crx_e, label="E")
	plt.plot(t, crx_g, label="dE/dt")
	plt.legend()
	plt.show()

	plt.title("Testing Gradient of CRy with observable X")
	plt.xlabel("Theta")
	plt.plot(t, cry_e, label="E")
	plt.plot(t, cry_g, label="dE/dt")
	plt.legend()
	plt.show()

	plt.title("Testing Gradient of CRz with observable Y")
	plt.xlabel("Theta")
	plt.plot(t, crz_e, label="E")
	plt.plot(t, crz_g, label="dE/dt")
	plt.legend()
	plt.show()

	x_g=[]
	y_g=[]
	z_g=[]

	x_e=[]
	y_e=[]
	z_e=[]


	p=[]
	for i in range(65):
		power = i/16
		p.append(power)
		a= X(target=0, power=power, phase=1.0, frozen=False)
		b= Y(target=0, power=power, phase=1.0, frozen=False)
		c= Z(target=0, power=power, phase=1.0, frozen=False)

		circuit_a=QCircuit.wrap_gate(a)
		circuit_b=QCircuit.wrap_gate(b)
		circuit_c=QCircuit.wrap_gate(c)
		O_a=Objective(observable=PZ(qubit=0),unitaries=circuit_a)
		O_b=Objective(observable=PX(qubit=0),unitaries=circuit_b)
		O_c=Objective(observable=PY(qubit=0),unitaries=circuit_c)

		e_a= SimulatorCirq().expectation_value(objective=O_a)
		e_b= SimulatorCirq().expectation_value(objective=O_b)
		e_c= SimulatorCirq().expectation_value(objective=O_c)
		g_a= SimulatorCirq().expectation_value(objective=grad(O_a)[0])
		g_b= SimulatorCirq().expectation_value(objective=grad(O_b)[0])
		g_c= SimulatorCirq().expectation_value(objective=grad(O_c)[0])

		x_e.append(e_a)
		y_e.append(e_b)
		z_e.append(e_c)

		x_g.append(g_a)
		y_g.append(g_b)
		z_g.append(g_c)

	plt.title("Testing Gradient of X with observable Z")
	plt.xlabel("Theta")
	plt.plot(p, x_e, label="E")
	plt.plot(p, x_g, label="dE/dt")
	plt.legend()
	plt.show()

	plt.title("Testing Gradient of Y with observable X")
	plt.xlabel("Theta")
	plt.plot(p, y_e, label="E")
	plt.plot(p, y_g, label="dE/dt")
	plt.legend()
	plt.show()

	plt.title("Testing Gradient of Z with observable Y")
	plt.xlabel("Theta")
	plt.plot(p, z_e, label="E")
	plt.plot(p, z_g, label="dE/dt")
	plt.legend()
	plt.show()