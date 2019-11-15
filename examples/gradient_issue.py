from openvqe.circuit.gradient import grad
from openvqe import Variable, gates
from numpy import pi

print("Weights are inconsistent")

a = Variable(name='a', value=pi/2)
U = gates.X(target=1) + gates.Ry(target=0, angle=a)
test = grad(U)['a']
print(test)

a = Variable(name='a', value=pi/2)
U = gates.X(target=1) + gates.Ry(target=0, angle=-a)
test = grad(U)['a']
print(test)