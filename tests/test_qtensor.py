import tequila as tq
import numpy

def make_expval_list():
    H = tq.paulis.X(0)
    Hz = tq.paulis.Z(0)
    U1 = tq.gates.Ry(angle='a',target=0)
    U2 = tq.gates.X(0)+U1

    U3 = tq.gates.Ry(angle='b',target=0)

    E1 = tq.ExpectationValue(H=H, U=U1)
    E2 = tq.ExpectationValue(H=H, U=U2)
    E3 = tq.ExpectationValue(H=H, U=U1+U3)
    return [E1, E2, E3, E1]

def test_qtensor_with_numbers():
    list1 = [0.5+tq.Objective(),1.5+tq.Objective(),0.75+tq.Objective(),2+tq.Objective()]
    list2 = [1+tq.Objective(),1+tq.Objective()]
    mat1 = tq.QTensor(objective_list=list1,shape = [2,2])
    vec1 = tq.QTensor(objective_list=list2,shape = [2])
    res = tq.simulate(numpy.dot(mat1,vec1))
    numpy.testing.assert_allclose(res,[2, 2.75],atol=1e-05)

def test_qtensor_with_objectives():
    list1 = make_expval_list()
    mat1  = tq.QTensor(objective_list=list1, shape=(2,2))
    E = tq.simulate(mat1, {'a':1.0,'b':0.5})
    F = numpy.array([[0.84147098, -0.84147098],[0.99749499, 0.84147098]])
    numpy.testing.assert_allclose(E,F,atol=1e-05)

def test_apply():
    list1 = make_expval_list()
    mat1  = tq.QTensor(objective_list=list1, shape=(2,2))
    mat2 = mat1.apply(numpy.exp)
    E = tq.simulate(mat2, {'a':1.0,'b':0.5})
    F = numpy.array([[2.31977682, 0.43107595], [2.71148102, 2.31977682]])
    numpy.testing.assert_allclose(E,F,atol=1e-05)

def test_count_expval():
    list1 = make_expval_list()
    mat1  = tq.QTensor(objective_list=list1, shape=(2,2))
    E = mat1.count_expectationvalues()
    assert E == 3

def test_add():
    list1 = make_expval_list()
    mat1  = tq.QTensor(objective_list=list1, shape=(2,2))
    E = tq.simulate(mat1+mat1,{'a':1.0,'b':0.5})
    F = tq.simulate(2*mat1,{'a':1.0,'b':0.5})
    G = tq.simulate(mat1*2,{'a':1.0,'b':0.5})
    H = tq.simulate(2*mat1+mat1*2,{'a':1.0,'b':0.5})
    A = numpy.array([[ 1.68294197, -1.68294197],[ 1.99498997, 1.68294197]])
    B = numpy.array([[ 3.36588394, -3.36588394],[ 3.98997995, 3.36588394]])
    numpy.testing.assert_allclose(E,A,atol=1e-05)
    numpy.testing.assert_allclose(F,A,atol=1e-05)
    numpy.testing.assert_allclose(G,A,atol=1e-05)
    numpy.testing.assert_allclose(H,B,atol=1e-05)

def test_dot():
    list1 = make_expval_list()
    mat1  = tq.QTensor(objective_list=list1, shape=(2,2))
    E = tq.simulate(numpy.dot(mat1,mat1),{'a':1.0,'b':0.5})
    F = numpy.array([[-0.13128967, -1.41614684], [ 1.67872618, -0.13128967]])
    numpy.testing.assert_allclose(E,F,atol=1e-05)

def test_grad():
    list1 = make_expval_list()
    mat1  = tq.QTensor(objective_list=list1, shape=(2,2))
    mat2 = tq.grad(mat1,'a')
    mat3 = tq.grad(mat2,'b')
    E1 = tq.simulate(mat1,{'a':1.0,'b':0.5})
    E2 = tq.simulate(mat2,{'a':1.0,'b':0.5})
    E3 = tq.simulate(mat3,{'a':1.0,'b':0.5})
    F1 = numpy.array([[0.84147098, -0.84147098],[ 0.99749499,0.84147098]])
    F2 = numpy.array([[ 0.54030231, -0.54030231],[0.0707372, 0.54030231]])
    F3 = numpy.array([[0.,0.],[-0.99749499, 0.]])
    numpy.testing.assert_allclose(E1,F1,atol=1e-05)
    numpy.testing.assert_allclose(E2,F2,atol=1e-05)
    numpy.testing.assert_allclose(E3,F3,atol=1e-05)