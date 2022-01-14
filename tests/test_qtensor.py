import tequila as tq
import numpy

def test_qtensor_with_numbers():
    list1 = [0.5+tq.Objective(),1.5+tq.Objective(),0.75+tq.Objective(),2+tq.Objective()]
    list2 = [1+tq.Objective(),1+tq.Objective()]
    mat1 = tq.QTensor(objective_list=list1,shape = [2,2])
    vec1 = tq.QTensor(objective_list=list2,shape = [2])
    res = tq.simulate(numpy.dot(mat1,vec1))
    assert res[0] == 2.