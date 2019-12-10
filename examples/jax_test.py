import jax
from jax import numpy as numpy
from matplotlib import pyplot as plt

def my_function(x):
    return 2.0*numpy.sin(x) + numpy.sin(x)**2

def test(f, df, title="title"):
    steps=25
    O = []
    dO = []
    for step in range(steps):
        var = 0.0 + step/steps*2*numpy.pi
        O.append(f(var))
        dO.append(df(var))

    fig = plt.figure()
    plt.plot(O, label="O")
    plt.plot(dO, label="dO")
    plt.title(title)
    plt.legend()
    plt.show()

f = my_function
df= jax.grad(f)
test(f, df)




