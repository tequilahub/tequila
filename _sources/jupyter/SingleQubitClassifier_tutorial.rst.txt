Single-qubit classifier with data re-uploading tutorial
=======================================================

This tutorial shows: - How to construct a fidelity cost function for an
optimization problem - How to construct a quantum classifier with one
qubit

Based on `Data re-uploading for a universal quantum
classifier <https://quantum-journal.org/papers/q-2020-02-06-226/>`__, A.
Pérez-Salinas, A. Cervera-Lierta, E. Gil-Fuster, and J. I. Latorre,
*Quantum **4**, 226 (2020)*.

.. code:: ipython3

    import tequila as tq
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy import random
    import time

Introduction
------------

Single-qubit operations are just rotations on the Bloch sphere. A
collection of single-qubit rotations can be reduced to a single one by
combining their angles: you can always move from one point to another on
the Bloch sphere with a single operation.

Because of this property, a single-qubit classifier with simple
parameterized rotations can not treat complex data. Even if it succeeds
to classify one data point, it will probably misclassify the others
since rotational angles have been only adapted to one particular point.
To circumvent this limitation one can introduce the data points into
these angles, so each rotation will be data point-dependent. This
methodology is called data re-uploading and it can be shown that a
single-qubit classifier can be universal using this technique.

The strategy to train this single-qubit classifier is the following.
Given a problem with 𝑛n classes, we choose 𝑛n vectors on the Bloch
sphere. Then, we train the classifier by constructing a cost function
that adds penalties if the final state of the classifier is far from the
target state that corresponds to its class.

The single-qubit classifier circuit is divided into layers. Each layer
comprises single-qubit rotations that encode a data training point and
parameters to be optimized.

.. math::  L\left(\vec{x};\vec{\theta}_{i}\right) = U\left(\vec{x}\right)U\left(\vec{\theta}_{i}\right) 

By considering more layers, the final state of the classifier will have
a richer structure in terms of the data point :math:`\vec{x}`.

.. math::  \mathcal{U}_{class}\left(\vec{x};\vec{\theta}_{1},\vec{\theta}_{2},...,\vec{\theta}_{l}\right) = L\left(\vec{x};\vec{\theta}_{1}\right)L\left(\vec{x};\vec{\theta}_{2}\right)\cdots L\left(\vec{x};\vec{\theta}_{l}\right)

We will run a :math:`\mathcal{U}_{class}` for each training point
:math:`\vec{x}`, but the parameters :math:`\vec{\theta}` are the same.
These are the variables to be optimized classically through the cost
function.

Model
-----

Let's define the model that we would like to classify. Let's start with
a simple model with two classes: a circle of radius
:math:`r =\sqrt{2/\pi}` centered at (0,0). Data points will be
distributed in a square of length 2 centered at (0,0). The particular
choice of the circle radius implies that a random classification will
have a :math:`\sim 50`\ % accuracy.

``circle`` function will have two parts: - ``random = True``: generate
and label random points according to their position inside or outside
the circle (used for training) - ``random = False``: computes the label
of a given point ``x_input`` (used for testing)

.. code:: ipython3

    np.random.seed(42)
    
    def circle(samples = False, random = True, x_input = False, center=[0.0, 0.0], radius=np.sqrt(2 / np.pi)):   
        """
        Args:
            samples (int): number of samples to generate
            random = True: generates and labels sample random points
            random = False: labels x_input point
            x_input (array[tuple]): given point to be labeled if random = False
            center (tuple): center of the circle
            radius (float): radius of the circle
    
        Returns:
            if random = True:
                xvals (array[tuple]): data points coordinates 
                yvals (array[int]): corresponding data labels
            if random = False:
                y (int): label of x_input point
        """
        xvals, yvals = [], []
    
        if random == True:
            for i in range(samples):
                x = 2 * (np.random.rand(2)) - 1
                y = 0
                if np.linalg.norm(x - center) < radius:
                    y = 1
                xvals.append(x)
                yvals.append(y)        
            return np.array(xvals), np.array(yvals)
        
        if random == False:
            y = 0
            if np.linalg.norm(x_input - center) < radius:
                y = 1
            return y

Target states
-------------

Next step is to fix the classes states. The classifier will be trained
to return these states depending on the data point. To reduce the
uncertanty, target states should be as much distanced as possible. With
a single qubit, that means to choose points in the Bloch sphere as much
separated as possible. For a two-class problem, an easy choice is
:math:`|0\rangle` and :math:`|1\rangle` states.

.. code:: ipython3

    # |0> : points inside the circle
    # |1> : points outside the circle
    def targ_wfn(y, nclass):
        """
        Arg:
            - y (int): class label
            - nclass: number of classes
        Returns:
            - wfn: wavefunction of the target state
        """
        if nclass == 2:
            if y==0:
                wfn = tq.QubitWaveFunction.from_array(np.asarray([1,0]))
            if y==1:
                wfn = tq.QubitWaveFunction.from_array(np.asarray([0,1]))
        else:
            raise Exception("nclass = {} is not considered".format(nclass))
        return wfn

Single-qubit classifier circuit
-------------------------------

The single-qubit classifier has a layer structure, i.e. we have to
decide how to design one layer and then how many layers we would like to
consider. We will consider the following structure: each layer is a
collection of rotational gates which angles are a linear function of a
data point with the free parameters to be optimized. In particular,

.. math:: L\left(\vec{x};\vec{\theta}_{i}\right) = R_{z}\left(x^{2}+\theta_{i}^{2}\right) R_{y}\left(x^{1}+\theta_{i}^{1}\right).

Then, each layer adds 2 parameters to be optimized. The data points
:math:`(x^1,x^2)` are re-uploaded in each layer.

.. code:: ipython3

    # single-qubit quantum classifier circuit
    def qcircuit(xval, param):
        """
        Arg:
            - xval (array[tuple]): data point
            - param (dict): parameters dictionary
        Returns:
            - qc: quantum circuit
        """
        layers = int((len(param))/2) # 2 parameters/layer
        # initialize the circuit 
        qc = tq.gates.Rz(0.0,0)
        for p in range(0,2*layers-1):
            # add layers to the circuit
            qc += tq.gates.Ry(xval[0] + param[p],0) + tq.gates.Rz(xval[1] + param[p+1],0) 
        return qc

Cost function
-------------

The cost function for this quantum classifier model will be constructed
from the fidelity of the classifier final state respect to the target
state of its corresponding class. It will penalize that the output state
is far from its label state.

First, we define the fidelity between two states as an objective (see
State Preparation Tutorial). Then, we construct the simplest cost
function of this kind: average of squared infidelities for all training
points :math:`M`:

.. math::  \chi^2 = \sum_{i=1}^{M}\left(1-|\langle\psi_{target}|\psi_{circuit}\rangle|^2\right)^2

.. code:: ipython3

    # Fidelity objective
    def fid(wfn_targ, qc):
        """
        Arg:
            - wfn_targ: target wavefunction
            - qc : quantum circuit 
        Returns:
            - O: objective
        """  
        rho_targ =  tq.paulis.Projector(wfn=wfn_targ)
        O = tq.Objective.ExpectationValue(U=qc, H=rho_targ)
        # fidelity = tq.simulate(O)
        return O 
    
    # cost function: sum of all infidelities for each data point respect the label state
    def cost(x, y, param, nclass):
        """
        Arg:
            - x (array[tuple]): training points
            - y (array[int]): labels of training points
            - param (dict): parameters dictionary
            - nclass (int): number of classes
        Returns:
            - loss/ len(x): loss objective
        """  
        loss = 0.0
        # M = len(y): number of training points
        for i in range(len(y)):
            
            # state generated by the classifier
            qc = qcircuit(x[i], param)
            # fidelity objective respect to the label state
            f = fid(targ_wfn(y[i],nclass), qc)
            
            loss = loss + (1 - f)**2
            
        return loss / len(x)

Training
--------

| We have now all the ingredients to train a single-qubit classifier
  with data re-uploading.
| If a gradient based optimization is chosen for this type of
  optimization problems, numerical gradients are adviced since
  analytical ones become quite expensive.

.. code:: ipython3

    layers = 3
    nclass = 2
    training_set = 400
    
    # generate the training set and its corresponding labels
    xdata, ydata = circle(training_set)
    
    # generate the variational parameters
    param = [tq.Variable(name='th_{}'.format(i)) for i in range(0,2*layers)]
    
    # initialize the variational parameters
    # note that due to the random initialization the result can be different from time to time
    # With gradient based optimization you might get stuck
    inval = {key : random.uniform(0, 2*np.pi) for key in param}
    
    grad = '2-point' # numerical gradient (= None: analytical gradient)
    mthd = 'rmsprop' # scipy minimization method
    mthd_opt = {'eps':1.e-4} # method options (that's the stepsize for the gradients)
    
    obj = cost(xdata, ydata, param, nclass) # objective to be optimized: cost function
    
    t0 = time.time()
    # depending on the optimizer this will take a while
    test = tq.minimize(objective=obj, initial_values=inval, method = mthd,
                                       gradient = grad, method_options = mthd_opt, silent=False)
    t1 = time.time()

Extract the results:

.. code:: ipython3

    print("loss = ", test.energy)
    print("method: ", mthd)
    print("method parameters: ", mthd_opt)
    print("execution time = ", (t1-t0)/60, " min")
    
    print(test.history.plot('energies', label='loss'))
    print(test.history.plot('angles', label=""))

Test
----

Once trained, we have the optimal parameters for the classifier stored
in ``test.angles``. We run again the classifier with the test data set
and with these parameters fixed.

.. code:: ipython3

    test_set = 1000
    
    # initialize
    xval_test, yval_test, yval_rand, yval_real = [], [], [], []
    suc = 0 # success
    suc_rand = 0 # random success
    
    for i in range(test_set):
        
        # random test point
        x = 2 * (np.random.rand(2)) - 1
        
        # state generated by the trained classifier
        qc = qcircuit(x, param)
        wfn_qc = tq.simulate(qc, variables=test.angles) 
            
        if nclass == 2:
            
            # compute the fidelity respect to one of the label states, the |0>
            f = abs(wfn_qc.inner(targ_wfn(0,nclass)))**2
    
            y = 1
            # if fidelity is >= 0.5, we conclude that this state belongs to |0> class
            # (|1> class otherwise)
            if f >= 0.5:
                y = 0
                
            # check the real class of the data point
            y_real = circle(random=False, x_input=x)
            
        else:
            raise Exception("nclass = {} is not considered".format(nclass))
        
        # compute success rate 
        if y == y_real:
            suc = suc + 1
            
        # compute random success rate 
        yrand = np.random.randint(0, nclass-1)
        if yrand == y_real:
            suc_rand = suc_rand + 1
            
        xval_test.append(x)
        yval_test.append(y)
        yval_real.append(y_real)
        
    print("success %: ", 100*suc/test_set,"%")
    print("random success %: ", 100*suc_rand/test_set,"%")

Print results:

.. code:: ipython3

    def plot_data(x, y, nclass, fig=None, ax=None):
        """
        Arg:
            - x (array[tuple]): data points
            - y (array[int]): data labels
            - nclass (int): number of classes
        Returns:
            - Plot
        """    
        if fig == None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
         
        # colors and labels
        col = ["red","blue","green","yellow"]
        lab = [0,1,2,3]
        
        for i in range(nclass):
            ax.scatter(x[y == lab[i], 0], x[y == lab[i], 1], c=col[i], s=20, edgecolor="k")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

.. code:: ipython3

    xval_test = np.array(xval_test)
    yval_test = np.array(yval_test)
    yval_real = np.array(yval_real)
    
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    
    plot_data(xval_test, yval_test, nclass, fig, axes[0])
    plot_data(xval_test, yval_real, nclass, fig, axes[1])
    
    axes[0].set_title("Single-qubit class. {} layers".format(layers))
    axes[1].set_title("True test data")
    fig.tight_layout(pad=0.5)
    plt.show()
    
    t2 = time.time()
    print("Total execution time: ", (t2-t0)/60," min.")

Improvements and customization
------------------------------

This tutorial just shows a simple classification example. It is
constructed in a way that one can easily change the classification model
and try more sophisticated problems comprising more classes. To do so,
one should define the target states for >2 classes, i.e. include more
vectors in the Bloch sphere.

The single-qubit classifier circuit can also be modified to include
higher dimensional data or to increase/reduce the number of parameters
per layer.

The cost function can also be improved. See for instance the weigthed
fidelity cost function proposed in the main reference.

Finally, the core of any variational algorithm is the minimization
method. Tequila provides many methods besides the scipy ones. See the
`optimizers
tutorial <https://github.com/aspuru-guzik-group/tequila/blob/master/tutorials/Optimizer_Tutorial.ipynb>`__
for more information. Notice also that the algorithm starts with a
random initialization. It is well-known that random initialization in
variational circuits leads to a barren-plateaus problem when computing
the gradients. This problem can be avoided by providing a good
initialization guess.

Another possibility to play with expoloiting other Tequila modules is to
use Phoenics for initial exploration and a gradient based optimizer with
the best Phoenics results as starting point (see Phoenics
documentation).

