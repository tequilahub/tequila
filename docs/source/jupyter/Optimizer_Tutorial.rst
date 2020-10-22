Optimization
============

Hello, and welcome to our tutorial on optimization. Here, we will
explore the four different optimizers which already have ``Tequila``
interfaces: a native GD optimizer, alongside interfaces for ``SciPy``,
``GPyOpt``, and ``Phoenics``.

.. code:: ipython3

    ### start at the start: import statements!
    import tequila as tq
    import numpy as np

Overview
--------

**How to optimize an ``Objective``:**

In ``tequila``, optimizers are accessed in several ways. They may be
instantiated as objects directly, which can then be called; they are
also accessible through ``tq.minimize``. Either of these methods have an
obligatory argument: an ``Objective``. ``tq.minimize`` also requires you
supply a ``method``, which must be a string; the call methods of the GD,
``SciPy``, and ``GPyOpt`` optimizers accept this key as well.

As keywords to any optimization call, you can pass down the usual
compilation kwargs for quantum simulation in ``Tequila``: \*
``backend``, a string indicating which quantum simulator to use \*
``samples``, an int indicating how many shots of the circuit to measure
(None means full wf simulation) \* ``device``, (generally) a string
indicating which (real or emulated) quantum computer to sample from
(requires samples be specified), \* ``noise``, the NoiseModel object to
apply to the circuits simulated (see the tutorial on noise).

additional keywords you might use include: \* ``variables``, a ``list``
of the ``Variable``\ s you want to optimize (the default is all of
them). \* ``initial_values``, which gives a start point to optimization.
If you supply arguments to ``variables``, you also need to supply
arguments to ``initial_values`` so that all non-optimized parameters
have a value to use. The default is random initialization (not
recomended) \* ``gradient``, which specifies which type of gradient will
be used : analytical gradients (Default): gradient=None, numerical
gradients: gradient={'method':'2-point', "stepsize":1.e-4}, custom
gradient objectives: gradient={tq.Variable:tq.Objective}, module
specific gradients: gradient="2-point" (to use for example ``SciPy``
finite difference stencils). \* ``silent``, silence outputs

Some of the optimizers take more, or different, keywords from the
others, so check the documentation for each one. In case the optimizer
has some degree of verbosity (currently, they all do), you can
deactivate this with ``silent=True``.

The following optimization methods are available on your system in
``Tequila``:

.. code:: ipython3

    tq.optimizers.show_available_optimizers()

We will use two different ``Objective``\ s for optimization in this
tutorial. The first of these is a two qubit expectation value with the
tractable but non trivial hamiltonian :math:`[Y(0)+Qm(0)]\otimes X(1)`,
where :math:`Qm=\frac{1}{2} (I - Z)`, the projector onto the \|1> state.

.. code:: ipython3

    ### optimizing the circuit in terms of pi makes the result of the optimization easier to interpret.
    
    a = tq.Variable(name="a")*tq.numpy.pi
    b = tq.Variable(name="b")*tq.numpy.pi
    c = tq.Variable(name="c")*tq.numpy.pi
    d = tq.Variable(name='d')*tq.numpy.pi
    
    U1 = tq.gates.H(target=[0])
    U1 += tq.gates.H(target=1)
    U1 += tq.gates.Ry(target=0, angle=a)
    U1 += tq.gates.Rz(target=1, angle=b)
    U1 += tq.gates.Z(target=1,control=0)
    U1 += tq.gates.Rx(target=0, angle=c)
    U1 += tq.gates.Rx(target=1,angle=d)
    U1 += tq.gates.Z(target=1,control=0)
    
    
    ### once we have a circuit, we pick a hamiltonian to optimize over
    H1=(tq.paulis.Y(0)+tq.paulis.Qm(0))*tq.paulis.X(1)
    O1=tq.ExpectationValue(U=U1,H=H1)
    
    ### we use the .draw function to pretty-print circuits via backend printers.
    print('We will optimize the following objective: \n')
    tq.draw(O1,backend='qiskit')

Our second ``Objective``, O2, will measure a 3-qubit circuit with
respect to the Hamiltonian :math:`Y(0)\otimes X(1) \otimes Y(2)`

.. code:: ipython3

    ### this time, don't scale by pi
    
    H2 = tq.paulis.Y(0)*tq.paulis.X(1)*tq.paulis.Y(2)
    U2 = tq.gates.Ry(tq.numpy.pi/2,0) +tq.gates.Ry(tq.numpy.pi/3,1)+tq.gates.Ry(tq.numpy.pi/4,2)
    U2 += tq.gates.Rz('a',0)+tq.gates.Rz('b',1)
    U2 += tq.gates.CNOT(control=0,target=1)+tq.gates.CNOT(control=1,target=2)
    U2 += tq.gates.Ry('c',1) +tq.gates.Rx('d',2)
    U2 += tq.gates.CNOT(control=0,target=1)+tq.gates.CNOT(control=1,target=2)
    O2 = tq.ExpectationValue(H=H2, U=U2)
    
    print('We will optimize the following objective: \n')
    tq.draw(O2, backend="qiskit")

Local Optimizers
----------------

We will begin this tutorial by focusing on local optimizers. By local
optimization, we mean the any optimization schema where the suggested
parameters at step t are always a transformation of the parameters
suggested at step t-1. This includes a large number of the standard
optimization techniques in use today, like gradient descent. ``Tequila``
comes with two local optimizers: a native gradient descent optimizer,
implementing a number of the most popular gradient descent algorithms
used in classical machine learning, as well as a plugin for the
``SciPy`` package (which is installed alongside ``Tequila``), which
allows the use of a number of gradient-free, gradient-based, and
hessian-based optimization methods.

The GD Optimizer
~~~~~~~~~~~~~~~~

we will start this tutorial by looking at the GD optimizer. Here is an
overview over the available optimization methods.

.. code:: ipython3

    tq.show_available_optimizers(module="gd")

As one sees, a variety of methods are available for optimization. Here,
'sgd' refers to the standard gradient descent algorithm, without
momentum. like all tequila optimizers, the GD optimizer has a minimize
function and most of the arguments are the same. However, there is one
important difference: the GD optimizer takes a learning rate, lr. This
parameter mediates step size in all of the GD optimizer methods; it is a
positive float which scales the step in the direction of the gradient.

We will now optimize O1, our two-qubit expectation value, choosing
starting angles equivalent to :math:`\frac{1}{4}\pi` for all four
variables, and optimizing via the
`'Adam' <https://towardsdatascience.com/_adam-latest-trends-in-deep-learning-optimization-6be9a291375c>`__
method.

.. code:: ipython3

    init={'a':0.25,'b':0.25,'c':0.25,'d':0.25}
    lr=0.1
    
    ### For even more fun, try using sampling with the samples keyword, 
    ### or pick your favorite backend with the 'backend' keyword!
    
    adam_result=tq.minimize(objective=O1,lr=lr,
                  method='adam',
                  maxiter=80,
                  initial_values=init,
                  silent=True)

The plots below show the trajectory of both the value of the objective
and the values of the angles as a function of time.

.. code:: ipython3

    adam_result.history.plot('energies')
    adam_result.history.plot('angles')
    print('best energy: ',adam_result.energy)
    print('optimal angles: ',adam_result.angles)

**We see that, minus a few hiccups, all the angles converge to optimimum
values.**

**Let's repeat what we did above, but with a few of the other methods!
Here's RMSprop:**

.. code:: ipython3

    init={'a':0.25,'b':0.25,'c':0.25,'d':0.25}
    lr=0.01
    rms_result=tq.minimize(objective=O1,lr=lr,
                  method='rmsprop',
                  maxiter=80,
                  initial_values=init,
                  silent=True)
    print('RMSprop optimization results:')
    rms_result.history.plot('energies')
    rms_result.history.plot('angles')
    print('best energy: ',rms_result.energy)
    print('optimal angles: ',rms_result.angles)

**... And here's Momentum:**

.. code:: ipython3

    init={'a':0.25,'b':0.25,'c':0.25,'d':0.25}
    lr=0.1
    mom_result=tq.minimize(objective=O1,lr=lr,
                  method='momentum',
                  maxiter=80,
                  initial_values=init,
                  silent=True)
    
    print('momentum optimization results:')
    mom_result.history.plot('energies')
    mom_result.history.plot('angles')
    print('best energy: ',mom_result.energy)
    print('optimal angles: ',mom_result.angles)

Note that when using the
`RMSprop <https://towardsdatascience.com/understanding-rmsprop-faster-neural-network-learning-62e116fcf29a>`__
method, we reduced the learning rate from 0.1 to 0.01. Different methods
may be more or less sensitive to choices of initial learning rate. Try
going back to the previous examples, and choosing different learning
rates, or different initial parameters, to gain a feel for how sensitive
different methods are.

**The GD optimizer, with the Quantum Natural Gradient:**

The Quantum Natural Gradient, or QNG, is a novel method of calculating
gradients for quantum systems, inspired by the natural gradient
sometimes employed in classical machine learning. The usual gradient we
employ is with respect to a euclidean manifold, but this is not the only
geometry -- nor even, the optimal geometry -- of quantum space. The QNG
is, in essence, a method of taking gradients with respect to (an
approximation to) the Fubini-Study metric. For information on how (and
why) the QNG is used, see `Stokes
et.al <https://arxiv.org/abs/1909.02108>`__.

Using the qng in Tequila is as simple as passing in the keyword
gradient='qng' to optimizers which support it, such as the GD optimizer.
We will use it to optimize O2, our 3 qubit ``Objective``, and then
compare the results to optimizing the same circuit with the regular
gradient.

.. code:: ipython3

    ### the keyword stop_count, below, stops optimization if no improvement occurs after 50 epochs.
    ### let's use a random initial starting point:
    init={k:np.random.uniform(-2,2) for k in ['a','b','c','d']}
    
    lr=0.01
    qng_result = tq.minimize(objective=O2,
                         gradient='qng',
                         method='sgd', maxiter=200,lr=lr,
                         initial_values=init, silent=True)

.. code:: ipython3

    qng_result.history.plot('energies')
    qng_result.history.plot('angles')
    print('best energy with qng: ',qng_result.energy)
    print('optimal angles with qng: ',qng_result.angles)

To gain appreciation for why one might use the QNG, let's optimize the
same circuit with the same learning rate and the same method, but
without QNG.

.. code:: ipython3

    lr=0.01
    sgd_noqng_result = tq.minimize(objective=O2,
                         gradient=None,
                         method='sgd', maxiter=200,lr=lr,
                         initial_values=init, silent=True)
    print('plotting what happens without QNG')
    sgd_noqng_result.history.plot('energies')
    sgd_noqng_result.history.plot('angles')
    print('best energy without qng: ',sgd_noqng_result.energy)
    print('optimal angles without qng: ',sgd_noqng_result.angles)

Though the starting point was random you will most likely see that the
QNG run achieved a greater degree of improvement -- it will not perform
worse --, and that the trajectories followed by angles there were
different from those followed by angles in the sgd-only optimization.
Feel free to play around with other methods, learning rates, or circuits
in the space below!

.. code:: ipython3

    ### Use this space to optimize your own circuits!

The SciPy Optimizer
~~~~~~~~~~~~~~~~~~~

``SciPy`` is one of the most popular optimization packages in
``Python``. It offers a wide variety of optimization strategies. We will
not cover them here; for a full exploration of all the ``SciPy``
methods, see `their
docs <https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html>`__.
Here, we will exhibit a few of the more powerful options available. Most
``SciPy`` keywords like ``method_options`` can just be passed to
``minimize`` directly in the same way as when using ``SciPy`` directly.

.. code:: ipython3

    tq.show_available_optimizers(module="scipy")

We will try three different optimizers: ``COBYLA``, which is
gradient-free, ``L-BFGS-B``, which employs gradients, and ``NEWTON-CG``,
which employs the Hessian.

.. code:: ipython3

    print('As a reminder: we will optimize:')
    tq.draw(O1, backend="qiskit")

.. code:: ipython3

    init={'a':0.25,'b':0.25,'c':0.25,'d':0.25}
    
    cobyla_result = tq.minimize(objective=O1,
                                method="cobyla", 
                                initial_values=init, 
                                tol=1.e-3, method_options={"gtol":1.e-3},
                                silent=True)
    
    cobyla_result.history.plot('energies')
    cobyla_result.history.plot('angles')
    print('best energy with cobyla: ',cobyla_result.energy)
    print('optimal angles with cobyla: ',cobyla_result.angles)

.. code:: ipython3

    lb_result = tq.minimize(objective=O1,
                                method="l-bfgs-b", 
                                initial_values=init, 
                                tol=1.e-3, method_options={"gtol":1.e-3},
                                silent=True)
    
    lb_result.history.plot('energies')
    lb_result.history.plot('angles')
    print('best energy with L-BFGS-B: ',lb_result.energy)
    print('optimal angles with L-BFGS-B: ',lb_result.angles)

.. code:: ipython3

    newton_result = tq.minimize(objective=O1,
                                method="newton-cg", 
                                initial_values=init, 
                                tol=1.e-3, method_options={"gtol":1.e-3},
                                silent=True)
    
    newton_result.history.plot('energies')
    newton_result.history.plot('angles')
    print('best energy with NEWTON-CG: ',newton_result.energy)
    print('optimal angles with NEWTON-CG: ',newton_result.angles)

All three of the methods converged to the same minimum, but not
necessarily to the same angles; the gradient and hessian based methods
converged to approximately the same angles in similar time.

**Scipy Extras: numerical gradients and Hessians** Scipy allows for the
use of numerical gradients. To use them, pass down keywords to the
``gradient`` argument, like ``'2-point'``. When using the numerical
gradients of ``SciPy`` it is often crucial to determine a feasible
stepsize for the procedure. This can be done with the ``method_options``
entry ``finite_diff_rel_step`` (for ``SciPy`` version 1.5 or higher) or
``eps`` (for ``SciPy`` version < 1.5).

Here is one example. **Please check your SciPy version!**

.. code:: ipython3

    lb_result = tq.minimize(objective=O1,
                                method="l-bfgs-b", 
                                initial_values=init,
                                gradient="2-point",
                                tol=1.e-3, method_options={"gtol":1.e-3, "finite_diff_rel_step":1.e-4}, # eps for scipy version < 1.5
                                silent=True)
    
    lb_result.history.plot('energies')
    lb_result.history.plot('angles')
    print('best energy with L-BFGS-B: ',lb_result.energy)
    print('optimal angles with L-BFGS-B: ',lb_result.angles)

**Scipy Extras: the QNG in SciPy** Scipy is also configured to use the
qng, just as the gd optimizer is. All one needs to do is set
``gradient=qng``. Let's See how QNG interacts with the ``BFGS``
optimizer. We will use 02, our 3-qubit expectationvalue, that we used
previously.

.. code:: ipython3

    init={k:np.random.uniform(-2,2) for k in ['a','b','c','d']}
    lr=0.01
    bfgs_qng_result = tq.minimize(objective=O2,
                         gradient='qng',
                         method='bfgs', maxiter=200,lr=lr,
                         initial_values=init, silent=True)
    print('plotting what happens with QNG')
    bfgs_qng_result.history.plot('energies')
    bfgs_qng_result.history.plot('angles')
    print('best energy with qng: ',bfgs_qng_result.energy)
    print('optimal angles with qng: ',bfgs_qng_result.angles)

.. code:: ipython3

    bfgs_noqng_result = tq.minimize(objective=O2,
                         gradient=None,
                         method='bfgs', maxiter=200,lr=lr,
                         initial_values=init, silent=True)
    print('plotting what happens without QNG')
    bfgs_noqng_result.history.plot('energies')
    bfgs_noqng_result.history.plot('angles')
    print('best energy without qng: ',bfgs_noqng_result.energy)
    print('optimal angles without qng: ',bfgs_noqng_result.angles)

Numerical and Customized Gradients
----------------------------------

| By default ``tequila`` compiles analytical gradients of the objectives
  using ``jax``, internal recompilation and the parameter shift rule.
  The default is not setting the ``gradient`` keyword or setting it to
  ``None``. The keyword can also be set to a dictionary (keys are the
  variables, values are the ``tequila`` objectives which are assumed to
  evaluate to the corresponding gradients of the objective).
| For example ``gradient=tq.grad(objective)`` will results have the same
  results as ``gradient=None`` or simply not setting it.

``tequila`` offers its own way of compiling numerical gradients which
can then be used troughout all gradient based optimizers. It can be
activated by setting ``gradient`` to a dictionary holding the finite
difference stencil as ``method`` as well as the ``stepsize``.

Numerical gradients of that type come with the cost of
2\*\ ``len(variables)`` and can lead to significantly cheaper gradients,
especially if many expectation values are involved in the objective
and/or if heavy recompilation of parametrized gates is necessary. Here
is a small example using our ``O2`` objective, here the numerical
2-point procedure leads to 4 expectation values in the gradients (while
anayltial gradients would lead to 8, set silent to False in the upper
example or remove the gradient statement here).

.. code:: ipython3

    lr=0.01
    num_result = tq.minimize(objective=O2,
                         gradient={"method":"2-point", "stepsize":1.e-4},
                         method='sgd', maxiter=200,lr=lr,
                         initial_values=0.1, silent=False)

.. code:: ipython3

    num_result.history.plot('energies')

``tequila`` currently offers ``2-point`` as well as ``2-point-forward``
and ``2-point-backward`` stencils as ``method``. The method can also be
set to a python function performing the task. Here is an example which
implements the same as ``2-point``. The function can be replaced by any
function with the same signature.

.. code:: ipython3

    import copy
    def my_finite_difference_stencil(obj, var_vals, key, step, *args, **kwargs):
            """
            calculate objective gradient by symmetric shifts about a point.
            Parameters
            ----------
            obj: Objective:
                objective to call.
            var_vals:
                variables to feed to the objective.
            key:
                which variable to shift, i.e, which variable's gradient is being called.
            step:
                the size of the shift; a small float.
            args
            kwargs
    
            Returns
            -------
            float:
                the approximated gradient of obj w.r.t variable key at point var_vals[key] as a float.
    
            """
            left = copy.deepcopy(var_vals)
            left[key] += step / 2
            right = copy.deepcopy(var_vals)
            right[key] -= step / 2
            return 1.0 / step * (obj(left, *args, **kwargs) - obj(right, *args, **kwargs))
    
    num_result = tq.minimize(objective=O2,
                         gradient={"method":my_finite_difference_stencil, "stepsize":1.e-4},
                         method='sgd', maxiter=200,lr=lr,
                         initial_values=0.1, silent=True)
    num_result.history.plot('energies')

The ``gradient`` keyword can also be replaced by a dictionary of
``tequila`` objectives which evaluate to gradients approximations of it.

Bayesian optimization
---------------------

`Bayesian optimization <https://arxiv.org/abs/1807.02811>`__ is a method
of global optimization, often used to tune hyperparameters in classical
learning. It has also seen use in the optimization of `quantum
circuits <https://arxiv.org/pdf/1812.08862.pdf>`__. Tequila currently
supports 2 different bayesian optimization algorithms:
`Phoenics <https://github.com/aspuru-guzik-group/phoenics>`__ and
`GPyOpt <https://github.com/SheffieldML/GPyOpt>`__, optimizers
originally developed for optimizing expensive experimental procedures in
chemistry. Click the links to get to the respective github pages, and
download the optimizers before continuing this tutorial.

GPyOpt
~~~~~~

GPyOpt can be used like any of our other optimizers. Like the GD and
SciPy optimizers, it also takes a 'method' keyword. 3 methods are
supported: ``'lbfgs'``,\ ``'DIRECT'``, and ``'CMA'``. See the ``GPyOpt``
github for more info.

.. code:: ipython3

    print('As a reminder, we will optimize')
    tq.draw(O1,backend='qiskit')

.. code:: ipython3

    ### let's use the lbfgs method.
    init={'a':0.25,'b':0.25,'c':0.25,'d':0.25}
    ### note: no lr is passed here! there are fewer tunable keywords for this optimizer.
    result=tq.minimize(objective=O1,
                  method='lbfgs',
                  maxiter=80,
                  initial_values=init)
    
    print('GPyOpt optimization results:')
    result.history.plot('energies')
    result.history.plot('angles')
    print('best energy: ',result.energy)
    print('optimal angles: ',result.angles)

**Don't worry, the plot's not broken!** Perhaps you are looking at the
plots above in horror. But, do take note: bayesian optimization is a
global, exploratory optimization method, designed to explore large
portions of parameter space while still seeking out optimality. Look at
the optimal energy again, and one sees that the best performance of this
optimization method matched that of all the gradient descent methods. We
will apply gpyopt, next, to the QNG example circuit above, and see how
bayesian optimization compares to QNG and SGD.

.. code:: ipython3

    print('Hey, remember me?')
    tq.draw(O2)
    ### the keyword stop_count, below, stops optimization if no improvement occurs after 50 epochs.
    ### let's use a random initial starting point:
    init={k:np.random.uniform(-2,2) for k in ['a','b','c','d']}
    
    gpy_result = tq.minimize(objective=O2,maxiter=25,method='lbfgs',
                         initial_values=init)
    
    gpy_result.history.plot('energies')
    print('best energy: ',gpy_result.energy)
    print('optimal angles: ',gpy_result.angles)

**In a very, very small number of steps, GPyOpt is able to match the
performance of SGD with the QNG.**

**There's a few extras you can access if you are well-familiar with
GPyOpt.** We return as part of ``result`` an attribute
``result.gpyopt_instance``, which is an instance of the native
``GPyOpt`` ``BayesianOptimization`` object -- the one built and run
during your optimization. It has some plotting features you can use.

.. code:: ipython3

    obj=gpy_result.gpyopt_instance
    obj.plot_convergence()

If your function has 1 or 2 parameters (but no more) you can also see a
plot of its acquisition function! see
`here <https://www.blopig.com/blog/2019/10/a-gentle-introduction-to-the-gpyopt-module/>`__
for more info!

You can also extract the acquisition function of the model itself, and
play with it (it takes ones object, an np array, as input), using:

``acq=result.gpyopt_instance.acquisition.acquisition_function``

Phoenics
~~~~~~~~

Finally, we turn to
```Phoenics`` <https://github.com/aspuru-guzik-group/phoenics>`__. This
algorithm, originally developed within the Aspuru-Guzik group, can be
accessed in the usual fashion. It's performance on the two-qubit
optimization circuit is shown below. Note that the number of datapoints
exceeds the provided **maxiter**; **maxiter** here controls the number
of parameter **batches** suggested by phoenics. phoenics suggests a
number of parameter sets to try out, per batch, that scales with the
number of parameters (in a nonlinear fashion), so you may want to set
maxiter lower if you are only playing around.

.. code:: ipython3

    init={'a':0.25,'b':0.25,'c':0.25,'d':0.25}
    print('With phoenics we will optimize:')
    print(O1)
    ### to see what you can pass down to phoenics, see the tequila documentation for that module.
    p_result=tq.minimize(objective=O1,
                method='Phoenics',
                maxiter=5,
                initial_values=init,
                silent=False)
    
    print('Phoenics optimization results on 2 qubit circuit:')
    p_result.history.plot('energies')
    p_result.history.plot('angles')
    print('best energy: ',p_result.energy)
    print('optimal angles: ',p_result.angles)

We also have returned to you the phoenics object. One interesting object
we can extract from this is the acquisition function. You can obtain
this indirectly, using
resut.object.bayesian\_network.kernel\_contribution. This function takes
a numpy array ( a point in your parameter space) and returns 2 numbers,
x and y; the acquisition function then has the value x\*y. Note: this is
often zero.

.. code:: ipython3

    kc=p_result.phoenics_instance.bayesian_network.kernel_contribution
    random_point=np.random.uniform(0,2*np.pi,4)
    f,s=kc(random_point)
    random_ac=f*s
    print('random point ', random_point, ' has acquisition function value ',random_ac)

This concludes our tutorial. Hope you had fun! Happy optimizing!
----------------------------------------------------------------

