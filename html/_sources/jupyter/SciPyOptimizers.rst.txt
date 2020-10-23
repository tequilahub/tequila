Tequila Tutorial:
=================

Handling Optimizers, Initializing parametrized circuits
-------------------------------------------------------

.. code:: ipython3

    # Import everything we need here
    import tequila as tq
    numpy = tq.numpy

In the following example we will optimize a simple two qubit VQE with an
Ansatz that is parametrized by two parameters. The Hamiltonian is also
very simple and defined in the box below.

.. math::  H = \sigma_x(1) + c \sigma_z(1) - \sigma_z(0)

The tutorial is **not** a guideline to optimization strategies but only
intends to show tequilas functionality

.. code:: ipython3

    c = 0.5
    H = tq.paulis.X(1) + c*tq.paulis.Z(1) - tq.paulis.Z(0)

| Lets take a look at the spectrum of this very simple Hamiltonian
| Keep in mind: Tequila is not a numerical linear algebra package, so
  better not try to diagonalize Hamiltonians
| we however added this feature for user convenience in examples
| In the future we might have interfaces to more powerfull numerical
  packages ... feel free to contribute :-)

.. code:: ipython3

    matrix = H.to_matrix()
    e, v = numpy.linalg.eigh(matrix)
    e

And here we initialize our Ansatz circuit which is parametrized by a and
b.

We show to ways to initialize Variables: Variable a is initialized with
a convenience string based initialization Variable b is initialized as
Variable which allows you to conveniently transforming it when
initializing gates. Here we do a simple rescaling as an example Note
that you don't need strings to name variables but can use any hashable
non-numeric type (meaning anything that is not interpreted as number).

For example: tq.Variable(name=(0,1,2,3)) will also work

The Ansatz has no specific meaning and it is more to show different ways
how to deal with variables. See for instance that you can scale them and
use the same variable in multiple gates. (See the other tutorials for
more examples)

.. code:: ipython3

    a = tq.Variable(name="a")
    b = tq.Variable(name="b")
    c = tq.Variable(name="c")
    
    a = 2.0*a*tq.numpy.pi # optimize a in units of 2pi
    b = 2.0*b*tq.numpy.pi # optimize b in units of 2pi
    c = 2.0*c*tq.numpy.pi # optimize c in units of 2pi
    
    U = tq.gates.Ry(target=0, angle=a)
    U += tq.gates.Ry(target=1, control=0, angle=b)
    U += tq.gates.X(target=1)
    U += tq.gates.Ry(target=1, control=0, angle=-b)
    U += tq.gates.Ry(target=0,angle=c)


.. code:: ipython3

    print(U)

.. code:: ipython3

    # nice output (depends on which backends you have installed)
    tq.draw(U)

In the next box we form an objective out of our Hamiltonian and our
Ansatz and pass it down to the optimizer. In the following boxes we show
how the results of the optimizer can be plotted

See further below for some small exercises and additional information

First lets see how objectives are created and simulated

.. code:: ipython3

    O = tq.Objective.ExpectationValue(U=U, H=H)
    variables = {"a":0.25, "b":0.25, "c":0.25}
    energy = tq.simulate(O, variables=variables)
    wfn = tq.simulate(U, variables=variables)
    evaluate_squared = tq.simulate(O**2, variables=variables)
    print("energy : {}".format(energy))
    print("wfn    : {}".format(wfn))
    print("squared: {}".format(evaluate_squared))

Now the objective can be optimized

| We will also set the initial values of the variables that can be
  passed to the optimizer.
| Values of variables are passed as dictionaries where the keys are
  tequila variables and the values are floats.

.. code:: ipython3

    initial_values = {'a':0.3, 'b':0.3, 'c':0.3}
    O = tq.Objective.ExpectationValue(U=U, H=H)
    result = tq.minimize(objective=O, method="bfgs", initial_values=initial_values, tol=1.e-3, method_options={"gtol":1.e-3})

.. code:: ipython3

    # final energy
    result.energy

Plot out the History: Note, that we choose bad initial points since they
are close to the maximum

.. code:: ipython3

    result.history.plot('energies')

.. code:: ipython3

    result.history.plot(property='angles', key=["a", "b"])

.. code:: ipython3

    # Convenience in the history plot
    result.history.plot(property='angles', key="a")
    result.history.plot(property=['angles', 'gradients'], key=["c"])

FAQ
===

1: How can I extract the parameters from a given circuit?
---------------------------------------------------------

Call the 'extract\_parameters' attribute and get back a list of all
Variables in the circuit

.. code:: ipython3

    angles = U.extract_variables()
    angles

2: How can I do measurement based simulation?
---------------------------------------------

Pass down the 'samples' keyword to simulate finite samples See the later
exercises to play around with sample number and optimization methods.

Feel free to play around with the number of samples Don't excpect
miracles from the optimizer, you might need to hit return a few times or
increase the number of samples.Note that stochastic gradients are not
yet supported Note also that we did not set the initial parameters, so
we will start with all parameters set to 0 which is a stationary point
in this example (full wavefunction simulation would get stuck, see also
the exercise below).

Sampling based simulation needs improvement. Don't expect too much, but
feel free to contribute

Lets only optimize veriable b and set the other to the correct value to
also show how that works

.. code:: ipython3

    O = tq.Objective.ExpectationValue(U=U, H=H)
    initial_values["a"] = 0.3
    initial_values["c"] = 0.3
    initial_values["b"] = 0.3
    result = tq.minimize(objective=O, variables=["b"], initial_values = initial_values, tol=1.e-3, samples=100, method="bfgs")
    result.history.plot('energies')
    print("result = ", result.energy)

3: Which Simulator was used and how can I choose the simulator?
---------------------------------------------------------------

You can pass down the simulator to the optimizer by the simulator
keyword (see below) If no specific simulator was chosen by you that
means the simulator is automatically picked. Which simulator is picked
depends on what simulators you have installed and if you demanded a full
wavefunction to be simulated or not.

You can check which simulators you have installed with the following

.. code:: ipython3

    print(tq.show_available_simulators())

Here is how you would initialize a simulator and pass it down the
optimizer. The if statement is just to prevent your Ipython kernel from
crashing when you have not installed the simulator Feel free to change
it to something you have installed

.. code:: ipython3

    if 'qiskit' in tq.INSTALLED_BACKENDS: # failsafe to only execute cell when qiskit is actually there
        O = tq.Objective.ExpectationValue(U=U, H=H)
        result = tq.minimize(objective=O, method="bfgs",
                                           initial_values=initial_values,
                                           backend="qiskit")
        result.history.plot()

.. code:: ipython3

    result.history.plot('angles')

4: Can I use numerical evaluation of gradients
----------------------------------------------

| Yes you can, by passing down ``use_gradient = False`` or
  ``use_gradient = "2-point"``.
| ``use_gradient = '3-point' or 'cs'`` are also possible for scipy
  methods which support them.
| Check out the documentation of scipy.optimize.minimize for that.
| You can also pass down further options (again, check scipy
  documentation) for different methods.
| An important additional option is ``eps`` which defines the stepsizes
  for the '2-point' method.

.. code:: ipython3

    result = tq.minimize(objective=O, 
                         method="bfgs",
                         initial_values=initial_values,
                         use_gradient='2-point',
                         method_options = {'eps':1.e-3})

5: Can I use Hessian based optimizations, and can I evaluate Hessians numerically?
----------------------------------------------------------------------------------

Yes you can, by just picking those methods (like for example 'dogleg' or
'newton-cg'). For all 'trust-\*' methods you can also pick different
options (see again scipy documentation)

| Numerical evaluation for hessians works in the same way as for
  gradients by passing down ``use_hessian`` instead of ``use_gradient``.
| Be aware that not all combinations of ``use_gradient`` and
  ``use_hessian`` will work (usually you need to have the gradients
  analytically) and that most scipy methods do not support numerical
  evaluation of Hessians. Also do not confuse that with methods which
  use an approximation of the Hessian. Again: Check the scipy
  documentation for more information

Here comes a small example where (note that the methods above will not
converge to the minimum for this initial\_values)

.. code:: ipython3

    tq.optimizer_scipy.OptimizerSciPy.hessian_based_methods

.. code:: ipython3

    options = {
        "initial_tr_radius":0.05,
        "max_tr_radius":0.1
    }
    
    result = tq.optimizer_scipy.minimize(objective=O,
                                        initial_values={"a":0.25, "b":0.25, "c":0.25},
                                        method = "trust-exact",
                                        method_options = options)


Exercises
=========

See farther down for solutions

Exercise 1
----------

You can pass down initial\_values to the optimizer in the same format as
you can do it with the circuit (see above). Figure out how to do that by
checking out the documentation of the 'minimize' function.

Exercise 2
----------

Figure out which method the optimizer above used and how to use a
different optimization method.

.. code:: ipython3

    # hints
    tq.show_available_optimizers()

Exercise 3
----------

If you initialize both parameters to 0.0 you will directly hit a
stationary point which causes the optimizer stop. Find out how you can
impose bounds on the variables in order to prevent the optimizer from
hitting that point.

Again: Check the documentation of the 'minimize' function.

Note: Not all optimization methods of SciPy support bounds on the
variables

Note: It is not enough to just restrict the point 0.0

Solutions
=========

Exercise 1 & 2
--------------

.. code:: ipython3

    O = tq.Objective.ExpectationValue(U=U, H=H)
    result = tq.minimize(objective=O, method='Nelder-Mead', maxiter=100, initial_values={'a':0.1, 'b':0.1})

.. code:: ipython3

    result.history.plot('energies')
    result.history.plot('angles')

Exercise 3
----------

.. code:: ipython3

    # this will get stuck (similar for other gradient based optimizers)
    zeroes = {'a':0.0, 'b':0.0}
    shift = 1.0
    O = tq.Objective.ExpectationValue(U=U, H=H)
    result = tq.minimize(objective=O, method='TNC', initial_values=zeroes)
    result.history.plot()

.. code:: ipython3

    # bounding the variables to keep it away from the stationary point which occurs at 0 and is periodic in 2pi
    # using negative values since that converges faster
    zeroes = {'a':0.0, 'b':0.0}
    shift = 1.0
    bounds = {'a':(-2*numpy.pi+0.1, -0.1), 'b':(-2*numpy.pi+0.1, -0.1), 'c':(-2*numpy.pi+0.1, -0.1) }
    O = tq.Objective.ExpectationValue(U=U, H=H)
    result = tq.minimize(objective=O, method='TNC', initial_values=zeroes, method_bounds=bounds)
    result.history.plot()

.. code:: ipython3

    result.history.plot('angles')
