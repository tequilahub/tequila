Frequently Asked Questions:
===========================

It is recommended to take a look at the ``BasicUsage`` notebook before looking at this
--------------------------------------------------------------------------------------

.. code:: ipython3

    import tequila as tq
    import numpy

Which optimization methods can I use?
-------------------------------------

``tq.show_available_optimizers`` shows you all installed modules no your
systems and the methods which ``tq.minimize`` understands. Method names
are not case sensitive when passed to ``tq.minimize``.

| In the end you see which modules are supported and which of them are
  actually installed on your system.
| The table with methods and modules will only show you the methods for
  modules that are currently installed within your environment.

| Of course you can also use tequila objectives for your own optimizers.
| You don't need to use the modules here.

.. code:: ipython3

    tq.show_available_optimizers()

Which simulators/Quantum-Backends can I use?
--------------------------------------------

``tq.show_available_simulators`` shows all simulators/quantum backends
which are supported by tequila as well as which are installed within
your current environment.

The default choice if you don't specify a backend when for example
simulating a tequila objective with ``tq.simulate`` is the first entry
of the supported backends which is installed on your system.

.. code:: ipython3

    tq.show_available_simulators()

Can I avoid re-translation/compilation on my objectives/circuits?
-----------------------------------------------------------------

| Yes you can. By calling ``tq.compile`` instead of ``tq.simulate``.
  This will give you back a callable objective.
| Check also the ``basic usage`` tutorial notebook

.. code:: ipython3

    U = tq.gates.H(target=1) + tq.gates.Rx(angle="a", target=0, control=1)
    
    # simulate the wavefunction with different variables
    wfn0 = tq.simulate(U, variables={"a": 1.0})
    wfn1 = tq.simulate(U, variables={"a": 2.0})
    
    print(wfn0)
    print(wfn1)
    
    # the same, but avoiding re-compilation
    # Note that your compiled object is translated to a quantum backend
    # if the backend was not set, tequila it will pick the default which depends
    # on which backends you have installed. You will seee it in the printout of the
    # compiled circuits
    compiled_U = tq.compile(U)
    wfn0 = compiled_U(variables={"a":1.0})
    wfn1 = compiled_U(variables={"a":2.0})
    
    print("compiled circuit:", compiled_U)
    print(wfn0)
    print(wfn1)
    
    
    # With Objectives it works in the same way
    H = tq.paulis.Y(0)
    E = tq.ExpectationValue(H=H, U=U)
    objective = E**2 + 1.0
    
    # simulate the objective with different variables
    result0 = tq.simulate(objective, variables={"a": 1.0})
    result1 = tq.simulate(objective, variables={"a": 2.0})
    
    print("compiled objective:", objective)
    print(result0)
    print(result1)
    
    # compile and then simulate
    compiled_objective = tq.compile(objective)
    result0 = compiled_objective(variables={"a":1.0})
    result1 = compiled_objective(variables={"a":2.0})
    
    print("compiled objective:", compiled_objective)
    print(result0)
    print(result1)

How can I run on a real quantum computer?
-----------------------------------------

| Tequila can both emulate -- and when possible, operate via -- real
  quantum devices. For example: IBM's Qiskit can be used to run on some
  of IBM's public accessible quantum computers. All you need for this is
  an ibm account (Follow the instructions under "Configure your IBM
  Quantum Experience credentials" here:
  https://github.com/Qiskit/qiskit-ibmq-provider).
| Tequila also supports Rigetti's PyQuil and Google's Cirq, but
  currently there are no publicly available devices.

Here is a small example with Qiskit (you need to have qiskit installed,
and an activaged IBMQ account for this). Alternatively you can also
externally initialize your chosen device, and pass this down instead of
a string.

You always need to set samples if you intend to run on a real (or
emulated) backend.

| If you have special access rights you can initialize the ``qiskit``
  quantum backend yourself and pass it down as ``device`` instead of the
  device name.
| ``device = provider.get_backend(name)`` or as a dictionary with
  ``qiskit`` provider and devicename
| ``device = {"provider":provider_instance, "name":device_name}``

Here is a small toy example that minimizes the square of a one qubit
expectation value (minimum is 0.0)

.. code:: ipython3

    import tequila as tq
    U = tq.gates.Ry(angle="a", target=0)
    H = tq.paulis.X(0)
    E = tq.ExpectationValue(H=H, U=U)
    
    # simulate the square of the expectation value with a specific set of variables
    result = tq.simulate(E**2, variables={"a":1.0}, samples=1000, backend="qiskit", device='ibmq_ourense'
    
    # optimize using ond of IMB's quantum computers as quantum backend
    # (check your ibm account for more information and keywords)
    # note that the names of the computer might have changed  
    result = tq.minimize(objective=E**2, method="cobyla", initial_values={"a":1.0}, samples=1000, backend="qiskit", device='ibmq_ourense')

How can I emulate a real quantum computer?
------------------------------------------

Emulation is performed similarly to running on real devices. All you
need to do is pass down the right string to the 'device' keyword. For
qiskit, these are the same as for regular backends, but have'fake\_' at
the beginning; I.E to emulate 'armonk;, set ``device="fake_armonk"``.
For PyQuil, this is done by adding '-qvm' to the end of the chosen
string, i.e, 'Aspen-8' becomes ``device=Aspen-8-qvm'``. For Cirq, only
emulation is currently possible; the only string options for cirq are
'foxtail','bristlecone','sycamore', and 'sycamore23'.

When emulating, a few things about the real device will be mimicked,
principally its native gate set and its connectivities. Emulation will
NOT include noisy emulation by default; If you want to emulate noise,
pass down the keyword ``noise='device'``. Using this option without
specifying a device will result in an error.

Below, we will emulate pyquil's Aspen 8, with emulated noise. You need
pyquil installed for this to work.

additionally: when real backends cannot be accessed, emulation will be
attempted, with a warning.

.. code:: ipython3

    U = tq.gates.Ry(angle="a", target=0)
    H = tq.paulis.X(0)
    E = tq.ExpectationValue(H=H, U=U)
    
    # simulate the square of the expectation value with a specific set of variables
    result = tq.simulate(E**2, variables={"a":1.0}, samples=1000, backend="pyquil")
    print('sampling from pyquil yielded: ', result)
    result = tq.simulate(E**2, variables={"a":1.0}, samples=1000, backend="pyquil",device='Aspen-8-qvm')
    print('sampling from pyquil while emulating Aspen-8 yielded: ', result)
    result = tq.optimizer_scipy.minimize(E**2, initial_values={"a":1.0}, samples=1000,
                                        backend='pyquil', device="Aspen-8-qvm", 
                                        noise='device')
    print('optimizing while emulating Aspen-8 with noise yielded a best energy of: ', result.energy)

Can I compile Objectives into different backends?
-------------------------------------------------

Yes you can. Tequila will print a warning if this happens. Warnings can
be ignored by filtering them out (see the python warnings documentation)

If a compiled circuit is used as input to compile then tequila will
re-compile the circuit to the new backend (it it differs from the
previous one)

If a compiled objective is used as input to compile then tequila will
only compile non-compiled expectationvalues into the different backend.
Already compiled expectation values will remain untouched

| Note that you need at least two different backends for the following
  cell to execute.
| Just change the key to whatever you have installed.

.. code:: ipython3

    backend1 = "qulacs"
    backend2 = "cirq"
    
    U = tq.gates.X(target=[0,1])
    print("Example Circuit: ", U)
    compiled_1 = tq.compile(U, backend=backend1)
    compiled_2 = tq.compile(compiled_1, backend=backend2)
    print("Circuit compiled to {} -> ".format(backend1), compiled_1)
    print("Circuit compiled to {} -> ".format(backend1), compiled_1)
    
    H = tq.paulis.X(0)*tq.paulis.Y(1) + tq.paulis.X([0,1])
    print("\nmake objective with H = ", H)
    objective = tq.ExpectationValue(H=H, U=U)
    compiled_1 = tq.compile(objective, backend=backend1)
    
    print("\nExpectationValues of objective 1:")
    print(compiled_1)
        
    objective2 = compiled_1 + objective # Its recommended to avoid those hybrids, but in principle it works
    
    print("\nExpectationValues of partly compiled objective:")
    print(objective2)
        
    compiled_2 = tq.compile(objective2, backend=backend2)
    print("\nExpectationValues of hybdrid compiled objective:")
    print(compiled_2)
    
    


How do I transform Measurements into Hamiltonians?
--------------------------------------------------

We can not answer this question in general, but we can try to give a
small example here.

Assume you have a quantum circuit with :math:`4` Qubits and you are
measuring Qubit :math:`0` and :math:`2`. You define your cost function
in the following way:

.. math::


   L(AB) = A + B, \qquad A,B \in \left\{ 0,1 \right\}  

meaning you accumulate the number of ones measured in your circuit.

The corresponding expectationvalue would be

.. math::


   L = \langle \Psi \rvert H \lvert \Psi \rangle \qquad H = 1 - \frac{1}{2}\left(Z(0) + Z(1)\right) 

The Hamiltonian could also be written as

.. math::


   H = 2\lvert 11 \rangle \langle 11 \rvert + \lvert 10 \rangle \langle 10 \rvert + \lvert 01 \rangle \langle 01 \rvert

Tequila provides the convenience function ``tq.gates.Projector`` to
initialize Hamiltonians like that

.. code:: ipython3

    2*tq.paulis.Projector("|11>") + tq.paulis.Projector("|01>") + tq.paulis.Projector("|10>")

| The projector can also be initialized with more structured
  ``QubitWaveFunction``\ s which can itself be initialized from array or
  string.
| Here are some examples

.. code:: ipython3

    wfn = tq.QubitWaveFunction.from_string("1.0*|00> + 1.0*|11>")
    wfnx = tq.QubitWaveFunction.from_array(arr=[1.0, 0.0, 0.0, 1.0])
    print(wfn == wfnx)
    wfn = wfn.normalize()
    print(wfn)
    
    P = tq.paulis.Projector(wfn=wfn)
    print("P = ", P)

Apart from ``Projector`` there is also ``KetBra`` which intialized more
general operators like

.. math::


   \lvert \Psi \rangle \langle \Phi \rvert

| Keep in mind that those are not hermitian.
| But they can be split up into their hermitian and anti hermitian part
  where both can then be used as hamiltonians for expectationvalues.

If the ``hermitian = True`` key is set, the function returns the
hermitian version of the operator (which is the same as the hermitian
part of the old operator)

.. math::


   \frac{1}{2}\left(\lvert \Psi \rangle \langle \Phi \rvert + \lvert \Phi \rangle \langle \Psi \rvert \right)

.. code:: ipython3

    wfn1 = tq.QubitWaveFunction.from_string("1.0*|00> + 1.0*|11>").normalize()
    
    op = tq.paulis.KetBra(bra=wfn1, ket="|00>")
    
    H1, H2 = op.split()
    
    print("operator=", op)
    print("hermitian part      = ", H1)
    print("anti-hermitian part =", H2)
    
    H = tq.paulis.KetBra(bra=wfn1, ket="|00>", hermitian=True)
    print("hermitianized operator = ", H)


Can I do basic operations on wavefunctions and operators without quantum backends?
----------------------------------------------------------------------------------

| In principle yes. But keep in mind that tequila was not made for this.
| However, some of those operations might come in handy for debugging or
  small examples.

You can not execute circuits without a simulator since they are just
abstract data types (no matrices or anything). Tequila has however its
own small debug simulator ``backend = symbolic`` but there is no reason
to use it if you have any other quantum backend installed.

Hamiltonians can be converted to matrices.

We give a few examples here

.. code:: ipython3

    wfn = tq.QubitWaveFunction.from_string("1.0*|0> + 1.0*|1>").normalize()
    H = 1.0/numpy.sqrt(2.0)*(tq.paulis.Z(0) + tq.paulis.X(0))
    wfn2 = wfn.apply_qubitoperator(H).simplify()
    
    print("|wfn>  = ", wfn)
    print("H      = ", H)
    print("H|wfn> = ", wfn2)

.. code:: ipython3

    wfn1 = tq.QubitWaveFunction.from_string("1.0*|0> + 1.0*|1>").normalize()
    wfn2 = tq.QubitWaveFunction.from_string("1.0*|0> - 1.0*|1>").normalize()
    print("<wfn1|wfn2> = ", wfn1.inner(wfn2))

.. code:: ipython3

    H = 1.0/numpy.sqrt(2.0)*(tq.paulis.Z(0) + tq.paulis.X(0))
    print(H.to_matrix())

Can I import an Hamiltonian from OpenFermion?
---------------------------------------------

Yes! OpernFermion is currently tequilas backend for Hamiltonians, which
makes importing from it straight forward. You just need to wrap the
OpenFermion QubitOperator into tequilas QubitHamiltonian.

We show a few examples

.. code:: ipython3

    from openfermion import QubitOperator
    
    # get OpenFermion QubitOperator from tequila QubitHamiltonian
    H = tq.paulis.X(0)
    of_operator = H.to_openfermion()
    
    print("{} = {}".format(type(H), H))
    print("{} = {}".format(type(of_operator), of_operator))
    
    # init tequila QubitHamiltonian with OpenFermion QubitOperator
    H = tq.QubitHamiltonian.from_openfermion(of_operator)
    print("{} = {}".format(type(H), H))
    
    # initialization from file os often read in the string form
    of_string = str(of_operator)
    tq_string = str(H)
    
    print(of_string)
    print(tq_string)
    
    H = tq.QubitHamiltonian.from_string(string=of_string, openfermion_format=True)
    print(H)
    H = tq.QubitHamiltonian.from_string(string=tq_string, openfermion_format=False)
    print(H)

Can I compile into a regular function instead of one which takes dictionaries?
------------------------------------------------------------------------------

Not recommended but yes. The order of the function arguments is the
order you get from ``extract_variables``

.. code:: ipython3

    U = tq.gates.Ry(angle="a", target=0)
    U += tq.gates.X(power = "b", target=1)
    H = tq.QubitHamiltonian.from_string("X(0)Z(1)")
    E = tq.ExpectationValue(H=H, U=U)
    
    f = tq.compile_to_function(E)
    
    print("order is : ", E.extract_variables())
    print(f(0.5, 1.0))
    print(tq.simulate(E, variables={"a":0.5, "b":1.0}))


If you also want to fix the samples and other entries to your compiled
objective you can build wrappers

.. code:: ipython3

    def mywrapper(compiled_obj, samples):
        return lambda *x: compiled_obj(*x, samples=samples)
    
    wrapped = mywrapper(f, samples=100)
    
    # don't expect same results, since samples are taken individually
    print(wrapped(1.0, 0.5)) # always takes 100 samples
    print(f(1.0, 0.5, samples=100)) # samples need to be given
    print(f(1.0, 0.5, samples=1000)) # but sampling rate can be changed
    print(f(1.0, 0.5)) # you can go back to full simulation which you cannot with the wrapped function

How do numerical gradients work?
--------------------------------

Yes this is possible by passing for example
``gradient={'method':'2-point', 'stepsize': 1.e-4}`` to the
``tq.minimize`` function.

The default is a central 2-point derivative stencil where ``h`` is the
stepsize:

.. math::

   \displaystyle
   \frac{\partial f}{\partial a} = \frac{f(a+\frac{h}{2}) - f(a-\frac{h}{2})} {h}

Other methods are: ``2-point-forward``: Forward derivative stencil:

.. math::

   \displaystyle
   \frac{\partial f}{\partial a} = \frac{f(a+h) - f(a)} {h}

``2-point-backward``: Backward derivative stencil:

.. math::

   \displaystyle
   \frac{\partial f}{\partial a} = \frac{f(a) - f(a-h)} {h}

| You can also use your own numerical derivative stencil by passing a
  callable function as ``method``.
| The function should have the signature which is given in the example
  below.

Here is an example:

.. code:: ipython3

    import tequila as tq
    # form a simple example objective
    H = tq.paulis.X(0)
    U = tq.gates.Ry(angle="a", target=0)
    E = tq.ExpectationValue(U=U, H=H)
    
    # make it more interesting by using analytical gradients for the objective
    # and numerical gradients to optimize it
    
    objective = tq.grad(E, 'a')**2 # integer multiples of pi/2 are minima
    
    # start from the same point in all examples
    initial_values = {'a': 2.3}

.. code:: ipython3

    # optimize with analytical derivatives
    result = tq.minimize(objective=objective, method="bfgs", initial_values=initial_values)
    #result.history.plot("energies")
    #result.history.plot("gradients")
    
    # optimize with 2-point stencil
    result = tq.minimize(objective=E, method="bfgs", gradient={'method': '2-point', 'stepsize':1.e-4}, initial_values=initial_values)
    #result.history.plot("energies")
    #result.history.plot("gradients")
    
    # optimize with custom stencil
    # here this is the same as the default
    import copy
    def mymethod(obj, angles, key, step, *args, **kwargs):
        left = copy.deepcopy(angles)
        left[key] += step / 2
        right = copy.deepcopy(angles)
        right[key] -= step / 2
        return 1.0 / step * (obj(left, *args, **kwargs) - obj(right, *args, **kwargs))
    
    result = tq.minimize(objective=E, method="bfgs", gradient={'method': mymethod, 'stepsize':1.e-4}, initial_values=initial_values)
    #result.history.plot("energies")
    #result.history.plot("gradients")
    
    # optimize with a scipy method and use scipys 2-point
    # the scipy protocol will have more function evaluations and less gradient evaluations for some methods
    # the stepsize in scipy has to be passed with the `method_options` dictionary
    # with the keyword `eps`
    result = tq.minimize(objective=E, method="bfgs", gradient='2-point', method_options={'eps':1.e-4}, initial_values=initial_values)
    #result.history.plot("energies")
    


Can I use the numerical gradient protocols from SciPy?
------------------------------------------------------

| Yes you can for all scipy methods by passing ``gradient='2-point'`` to
  ``tq.minimize``.
| See the scipy documentation for the stepsize and more options which
  can be passed with ``method_options`` dictionary where the key for the
  stepsize is usually ``eps``. Note that not all scipy methods support
  numerical gradients,but cyou can always fall back to tequilas
  numerical gradients. See the previous cell for an example.

