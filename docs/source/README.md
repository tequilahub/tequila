![Image](_static/tequila_blackboard.svg)

# Tequila

Tequila is an Extensible Quantum Information and Learning Architecture where the main goal is to simplify and accelerate implementation of new ideas for quantum algorithms. 
It operates on abstract data structures allowing the formulation, combination, automatic differentiation and optimization of generalized objectives.
Tequila can execute the underlying quantum expectation values on state of the art simulators as well as on real quantum devices.

[You can get an overview from this presentation](tequila_presentation)  

[Get started with our Tutorials](https://github.com/aspuru-guzik-group/tequila/blob/master/tutorials)

# Quantum Backends
Currently supported
- [Qulacs](https://github.com/qulacs/qulacs)
- [Qiskit](https://github.com/qiskit/qiskit)
- [Cirq](https://github.com/quantumlib/cirq)
- [PyQuil](https://github.com/rigetti/pyquil)

Tequila detects backends automatically if they are installed on your systems.  
All of them are available over standard pip installation like for example `pip install qulacs`.  
For best performance tt is recommended to have `qulacs` installed.

# QuantumChemistry:
Tequila supports [Psi4](https://github.com/psi4/psi4).  
In a conda environment this can be installed with  
`conda install psi4 -c psi4`

# Installation
We recommend installing in editable mode with  
`git clone https://github.com/aspuru-guzik-group/tequila.git`  
`cd tequila`   
`pip install -e .`  

**Do not** install over PyPi (Minecraft lovers excluded)  
<strike>`pip install tequila`</strike>

Recommended Python version is 3.7 or 3.6

# Getting Started
Check out the tutorial notebooks provided in tutorials.

## Tequila Hello World
```python
# optimize a one qubit example

# define a variable
a = tq.Variable("a")
# define a simple circuit
U = tq.gates.Ry(angle=a*pi, target=0)
# define an Hamiltonian
H = tq.paulis.X(0)
# define an expectation value
E = tq.ExpectationValue(H=H, U=U)
# optimize the expectation value
result = tq.minimize(method="bfgs", objective=E**2)
# check out the optimized wavefunction
wfn = tq.simulate(U, variables=result.angles)
print("optimized wavefunction = ", wfn)
# plot information about the optimization
result.history.plot("energies")
result.history.plot("angles")
result.history.plot("gradients")
```

## Chemistry Hello World
```python
# define a molecule within an active space
active = {"a1": [1], "b1":[0]}
molecule = tq.quantumchemistry.Molecule(geometry="lih.xyz", basis_set='6-31g', active_orbitals=active, transformation="bravyi-kitaev")

# get the qubit hamiltonian
H = molecule.make_hamiltonian()

# make the UCCSD ansatz with cc2 ordering
U = molecule.make_uccsd_ansatz(initial_amplitudes="cc2", trotter_steps=1)

# define the expectationvalue
E = tq.ExpectationValue(H=H, U=U)

# compute reference energies
fci = molecule.compute_energy("fci")
cisd = molecule.compute_energy("detci", options={"detci__ex_level": 2})

# optimize
result = tq.minimize(objective=E, method="BFGS", gradient="2-point", method_options={"eps":1.e-3}, initial_values={k:0.0 for k in E.extract_variables()})

print("VQE : {:+2.8}f".format(result.energy))
print("CISD: {:+2.8}f".format(cisd))
print("FCI : {:+2.8}f".format(fci))
```

# Dependencies
Support for additional optimizers can be activated by intalling them in your environment.  
Tequila will then detect them automatically.  
Currently those are: [Phoenics](https://github.com/aspuru-guzik-group/phoenics)
 and [GPyOpt](https://sheffieldml.github.io/GPyOpt/).

# Documentation
You can build the documentation by navigating to `docs` and entering `make html`.  
Open the documentation with a browser over like `firefox docs/build/html/index.html`

# Troubleshooting
If you experience trouble of any kind or if you either want to implement a new feature or want us to implement a new feature that you need
don't hesitate to contact us directly or raise an issue here on github

## Windows
You can in principle use tequila with windows as OS and have almost full functionality.  
You will need to replace Jax with autograd for it to work.  
In order to do so: Remove `jax` and `jaxlib` from `setup.py` and `requirements.txt` and add `autograd` instead.

In order to install qulacs you will need latest GNU compilers (at least gcc-7).  
They can be installed for example over visual studio.

## Mac OS
Tequila runs on Macs. You might get in trouble with the installing qulacs since it currently does not work with Apples clang compiler. You need to install latest GNU compilers and set them as default before installing qulacs over pip.

