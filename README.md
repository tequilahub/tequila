# Tequila

Tequila is an Extensible Quantum Information and Learning Architecture where the main goal is to simplify and accelerate implementation of new ideas for quantum algorithms.   
It operates on abstract data structures allowing the formulation, combination, automatic differentiation and optimization of generalized objectives.
Tequila can execute the underlying quantum expectation values on state of the art simulators as well as on real quantum

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
clone this repository, cd to the main directory (where `setup.py` is located) and hit  
`pip install .`  
We recommend installing in developer mode with  
`pip install -e .`

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

# Dependencies
Support for additional optimizers can be activated by intalling them in your environment.  
Tequila will then detect them automatically.  
Currently those are: [Phoenics](https://github.com/aspuru-guzik-group/phoenics)
 and [GPyOpt](https://sheffieldml.github.io/GPyOpt/).

# Troubleshooting
If you experience trouble of any kind or if you either want to implement a new feature or want us to implement a new feature that you need
don't hesitate to contact us directly or raise an issue here on github
