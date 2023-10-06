[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENCE) [![DOI](https://zenodo.org/badge/259718912.svg)](https://zenodo.org/badge/latestdoi/259718912) [![PyPI version](https://badge.fury.io/py/tequila-basic.svg)](https://badge.fury.io/py/tequila-basic) ![CI](https://github.com/tequilahub/tequila/actions/workflows/ci_basic.yml/badge.svg)

# Tequila

Tequila is an abstraction framework for (variational) quantum algorithms.  
It operates on abstract data structures allowing the formulation, combination, automatic differentiation and optimization of generalized objectives.
Tequila can execute the underlying quantum expectation values on state of the art simulators as well as on real quantum devices.  

- [overview article](https://arxiv.org/abs/2011.03057)   
- [tequila in a nutshell](https://kottmanj.github.io/tequila-in-a-nutshell/#/)  
- [getting started](https://jakobkottmann.com/posts/tq-get-started/)    
- [circuits in tequila](https://jakobkottmann.com/posts/tq-circuits/)  
- [notebook collection](https://github.com/tequilahub/tequila-tutorials)  
- [talks and slides](https://kottmanj.github.io/talks_and_material/)  

# Installation
Recommended Python version is 3.8-3.9.   
Tequila supports linux, osx and windows. However, not all optional dependencies are supported on windows.  

## Install from PyPi
**Do not** install like this: (Minecraft lovers excluded)
<strike>`pip install tequila`</strike>

You can install tequila from PyPi as:
```bash
pip install tequila-basic
```
this will install tequila with all essential dependencies.
We recommend to install some fast quantum backends, like qulacs or qibo, as well.
Those can be installed before or after you install tequila.
```bash
# install basic tequila
pip install tequila-basic
# install qulacs and/or other backends and use it within tequila
pip install qulacs
```

## Install from github 
You can install `tequila` directly with pip over:
```bash
pip install git+https://github.com/tequilahub/tequila.git
```
Install from devel branch (most recent updates):
```bash
pip install git+https://github.com/tequilahub/tequila.git@devel
```

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

## Chemistry Hello World (Madness backend)   
see below for installation of dependencies  
```python
import tequila as tq

# initialize molecule (also works over .xyz files --> see next example)
geomstring="Li 0.0 0.0 0.0\nH 0.0 0.0 1.6"
mol = tq.Molecule(geometry=geomstring)

# get the qubit hamiltonian
H = mol.make_hamiltonian()

# get the ansatz (circuit)
U = mol.make_ansatz(name="SPA") # or e.g. UpCCGSD

# define the expectation value
E = tq.ExpectationValue(H=H, U=U)

# minimize the expectation value
result = tq.minimize(E)

# optional:compute classical reference energies
# needs pyscf as well
cisd = mol.compute_energy("cisd")
fci = mol.compute_energy("fci")

print("VQE : {:+2.8}f".format(result.energy))
print("CISD: {:+2.8}f".format(cisd))
print("FCI : {:+2.8}f".format(fci))

```

## Chemistry Hello World (Psi4 or PySCF backend)
see below for installation of dependencies  
```python
# define a molecule within an active space
active_orbitals=[1,2,5]
molecule = tq.quantumchemistry.Molecule(geometry="lih.xyz", basis_set='6-31g', active_orbitals=active, transformation="bravyi-kitaev")

# get the qubit hamiltonian
H = molecule.make_hamiltonian()

# create an k-UpCCGSD circuit of order k
U = molecule.make_ansatz(name="UpCCGSD")

# define the expectationvalue
E = tq.ExpectationValue(H=H, U=U)

# compute reference energies
fci = molecule.compute_energy("fci")
cisd = molecule.compute_energy("detci", options={"detci__ex_level": 2})

# optimize
result = tq.minimize(objective=E, method="BFGS", initial_values=0.0)

print("VQE : {:+2.8}f".format(result.energy))
print("FCI : {:+2.8}f".format(fci))
```

Do you want to create your own methods? Check out the [tutorials](https://github.com/tequilahub/tequila-tutorials)!

# Some Research projects using Tequila
J.S. Kottmann, A. Anand, A. Aspuru-Guzik.  
A Feasible Approach for Automatically Differentiable Unitary Coupled-Cluster on Quantum Computers.  
Chemical Science, 2021, [doi.org/10.1039/D0SC06627C](https://doi.org/10.1039/D0SC06627C).  
[arxiv:2011.05938](https://arxiv.org/abs/2011.05938)  
General techniques are implemented in the chemistry modules of tequila.  
See the [tutorials](https://github.com/aspuru-guzik-group/tequila-tutorials) for examples.  

J.S. Kottmann, P. Schleich, T. Tamayo-Mendoza, A. Aspuru-Guzik.  
Reducing Qubit Requirements while Maintaining Numerical Precision for the Variational Quantum Eigensolver: A Basis-Set-Free Approach.  
J.Phys.Chem.Lett., 2021, [doi.org/10.1021/acs.jpclett.0c03410](https://doi.org/10.1021/acs.jpclett.0c03410).  
[arxiv:2008.02819](https://arxiv.org/abs/2008.02819)  
[example code](https://github.com/aspuru-guzik-group/tequila-tutorials/blob/main/chemistry/BasisSetFreeVQEExample.ipynb)  
[tutorial on the madness interface](https://nbviewer.org/github/tequilahub/tequila-tutorials/blob/main/chemistry/MadnessInterface.ipynb)  

A. Cervera-Lierta, J.S. Kottmann, A. Aspuru-Guzik.  
The Meta-Variational Quantum Eigensolver.  
[arxiv:2009.13545](https://arxiv.org/abs/2009.13545)  
[example code](https://github.com/aspuru-guzik-group/Meta-VQE)    

J.S. Kottmann, M. Krenn, T.H. Kyaw, S. Alperin-Lea, A. Aspuru-Guzik.  
Quantum Computer-Aided design of Quantum Optics Hardware.  
[arxiv:2006.03075](https://arxiv.org/abs/2006.03075)  
[example code](https://github.com/kottmanj/Photonic)  
[slides](https://github.com/kottmanj/Photonic/blob/master/slides.pdf)  

A. Anand, M. Degroote, A. Aspuru-Guzik.  
Natural Evolutionary Strategies for Variational Quantum Computation.  
[arxiv:2012.00101](https://arxiv.org/abs/2012.00101)  

J. S. Kottmann, A. Aspuru-Guzik,  
Optimized Low-Depth Quantum Circuits for Molecular Electronic Structure using a Separable Pair Approximation,  
[arxiv:2105.03836](https://arxiv.org/abs/2105.03836)  
[example code](https://nbviewer.org/github/tequilahub/tequila-tutorials/blob/main/chemistry/SeparablePairAnsatz.ipynb)   
 
K. Choudhary,  
Quantum Computation for Predicting Electron and Phonon Properties of Solids  
[arxiv:2102.11452](https://arxiv.org/abs/2102.11452)  


P. Schleich, J.S. Kottmann, A. Aspuru-Guzik,  
Improving the Accuracy of the Variational Quantum Eigensolver for Molecular Systems by the Explicitly-Correlated Perturbative [2]-R12-Correction  
[arxiv:2110.06812](https://arxiv.org/abs/2110.06812)  
[tutorial](https://nbviewer.org/github/tequilahub/tequila-tutorials/blob/main/chemistry/F12Correction.ipynb)  

M. Weber, A. Anand, A. Cervera-Lierta, J. S. Kottmann, T.-H. Kyaw, B. Li, A. Aspuru-Guzik, C. Zhang and Z. Zhao,  
Toward Reliability in the NISQ Era: Robust Interval Guarantee for Quantum Measurements on Approximate States  
[arxiv:2110.09793](https://arxiv.org/abs/2110.09793)  
[tutorial](https://nbviewer.org/github/tequilahub/tequila-tutorials/blob/main/research/RobustnessIntervals.ipynb)  
  
M. S. Rudolph, S. Sim, A. Raza, M. Stechly, J. R. McClean, E. R. Anschuetz, L. Serrano, A. Perdomo-Ortiz  
ORQVIZ: Visualizing High-Dimensional Landscapes in Variational Quantum Algorithms  
[arxiv:2111.04695](https://arxiv.org/abs/2111.04695)  

P. Schleich    
Regularization of Quantum Chemistryon Quantum Computers by means of Explicit Correlation  
[Master thesis](http://www.acom.rwth-aachen.de/_media/3teaching/00projects/2020_ma_schleich.pdf)  
  
T.-H. Kyaw, T. Menke, S. Sim, A. Anand, N. P. D. Sawaya, W. D. Oliver, G. G. Guerreschi, A. Aspuru-Guzik  
Quantum computer-aided design: digital quantum simulation of quantum processors  
[arxiv:2006.03070](https://arxiv.org/abs/2006.03070)  

Z. P. Bansingh, T.-C. Yen, P. D. Johnson, A. F. Izmaylov  
Fidelity overhead for non-local measurements in variational quantum algorithms  
[arxiv:2205.07113](https://arxiv.org/abs/2205.07113)  


H. Lim, H.-N. Jeon, J.-K. Rhee, B. Oh, K. T. No  
Quantum computational study of chloride ion attack on chloromethane for chemical accuracy and quantum noise effects with UCCSD and k-UpCCGSD ansatzes  
[arxiv:2112.15314](https://arxiv.org/abs/2112.15314)  

A, Meijer- van de Griend, J. K. Nurminen  
QuantMark: A Benchmarking API for VQE Algorithms  
[DOI:10.1109/TQE.2022.3159327](https://doi.org/10.1109/TQE.2022.3159327)  
[QuantMark Codebase](https://github.com/QuantMarkFramework/LibMark/)  

A. Anand, J.S. Kottmann, A. Aspuru-Guzik  
Quantum compression with classically simulatable circuits  
[code](https://github.com/AbhinavUofT/GA_for_encoder)  
[arxiv:2207.02961](https://arxiv.org/abs/2207.02961)  

J.S. Kottmann  
Molecular Circuit Design: A Graph-Based Approach  
[arxiv:2207.12421](https://arxiv.org/abs/2207.12421)  
[example code](https://github.com/tequilahub/tequila-tutorials/blob/main/chemistry/GraphBasedCircuitDesign.ipynb)  

T.-H. Kyaw, M. B. Soley, B. Allen, P. Bergold, C. Sun, V. S. Batista, A. Aspuru-Guzik  
Variational quantum iterative power algorithms for global optimization  
[arxiv:2208.10470](https://arxiv.org/abs/2208.10470)  
[code](https://github.com/aspuru-guzik-group/qipa)  

R.A. Lang, A. Ganeshram, A. Izmaylov  
Growth reduction of similarity transformed electronic Hamiltonians in qubit space  
[arxiv:2210.03875](https://arxiv.org/abs/2210.03875)  
  
K. Gratsea, C. Sun, P.D. Johnson  
When to Reject a Ground State Preparation Algorithm  
[arxiv:2212.09492](https://doi.org/10.48550/arXiv.2212.09492)  

R.P. Pothukuchi, L. Lufkin, Y.J. Shen, A. Simon, R. Thorstenson, B.E. Trevisan, M. Tu, M. Yang, B. Foxman, V. S. Pothukuchi, G. Epping, B. J. Jongkees, T.-H. Kyaw, J. R. Busemeyer, J. D Cohen, A. Bhattacharjee  
Quantum Cognitive Modeling: New Applications and Systems Research Directions  
[arxiv:2309.00597](https://arxiv.org/abs/2309.00597)  

T.-H. Kyaw, M. B. Soley, B. Allen, P. Bergold, C. Sun, V.S. Batista and A. Aspuru-Guzik  
Boosting quantum amplitude exponentially in variational quantum algorithms  
[10.1088/2058-9565/acf4ba](doi.org/10.1088/2058-9565/acf4ba)  

A.G. Cadavid, I. Montalban, A. Dalal, E. Solano, N.N. Hegade  
Efficient DCQO Algorithm within the Impulse Regime for Portfolio Optimization  
[arxiv:2308.15475](https://arxiv.org/abs/2308.15475)  

Let us know, if you want your research project and/or tutorial to be included in this list!

# Dependencies
Support for additional optimizers or quantum backends can be activated by intalling them in your environment.
Tequila will then detect them automatically.
Currently those are: [Phoenics](https://github.com/aspuru-guzik-group/phoenics)
 and [GPyOpt](https://sheffieldml.github.io/GPyOpt/).
Quantum backends are treated in the same way.

## Quantum Backends
Currently supported
- [Qulacs](https://github.com/qulacs/qulacs) (recommended)
- [Qibo](https://github.com/Quantum-TII/qibo) -- currently needs to be qibo==0.1.1
- [Qiskit](https://github.com/qiskit/qiskit)  
- [Cirq](https://github.com/quantumlib/cirq)
- [PyQuil](https://github.com/rigetti/pyquil)
- [QLM](https://atos.net/en/solutions/quantum-learning-machine) (works also whith [myQLM](https://myqlm.github.io/index.html))

Tequila detects backends automatically if they are installed on your systems.
All of them are available over standard pip installation like for example `pip install qulacs`.
For best performance it is recommended to have `qulacs` installed.

## QuantumChemistry:
Currently supported
### [Psi4](https://github.com/psi4/psi4).
In a conda environment this can be installed with
```bash
conda install psi4 -c psi4
```
Here is a small [tutorial](https://nbviewer.org/github/tequilahub/tequila-tutorials/blob/main/chemistry/ChemistryModule.ipynb) that illustrates the usage.

### [Madness](https://github.com/kottmanj/madness)  
In a conda environment this can be installed with  
```bash
conda install madtequila -c kottmann
```  
This installs a modified version of madness ready to use with tequila.  
Alternatively it can be compiled from the sources provided in this [fork](https://github.com/kottmanj/madness) (follow readme instructions there).  
Here is a small [tutorial](https://nbviewer.org/github/tequilahub/tequila-tutorials/blob/main/chemistry/MadnessInterface.ipynb) that illustrates the usage. For fast performance it is recommended to not use the conda package.

### [PySCF](https://github.com/pyscf/pyscf)  
Install with
```bash
pip install pyscf
```  
Works similar as Psi4. Classical methods are also integrated in the madness interface allowing to use them in a basis-set-free representation.

# Documentation
You can build the documentation by navigating to `docs` and entering `make html`.
Open the documentation with a browser over like `firefox docs/build/html/index.html`
Note that you will need some additional python packages like `sphinx` and `mr2` that are not explicitly listed in the requirements.txt

You can also visit our prebuild online [documentation](https://tequilahub.github.io/tequila/)
that will correspond to the github master branch

# How to contribute
If you find any bugs or inconveniences in `tequila` please don't be shy and let us know.
You can do so either by raising an issue here on github or contact us directly.

If you already found a solution you can contribute to `tequila` over a pull-request.
Here is how that works:

1. Make a fork of `tequila` to your own github account.
2. Checkout the `devel` branch and make sure it is up to date with the main [github repository](https://github.com/aspuru-guzik-group/tequila).
3. Create and checkout a new branch from `devel` via `git branch pr-my-branch-name` followed by `git checkout pr-my-branch-name`. By typing `git branch` afterwards you can check which branch is currently checked out on your computer.
4. Introduce changes to the code and commit them with git.
5. Push the changes to *your* github account
6. Log into github and create a pull request to the main [github repository](https://github.com/aspuru-guzik-group/tequila). The pull-request should be directed to the `devel` branch (but we can also change that afterwards).

If you plan to introduce major changes to the base library it can be beneficial to contact us first.
This way we might be able to avoid conflicts before they arise.

If you used `tequila` for your research, feel free to include your algorithms here, either by integrating it into the core libraries or by demonstrating it with a notebook in the tutorials section. If you let us know about it, we will also add your research article in the list of research projects that use tequila (see above).

# Troubleshooting
If you experience trouble of any kind or if you either want to implement a new feature or want us to implement a new feature that you need:
Don't hesitate to contact us directly or raise an issue here on github.

## PySCF
If pyscf crashes on import with
```
Using default_file_mode other than 'r' is no longer supported. Pass the mode to h5py.File() instead
```
then you need to downgrade the h5py version
```
pip install --upgrade 'h5py <= 3.1' 
```
The issue will probably be fixed soon in pyscf.

## import errors on openfermion and cirq
This is fixed on master and devel but not yet on PyPi (v1.5.1)    
You can avoid it by downgrading cirq and openfermion  
```bash
pip install --upgrade "openfermion<=1.0.0"
pip install --upgrade "cirq<=0.9.1"
```  


## Qiskit backend
Qiskit version 0.25 is not yet supported.
`pip install --upgrade qiskit<0.25` fixes potential issues. If not: Please let us know.

## Circuit drawing
Standard graphical circuit representation within a Jupyter environment is often done using `tq.draw`.
Without further keywords `tequial` will try to create and compile a [qpic](https://github.com/qpic/qpic) file.
For proper display you will need the following dependencies: qpic, pdflatex and convert/ImageMagick (pre-installed on most GNU/Linux distributions, not pre-installed on macs).
On GNU/Linux distributions sometimes the permissions of `convert` to convert pdf to png are not granted, resulting in an error when trying to use `tq.draw`.
Click [here](https://stackoverflow.com/questions/52998331/imagemagick-security-policy-pdf-blocking-conversion?answertab=oldest#tab-top) for a possible solution.

In general, there is no reason to worry if `tq.draw` does not function properly.
It is just one way to display circuits, but not neccessary to have.
Alternatives are:
- Use `tq.draw(circuit, backend="qiskit")` (or `backend=cirq` )
- translate to qiskit/cirq and use their functionality ( `qiskit_circuit = tq.compile(circuit, backend='qiskit').circuit` )
- directly create pdfs: `tq.circuit.export_to(circuit, filename="my_name.pdf")` (will also create `my_name.qpic` that can be used with qpic)
- use `print(circuit)` (does not look pretty, but carries the same information).
- become a contributor and implement your own graphical circuit representation and create a pull-request.

## Qulacs simulator
You will need cmake to install the qulacs simulator
`pip install cmake`

You don't need `qulacs` for tequila to run (although is is recommended)
To install without `qulacs` just remove the `qulacs` line from `requirements.txt`
It can be replaced by one (or many) of the other supported simulators.
Note that simulators can also be installed on a later point, they don't need to be installed with `tequila`.
As long as they are installed within the same python environment `tequila` can detect them.

## Windows
You can in principle use tequila with windows as OS and have almost full functionality.
You will need to replace `Jax` with `autograd` for it to work.
In order to do so: Remove `jax` and `jaxlib` from `setup.py` and `requirements.txt` and add `autograd` instead.

In order to install qulacs you will need latest GNU compilers (at least gcc-7).
They can be installed for example over visual studio.

## Mac OS
Tequila runs on Mac OSX.
You might get in trouble with installing qulacs since it currently does not work with Apple's clang compiler.
You need to install latest GNU compile (at least gcc-7 and g++7) and set them as default before installing qulacs over pip.

## Qibo and GPyOpt
Currently you can't use Qibo and GPyOpt within the same environment.
