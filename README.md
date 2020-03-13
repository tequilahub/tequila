# Tequila
Code for tequila: a Python framework for algorithm development on quantum computers

# Simulators
Currently supported
- Qulacs
- Qiskit
- Cirq
- Pyquil

# QuantumChemistry:
If you are intenting to use psi4:
psi4 has no default pip installer, so it is not part of requirements.txt
if you are using conda you can install it with
conda install psi4 -c psi4

# Installation
We currently recommend to install in development version
Here is the full procedure for that
1. cd into the tequila main directory (that is the directory where setup.py is)
2. (optional) check out the 'devel' branch for the most recent version with 'git checkout devel'
3. (optional) modify the dependencies.txt file to your needs (see also dependencies section in this readme)
4. install dependencies with 'pip install -r requirements.txt'
5. install tequila with 'pip install -e . ' (note the dot)
6. (optional) test tequila by going to tests and typing 'pytest'
(dependency tests might fail if you have not installed all packages, don't worry about that)

# Dependencies
All packages listed in requirements.txt are needed to work with tequila
There are some optional packages which you can comment in if you want them
like pyquil and qiskit
If you want to use phoenics or gpyopt optimizers you need to install the packages listed in requirements_phoenics.txt (and the same for gpyopt)
If you are using python 3.6 install the packages in requirements_phoenics_36.txt

# Troubleshooting
If you experience trouble of any kind or if you either want to implement a new feature or want us to implement a new feature that you need
don't hesitate to contact one of the developers or write in the #open-vqe slack channel
