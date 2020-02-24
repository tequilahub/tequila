# Tequila
Code for Tequila: a Python framework for variational quantum algorithms

# Simulators
Currently supported
- Qulacs
- Qiskit
- Cirq
- Pyquil

# QuantumChemistry:
If you are intending to use psi4:
psi4 has no default pip installer, so it is not part of requirements.txt
if you are using conda you can install it with
conda install psi4 -c psi4
PSI4 WILL NOT RUN WITH PYTHON 3.7x

# Installation as user
Just run 'pip install . ' in the OpenVQE directory (thats the directory where setup.py is)
Feel free to comment out simulators in requirements.txt if you do not want to install them all (but you should at least install one of them)

# Installation as developer
same as above but run 'python -m pip install setyp.py -e .' 

# Usage without installation
1. go into the OpenVQE main directory (where setup.py and requirements.txt are)
2. type 'pip install -r requirements.txt' to install all necessary packages
3. if you don't want to install all simulators, feel free to comment them out in requirements.txt (install at least cirq)
4. Add the OpenVQE main directory to your PYTHONPATH so that your python can find it (for linux and mac see below)

# Add OpenVQE to your PYTHONPATH (Linux and Mac)
This is only necessary if you have NOT installed OpenVQE
just type 'export PYTHONPATH=${PYTHONPATH}:/path/where/you/have/the/code/OpenVQE/
To avoid missunderstandings:
ls /path/where/you/have/the/code/OpenVQE/
contains: setup.py, requirements.txt, examples/, openvqe/, tests/

# Add OpenVQE to your PYTHONPATH (Windows)
I have no Idea. Alba, Chengran and Mario managed to achieve it, so they might be able to tell you :-)

# Test if everything works
go to OpenVQE/tests
type 'pytest'
the UnaryStatePrep test might fail. If so start it new untill it finishes

# Troubleshooting
If you experience trouble of any kind or if you either want to implement a new feature or want us to implement a new feature that you need
don't hesitate to contact one of the developers or write in the #open-vqe slack channel
