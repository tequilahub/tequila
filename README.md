# OpenVQE
Code for OpenVQE: a Python framework for variational quantum eigensolvers

# Simulators
Currently supported
- Qiskit
- Cirq
- Pyquil

# Installation
Just run 'pip install . ' in the OpenVQE directory (thats the directory where setup.py is)
Since the package is currently under development we haven't spend much time on installers .... but maybe it works :-)
If you plan to develop the code we recommend to not install it anyway (see below)

# Usage without installation
1. go into the OpenVQE main directory (where setup.py and requirements.txt are)
2. type 'pip install -r requirements.txt' to install all necessary packages
3. if you don't want to install all simulators, feel free to comment them out in requirements.txt
4. Add the OpenVQE main directory to your PYTHONPATH so that your python can find it (for linux and max see below)

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
pyquil tests might fail if you don't have rigettis QVM installed (that is fine)
if you commented out simulators in the installation process above, then those tests will of course also fail

# Troubleshooting
If you experience trouble of any kind or if you either want to implement a new feature or want us to implement a new feature that you need
don't hesitate to contact one of the developers or write in the #open-vqe slack channel
