# Installation

We recommend installing in editable mode with  
`git clone https://github.com/aspuru-guzik-group/tequila.git`  
`cd tequila`   
`pip install -e .`  

**Do not** install over PyPi (Minecraft lovers excluded)  
<strike>`pip install tequila`</strike>

Recommended Python version is 3.7 or 3.6

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

