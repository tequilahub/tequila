# OpenVQE
Code for OpenVQE: a Python framework for extending/improving VQE

- Look into Keras, TF and how they modularized 

# General Structure:
As far as I see it would make sense to subdivide into the following modules 
- state preparation
- hamiltonian (getting expectation values)
- optimization
- simulator
- postprocessing
- data storage
- data analysis (plotter)

(Should we have something like `Model` in Keras but like `VQE_Model` that takes in all the input/setting and runs the algorithm?)

Global Parameter Class or every module with own parameters (which might be inherited from common base or each other)?
python dataclass structure is convenient

# State Preparations
- Classic UCC parametrization (exponential of Paulis)
- Easy to make custom made circuits
- Enforcing Symmetry while reducing parameters (https://arxiv.org/pdf/1904.10910.pdf). 
  At least the Particle-Number part looks reasonable. Spin part looks expensive. 
- Static vs. dynamic ansatz (e.g. static: UCCSD, hardware-efficient; dynamic: ADAPT-VQE, ROTOSELECT)

# Hamiltonian/Objective Function
- QC interfaces (psi4, pyscf) should be sufficient. 
  Experienced difficulties with openfermionpsi4/pyscf regarding flexibility.
  So we might need our own interfaces
- Maybe include easy to use interfaces for common models: Hubbard, Heisenberg etc
- Make it flexible (easy to define custom Hamiltonians)
- General objective functions?

# Optimization
- Look into Keras, TF and how they modularized 
- Analytical gradients
- Optimization strategy: layer-wise training, effective identity, ROTOSELECT

# Simulator
- Testing Intel-QS right now for other project. Let's see how it performs
- Otherwise I would include most of the default easy to use python-supported libraries
  Cirq, Forest/qvm, Pennylane (supports automatic differentiation)
  
# Postprocessing
- Marginals (https://arxiv.org/abs/1801.03524)
- Error mitigation techniques (IBM's extrapolation)?
- Excited states?

# Data storage
- HDF5, like OpenFermion's `MolecularData`

# Data analysis
- Basic plotting capabilities (e.g. energy and energy error vs. bond length)
