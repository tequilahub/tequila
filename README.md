# OpenVQE
Code for OpenVQE: framework for extending/improving VQE

- Look into Keras, TF and how they modularized 


@Hannah: Feel free to edit or tell me that some of those suggestions should not be included

# General Structure:
As far as I see it would make sense to subdivide into the following modules 
-state preparation
-hamiltonian (getting expectation values)
-optimization
-simulator
Global Parameter Class or every module with own parameters (which might be inherited from common base or each other)?
python dataclass structure is convenient

# State Preparations
- Classic UCC parametrization (exponential of Paulis)
- easy to make custom made circuits
- Enforcing Symmetry while reducing parameters
  https://arxiv.org/pdf/1904.10910.pdf
  At least the Particle-Number part looks reasonable
  Spin part looks expensive

# Hamiltonian
- QC interfaces (psi4, pyscf) should be sufficient
  Experienced difficulties with openfermionpsi4/pyscf regarding flexibility
  So we might need our own interfaces
- Maybe include easy to use interfaces for common models
  Hubbard, Heisenberg etc
- Make it flexible (easy to define custom Hamiltonians)

# Optimization
- Look into Keras, TF and how they modularized 

# Simulator
- Testing Intel-QS right now for other project
  lets see how it performs
- Otherwise I would include most of the default easy to use python-supported libraries
  Cirq, forest/qvm
