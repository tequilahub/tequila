=========================
Tequila Library Reference
=========================

Tequila's functionalities are provided through the following modules.

.. currentmodule:: tequila.hamiltonian

Hamiltonian
===========

.. rubric:: Modules
.. autosummary::
   :toctree: hamiltonian
   :nosignatures:
   
   paulis


.. currentmodule:: tequila.circuit
    
Circuit
=======
.. rubric:: Modules
.. autosummary::
   :toctree: circuit
   :nosignatures:

   gates



.. currentmodule:: tequila.optimizers

Optimizers
===========

.. rubric:: Functions
.. autosummary::
   :toctree: optimizers
   :nosignatures:
   
   minimize
   show_available_optimizers

.. rubric:: Modules
.. autosummary::
   :toctree: optimizers
   :nosignatures:

   optimizer_scipy
   optimizer_phoenics

.. currentmodule:: tequila.simulators.simulator_api

Simulators API
==============
   
.. rubric:: Functions
.. autosummary::
   :toctree: simulators
   :nosignatures:

   simulate
   compile
   compile_circuit
   compile_objective
   compile_to_function
   draw
   pick_backend
   show_available_simulators

.. rubric:: Classes
.. autosummary::
   :toctree: simulators
   :nosignatures:

   BackendTypes
  

.. currentmodule:: tequila.quantumchemistry

Quantum Chemistry
=================

.. rubric:: Function
.. autosummary::
   :toctree: quantumchemistry
   :nosignatures:

   Molecule
   MoleculeFromOpenFermion

.. rubric:: Classes
.. autosummary::
   :toctree: quantumchemistry
   :nosignatures:
    
   QuantumChemistryPsi4
 
