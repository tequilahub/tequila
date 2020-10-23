Optimize Measurements by grouping into commuting cliques
========================================================

Quantum mechanics allows simultaneuous measurements of all mutually
commuting operators. VQE however permits only the measurements of a much
more restricted set of operators, the qubit-wise commuting operators
(Verteletskyi et al., “Measurement Optimization in the Variational
Quantum Eigensolver Using a Minimum Clique Cover" `J. Chem. Phys. 152,
124114 (2020) <https://doi.org/10.1063/1.5141458>`__). But through
certain unitary transformations any group of mutually commuting
operators can be tranformed into their qubit-wise commuting form (Yen et
al., "Measuring all compatible operators in one series of single-qubit
measurements using unitary transformations" `Chem. Theory Comput. 2020,
16, 4, 2400–2409 <https://doi.org/10.1021/acs.jctc.0c00008>`__).

How to Use:
-----------

This is how the technique can be used in tequila. Note that you will
only benefit from optimizing measurements when you are running on a real
quantum backend or when you simulate with finite samples. The difference
in timings is just an indicator that different processes happen, in this
specific example we have an Hamiltonian with 4 terms where optimization
of the measurements reduces those to 2 commuting groups (see the next
section for details)

.. code:: ipython3

    import tequila as tq
    import time
    tq.show_available_simulators()
    backend = None
    
    H = tq.paulis.Z([0,1]) + tq.paulis.Y([0,1]) + tq.paulis.X([0,1]) + tq.paulis.X(0) + tq.paulis.Z([0,1,2,3,4,5,6])
    
    U = tq.gates.ExpPauli(angle = "a", paulistring=tq.PauliString.from_string('X(0)Y(1)'))
    U += tq.gates.X(target=[0,1,2,3,4,5,6])
    
    E1 = tq.ExpectationValue(H=H, U=U) 
    E2 = tq.ExpectationValue(H=H, U=U, optimize_measurements = True)
    
    print(H)
    start = time.time()
    print(tq.simulate(E1, variables={"a":1.0}, samples=100000, backend=backend))
    print("time : {}s".format(time.time()-start))
    
    start = time.time()
    print(tq.simulate(E2, variables={"a":1.0}, samples=100000, backend=backend))
    print("time : {}s".format(time.time()-start))
    
    


Behind the Scenes
-----------------

.. code:: ipython3

    import tequila as tq
    import numpy as np
    from tequila.hamiltonian import QubitHamiltonian, paulis
    from tequila.grouping.binary_rep import BinaryHamiltonian

The following examples shows how to partition a given Hamiltonian into
commuting parts and how to find the unitary transformation needed to
transform the commuting terms into qubit-wise commuting form that is
easy to measure.

The Hamiltonian is simply

.. math::  H = \sigma_z(0)\sigma_z(1) + \sigma_y(0)\sigma_y(1) + \sigma_x(0)\sigma_x(1) + \sigma_x(0)

where :math:`\sigma_z(0)\sigma_z(1)`, :math:`\sigma_y(0)\sigma_y(1)`
does not commute with :math:`\sigma_x(0)`, so two separate measurements
are needed.

.. code:: ipython3

    H = paulis.Z(0) * paulis.Z(1) + paulis.Y(0) * paulis.Y(1) + \
        paulis.X(0) * paulis.X(1) + paulis.X(0) 


Here we use the binary representation of the Hamiltonian for
partitioning. The method commuting\_groups gets back a list of
BinaryHamiltonian whose terms are mutually commuting.

Call to\_qubit\_hamiltonian to visualize.

.. code:: ipython3

    binary_H = BinaryHamiltonian.init_from_qubit_hamiltonian(H)
    commuting_parts = binary_H.commuting_groups()

.. code:: ipython3

    print(len(commuting_parts)) # Number of measurements needed
    print(commuting_parts[0].to_qubit_hamiltonian())
    print(commuting_parts[1].to_qubit_hamiltonian())

The second group of terms :math:`H_2` are not currently qubit-wise
commuting and cannot be directly measured on current hardware. They
require further unitary transformation :math:`U` to become qubit-wise
commuting. The following code identifies two bases (list of
BinaryPauliString) that encodes the unitary transformation as

.. math::  U = \prod_i \frac{1}{\sqrt{2}} (\text{old_basis}[i] + \text{new_basis}[i])

 such that :math:`UH_2U` is qubit-wise commuting.

.. code:: ipython3

    qubit_wise_parts, old_basis, new_basis = commuting_parts[1].single_qubit_form()

.. code:: ipython3

    def display_basis(basis):
        for term in basis:
            print(QubitHamiltonian.from_paulistrings(term.to_pauli_strings()))
    print('Old Basis')
    display_basis(old_basis)
    print('\nNew Basis')
    display_basis(new_basis)

The transfromed term :math:`UH_2U` is qubit-wise commuting.

.. code:: ipython3

    print(qubit_wise_parts.to_qubit_hamiltonian())

Get the circuit for the unitary transformation to implement the
measurement scheme. The next function illustrates what is happening if
expectation values are formed with the ``optimize_measurements`` keyword
meaning it does the same as
``tq.ExpectationValue(H=H,U=U, optimize_measurements=True)``

.. code:: ipython3

    def optimize_measurements(H, U):
        binary_H = BinaryHamiltonian.init_from_qubit_hamiltonian(H)
        commuting_parts = binary_H.commuting_groups()
        result = tq.Objective()
        for cH in commuting_parts:
            qwc, Um = cH.get_qubit_wise()
            Etmp = tq.ExpectationValue(H=qwc, U=U+Um)
            result += Etmp
        
        return result
            

The new measurement scheme produces the same expectation values

.. code:: ipython3

    U = tq.gates.ExpPauli(angle = "a", paulistring=tq.PauliString.from_string('X(0)Y(1)'))
    E1 = tq.ExpectationValue(H=commuting_parts[1].to_qubit_hamiltonian(), U=U)
    E2 = optimize_measurements(H=commuting_parts[1].to_qubit_hamiltonian(), U=U)
    
    variables = {"a" : np.random.rand(1) * 2 * np.pi}
    print('Without measurement grouping')
    print(tq.simulate(E1, variables))
    print('\nWith measurement grouping')
    print(tq.simulate(E2, variables))

The get\_qubit\_wise methods always transforms the hamiltonian into the
qubit-wise commuting form with only z on each qubit. This is done via
extra single-qubit unitaries.

.. code:: ipython3

    print('The qubit-wise commuting hamiltonian, but does not have all z')
    print(commuting_parts[0].to_qubit_hamiltonian())
    
    qwc, U = commuting_parts[0].get_qubit_wise()
    print('\nThe same qubit-wise commuting hamiltonian with all z')
    print(qwc)
    print('\nThe corresponding single qubit unitaries')
    print(U)
