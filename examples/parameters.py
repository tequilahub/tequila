"""
Example how to use parameters
"""
from openvqe.parameters.parameters import Parameters

if __name__=="__main__":

    print("Use OpenVQE parameters:")

    # initialize manually:
    parameters = Parameters()
    print("OpenVQE parameters (default values)", parameters)



    # if hamiltonian type is not QC then qc_data will not be initiallized
    parameters2 = Parameters(hamiltonian=Parameters.Hamiltonian(type="CUSTOM"))
    print("parameters2\n", parameters2)

    parameters.name="ASD"
    parameters.preparation.decomposition='TROTTER'
    parameters.hamiltonian.name="HAMILTON"
    parameters.optimizer.type='COBYLA'
    # access/change parameters like this
    maxiter = parameters.optimizer.maxiter
    parameters.qc.basis_set='sto-3g'

    print("parameters1\n", parameters)
    print("parameters2\n", parameters2)



    #print(type(parameters.hamiltonian.supported.qc))