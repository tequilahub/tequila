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
    parameters2 = Parameters()
    print("parameters2\n", parameters2)

    parameters.name="ASD"
    # access/change parameters like this
    maxiter = parameters.optimizer.maxiter
    parameters.hamiltonian.qc_data.basis_set='sto-3g'
    parameters.hamiltonian.qc_data.name = 'asd'

    print("parameters1\n", parameters)
    print("parameters2\n", parameters2)



    #print(type(parameters.hamiltonian.supported.qc))