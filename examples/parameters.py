"""
Example how to use parameters
"""
from openvqe.parameters import Parameters

if __name__=="__main__":

    print("Use OpenVQE parameters:")

    # initialize manually:
    parameters1 = Parameters()
    print("OpenVQE parameters (default values)\n", parameters1)

    #change/access parameters
    parameters1.comment= "Change the comment"
    parameters1.preparation.decomposition='TROTTER'
    parameters1.hamiltonian.name="HAMILTON"
    parameters1.optimizer.type='COBYLA'
    # access/change parameters like this
    maxiter = parameters1.optimizer.maxiter
    #parameters1.qc.basis_set='sto-3g'

    # read in xyz file
    #parameters1.qc.geometry='data/h2o.xyz'
    #print("read in geometry is:\n", parameters1.qc.get_geometry())


    # if hamiltonian type is not QC then qc_data will not be initiallized
    parameters2 = Parameters(hamiltonian=Parameters.Hamiltonian(type="CUSTOM"))

    print("parameters1\n", parameters1)
    print("parameters2\n", parameters2)

    # print parameters to file
    parameters1.print_to_file(filename='output', name='parameters1', write_mode='w')
    parameters2.print_to_file(filename='output', name='parameters2')

    # print out a template
    defaults = Parameters()
    defaults.print_to_file(filename='input_template', name='comment', write_mode='w')

    # read back in
    parameters1x = Parameters.read_from_file(filename='output', name='parameters1')
    parameters2x = Parameters.read_from_file(filename='output', name='parameters2')

    print("Does it work?")
    print(parameters1x == parameters1)
    print(parameters2x == parameters2)
    print(not parameters1x == parameters2)
    print(not parameters2x == parameters1)

    #print(type(parameters.hamiltonian.supported.qc))

    # convert to dictionary
    d = Parameters().__dict__
    print(d)