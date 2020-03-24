from tequila import TequilaException
from tequila.hamiltonian import paulis
from tequila.objective.objective import Objective, ExpectationValueImpl, Variable, ExpectationValue
from tequila.circuit.circuit import QCircuit
from tequila.simulators.simulator_api import compile_objective
from tequila.circuit.gradient import __grad_expectationvalue

def get_generator(gate):
    if gate.name.lower() == 'rx':
        gen=paulis.X(gate.target[0])
    elif gate.name.lower() == 'ry':
        gen=paulis.Y(gate.target[0])
    elif gate.name.lower()  == 'rz':
        gen=paulis.Z(gate.target[0])
    elif gate.name.lower() == 'phase':
        gen=paulis.Qm(gate.target[0])
    else:
        print(gate.name.lower())
        raise TequilaException('cant get the generator of a non Gaussian gate, you fool!')
    return gen


def qng_metric_tensor_blocks(objective,initial_values,samples=None,backend=None,noise_model=None):
    if len(objective.args) is not 1:
        raise TequilaException('sorry, no generalized QNG yet')

    U=objective.args[0].U
    moments=U.canonical_moments
    sub=[QCircuit.from_moments(moments[:i]) for i in range(1,len(moments),2)]
    parametric_moms=[moments[i] for i in range(1,len(moments)+1,2)]
    generators =[]
    for pm in parametric_moms:
        set=[]
        if len(pm.gates) is not 0:
            for gate in pm.gates:
                gen=get_generator(gate)
                set.append(gen)
        if len(set) is not 0:
            generators.append(set)
        else:
            generators.append(None)
    blocks=[]
    for i,set in enumerate(generators):
        if set is None:
            pass
        else:
            block=[[0 for _ in range(len(set))] for _ in range(len(set))]
            for k,gen1 in enumerate(set):
                for q,gen2 in enumerate(set):
                    if k == q:
                        arg= (ExpectationValue(U=sub[i], H=gen1 * gen1) - ExpectationValue(U=sub[i],H=gen1)**2)/4
                    else:
                        arg = (ExpectationValue(U=sub[i], H=gen1 * gen2) - ExpectationValue(U=sub[i], H=gen1)*ExpectationValue(U=sub[i],H=gen2) ) / 4
                    block[k][q] = compile_objective(arg, variables=initial_values, samples=samples, backend=backend,noise_model=noise_model)
            blocks.append(block)
    return blocks
