import openfermion

import tequila


def fermion_to_majorana(ops: openfermion.FermionOperator):
    """
    We use openfermion to get Majorana operators based on fermion operators
    """
    return openfermion.get_majorana_operator(ops)


def majo_to_graph(majorana: openfermion.MajoranaOperator):
    """

    """
    one_dim_chain = ""
    indices_pairs = []
    for term in majorana.terms:
        if majorana.terms[term] != 0 and term != ():
            indices_pairs.append(term)

    for pair in indices_pairs:
        if pair[0] % 2 == 0 and pair[1] == pair[0] + 1:
            one_dim_chain += "B{site} ".format(site=pair[0] // 2)
        elif pair[0] % 2 == 0 and pair[1] % 2 == 0:
            one_dim_chain += "A{site1}{site2} ".format(site1=pair[0] // 2,
                                                       site2=pair[1] // 2)
        elif pair[0] % 2 == 0 and pair[1] % 2 == 1:
            one_dim_chain += "A{site1}{site2}*B{site2} ".format(
                site1=pair[0] // 2, site2=(pair[1] - 1) // 2)
        elif pair[0] % 2 == 1 and pair[1] % 2 == 0:
            one_dim_chain += "B{site1}*A{site1}{site2} ".format(
                site1=(pair[0] - 1) // 2, site2=pair[1] // 2)
    return one_dim_chain


def op_to_pauli(op: str):
    """
    op is an Edge or Vertex operator
    examples are 'A12', 'B2'
    """
    if op[0] == "A":
        if op[1] < op[2]:
            return tequila.QubitHamiltonian(openfermion.QubitOperator(
                "Y{site1} X{site2}".format(site1=op[1], site2=op[2])))
        else:
            return tequila.QubitHamiltonian(openfermion.QubitOperator(
                "Y{site1} X{site2}".format(
                    site1=op[1], site2=op[2]), coefficient=-1))
    elif op[0] == "B":
        return tequila.QubitHamiltonian(openfermion.QubitOperator(
            "X{site} Y{site}".format(site=op[1]), coefficient=1j))


def graph_to_pauli(graph: str):
    """
    graph is a string representation of all the edge and vertex operators
    Example is graph = 'B1*A12 A12*B2'
    """
    op_list = graph.split()
    paulis = tequila.QubitHamiltonian()
    for graph_op in op_list:
        smaller_list = graph_op.split("*")
        tempo_pauli = tequila.QubitHamiltonian()
        for op in smaller_list:
            tempo_pauli *= op_to_pauli(op)

        paulis += tempo_pauli
    return paulis

