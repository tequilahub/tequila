"""
Export QCircuits as qasm code

A. W. Cross, L. S. Bishop, J. A. Smolin, and J. M. Gambetta, e-print arXiv:1707.03429v2 [quant-ph] (2017).
https://arxiv.org/pdf/1707.03429v2.pdf
"""
from tequila import TequilaException
from tequila.circuit import QCircuit
from tequila.circuit.compiler import Compiler


def export_open_qasm(circuit: QCircuit, variables=None, version="2.0", filename=None) -> str:
    """
    Allow export to different versions of OpenQASM

    Args:
        circuit: to be exported to OpenQASM
        variables: optional dictionary with values for variables
        version: of the OpenQASM specification, optional
        filename: optional file name to save the generated OpenQASM code

    Returns:
        str: OpenQASM string
    """

    if version == "2.0":
        result = convert_to_open_qasm_2(circuit=circuit, variables=variables)
    else:
        return "Unsupported OpenQASM version : " + version
    # TODO: export to version 3

    if filename is not None:
        with open(filename, "w") as file:
            file.write(result)
            file.close()

    return result


def import_open_qasm(qasm_code: str, variables=None, version="2.0") -> QCircuit:
    """
    Allow import from different versions of OpenQASM

    Args:
        qasm_code: string with the OpenQASM code
        variables: optional dictionary with values for variables
        version: of the OpenQASM specification, optional

    Returns:
        QCircuit: equivalent to the OpenQASM code received
    """

    print("Import to Open QASM", qasm_code)
    # TODO: Implement import from open qasm
    return QCircuit()


def import_open_qasm_from_file(filename: str, variables=None, version="2.0") -> QCircuit:
    """
    Allow import from different versions of OpenQASM from a file

    Args:
        filename: string with the file name with the OpenQASM code
        variables: optional dictionary with values for variables
        version: of the OpenQASM specification, optional

    Returns:
        QCircuit: equivalent to the OpenQASM code received
    """

    with open(filename, "r") as file:
        qasm_code = file.read()
        file.close()

    return import_open_qasm(qasm_code, variables=variables, version=version)


def convert_to_open_qasm_2(circuit: QCircuit, variables=None) -> str:
    """
    Allow export to OpenQASM version 2.0

    Args:
        circuit: to be exported to OpenQASM
        variables: optional dictionary with values for variables

    Returns:
        str: OpenQASM string
    """

    if variables is None and not (len(circuit.extract_variables()) == 0):
        raise TequilaException(
            "You called export_open_qasm for a parametrized type but forgot to pass down the variables: {}".format(
                circuit.extract_variables()))

    compiler = Compiler(multitarget=True,
                        multicontrol=False,
                        trotterized=True,
                        generalized_rotation=True,
                        exponential_pauli=False,
                        controlled_exponential_pauli=True,
                        hadamard_power=True,
                        controlled_power=True,
                        power=True,
                        toffoli=True,
                        controlled_phase=True,
                        phase=True,
                        phase_to_z=True,
                        controlled_rotation=True,
                        swap=True,
                        cc_max=True,
                        gradient_mode=True)

    compiled = compiler(circuit, variables=variables)

    result = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n"

    qubits_names = dict()
    for q in compiled.qubits:
        name = "q[" + str(q) + "]"
        qubits_names[q] = name

    result += "qreg q[" + str(compiled.n_qubits) + "];\n"
    result += "creg c[" + str(compiled.n_qubits) + "];\n"

    for g in compiled.gates:

        if g.is_controlled() or g.name.lower() == "swap":

            result += name_and_params(g, variables)

            params = list(map(lambda c: qubits_names[c], g.control))
            params += (list(map(lambda t: qubits_names[t], g.target)))
            result += ','.join(params)

            result += ";\n"

        else:

            for t in g.target:
                result += name_and_params(g, variables)

                result += qubits_names[t]

                result += ";\n"

    return result


def name_and_params(g, variables):
    """
    Determines the quantum gate name and its parameters if applicable

    Args:
        g: gate to get its name
        variables: dictionary with values for variables

    Returns:
        str: name (and parameter) to the gate specified
    """

    res = ""

    if len(g.control) > 0:
        res += "c" + g.name.lower()
    else:
        res += g.name.lower()

    if hasattr(g, "parameter") and g.parameter is not None:
        res += "(" + str(g.parameter(variables)) + ")"

    res += " "

    return res


compiler_arguments_qasm = {
    "trotterized": True,
    "swap": True,
    "multitarget": True,
    "controlled_rotation": True,
    "gaussian": True,
    "exponential_pauli": True,
    "controlled_exponential_pauli": True,
    "phase": True,
    "power": True,
    "hadamard_power": True,
    "controlled_power": True,
    "controlled_phase": True,
    "toffoli": True,
    "phase_to_z": True,
    "cc_max": True
}
