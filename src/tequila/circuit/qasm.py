"""
Export QCircuits as qasm code

OPENQASM version 2.0 specification from:
A. W. Cross, L. S. Bishop, J. A. Smolin, and J. M. Gambetta, e-print arXiv:1707.03429v2 [quant-ph] (2017).
https://arxiv.org/pdf/1707.03429v2.pdf
"""
from tequila import TequilaException
from tequila.circuit import QCircuit
from tequila.circuit.compiler import CircuitCompiler
import tequila.circuit.gates as gates
from numpy import pi
from typing import Dict
import typing


def export_open_qasm(circuit: QCircuit, variables=None, version: str = "2.0", filename: str = None, zx_calculus: bool = False) -> str:
    """
    Allow export to different versions of OpenQASM

    Args:
        circuit: to be exported to OpenQASM
        variables: optional dictionary with values for variables
        version: of the OpenQASM specification, optional
        filename: optional file name to save the generated OpenQASM code
        zx_calculus: indicate if y-gates must be transformed to xz equivalents

    Returns:
        str: OpenQASM string
    """

    if version == "2.0":
        result = convert_to_open_qasm_2(circuit=circuit, variables=variables, zx_calculus=zx_calculus)
    else:
        return "Unsupported OpenQASM version : " + version
    # TODO: export to version 3

    if filename is not None:
        with open(filename, "w") as file:
            file.write(result)
            file.close()

    return result


def import_open_qasm(qasm_code: str, version: str = "2.0", rigorous: bool = True) -> QCircuit:
    """
    Allow import from different versions of OpenQASM

    Args:
        qasm_code: string with the OpenQASM code
        version: of the OpenQASM specification, optional
        rigorous: indicates whether the QASM code should be read rigorously

    Returns:
        QCircuit: equivalent to the OpenQASM code received
    """

    if version == "2.0":
        result = parse_from_open_qasm_2(qasm_code=qasm_code, rigorous=rigorous)
    else:
        return "Unsupported OpenQASM version : " + version
    # TODO: export to version 3

    return result


def import_open_qasm_from_file(filename: str, version: str = "2.0", rigorous: bool = True) -> QCircuit:
    """
    Allow import from different versions of OpenQASM from a file

    Args:
        filename: string with the file name with the OpenQASM code
        variables: optional dictionary with values for variables
        version: of the OpenQASM specification, optional
        rigorous: indicates whether the QASM code should be read rigorously

    Returns:
        QCircuit: equivalent to the OpenQASM code received
    """

    with open(filename, "r") as file:
        qasm_code = file.read()
        file.close()

    return import_open_qasm(qasm_code, version=version, rigorous=rigorous)


def convert_to_open_qasm_2(circuit: QCircuit, variables=None, zx_calculus: bool = False) -> str:
    """
    Allow export to OpenQASM version 2.0

    Args:
        circuit: to be exported to OpenQASM
        variables: optional dictionary with values for variables
        zx_calculus: indicate if y-gates must be transformed to xz equivalents

    Returns:
        str: OpenQASM string
    """

    if variables is None and not (len(circuit.extract_variables()) == 0):
        raise TequilaException(
            "You called export_open_qasm for a parametrized type but forgot to pass down the variables: {}".format(
                circuit.extract_variables()))

    compiler = CircuitCompiler(multitarget=True,
                               multicontrol=False,
                               trotterized=True,
                               generalized_rotation=True,
                               exponential_pauli=True,
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
                               gradient_mode=False,
                               ry_gate=zx_calculus,
                               y_gate=zx_calculus,
                               ch_gate=zx_calculus)

    compiled = compiler(circuit, variables=None)

    result = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n"

    qubits_names: Dict[int, str] = {}
    for q in compiled.qubits:
        name = "q[" + str(q) + "]"
        qubits_names[q] = name

    result += "qreg q[" + str(compiled.n_qubits) + "];\n"
    result += "creg c[" + str(compiled.n_qubits) + "];\n"

    for g in compiled.gates:

        control_str = ''
        if g.is_controlled():

            if len(g.control) > 2:
                raise TequilaException(
                    "Multi-controls beyond 2 not yet supported for OpenQASM 2.0. Gate was:\n{}".format(g))

            controls = list(map(lambda c: qubits_names[c], g.control))
            control_str = ','.join(controls) + ','

        gate_name = name_and_params(g, variables)
        for t in g.target:
            result += gate_name
            result += control_str
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

    for c in range(len(g.control)):
        res += "c"

    res += g.name.lower()

    if hasattr(g, "parameter") and g.parameter is not None:
        res += "(" + str(g.parameter(variables)) + ")"

    res += " "

    return res


def parse_from_open_qasm_2(qasm_code: str, rigorous: bool = True) -> QCircuit:

    lines = qasm_code.splitlines()
    clean_code = []
    # ignore comments
    for line in lines:
        if line.find("//") != -1:
            clean_line = line[0:line.find("//")].strip()
        else:
            clean_line = line.strip()
        if clean_line:
            clean_code.append(clean_line)

    if clean_code[0].startswith("OPENQASM"):
        clean_code.pop(0)
    elif rigorous:
        raise TequilaException("File must start with the 'OPENQASM' directive")

    if clean_code[0].startswith('include "qelib1.inc";'):
        clean_code.pop(0)
    elif rigorous:
        raise TequilaException("File must import standard library (qelib1.inc)")

    code_circuit = "\n".join(clean_code)
    # separate the custom command definitions from the normal commands
    custom_gates_map: Dict[str, QCircuit] = {}
    while True:
        i = code_circuit.find("gate ")
        if i == -1:
            break
        j = code_circuit.find("}", i)
        custom_name, custom_circuit = parse_custom_gate(code_circuit[i:j + 1], custom_gates_map=custom_gates_map)
        custom_gates_map[custom_name] = custom_circuit
        code_circuit = code_circuit[:i] + code_circuit[j + 1:]

    # parse regular commands
    commands = [s.strip() for s in code_circuit.split(";") if s.strip()]
    qregisters: Dict[str, int] = {}

    circuit = QCircuit()
    for c in commands:
        partial_circuit = parse_command(command=c, custom_gates_map=custom_gates_map, qregisters=qregisters)
        if partial_circuit is not None:
            circuit += partial_circuit

    return circuit


def parse_custom_gate(gate_custom: str, custom_gates_map: Dict[str, QCircuit]) -> (str, QCircuit):
    """
    Parse custom gates code

    Args:
        gate_custom: code with custom gates
    """
    gate_custom = gate_custom[5:]
    spec, body = gate_custom.split("{", 1)

    if "(" in spec:
        i = spec.find("(")
        j = spec.find(")")
        if spec[i + 1:j].strip():
            raise TequilaException("Parameters for custom gates not supported: {}".format(spec))
        spec = spec[:i] + spec[j + 1:]

    spec = spec.strip()

    if " " in spec:
        name, qargs = spec.split(" ", 1)
        name = name.strip()
        qargs = qargs.strip()
    else:
        raise TequilaException("Custom gate specification doesn't have any arguments: {}".format(spec))

    custom_qregisters: Dict[str, int] = {}
    for qarg in qargs.split(','):
        custom_qregisters[qarg] = len(custom_qregisters)

    body = body[:-1].strip()
    commands = [s.strip() for s in body.split(";") if s.strip()]

    custom_circuit = QCircuit()
    for c in commands:
        partial_circuit = parse_command(command=c, custom_gates_map=custom_gates_map, qregisters=custom_qregisters)
        if partial_circuit is not None:
            custom_circuit += partial_circuit

    return name, custom_circuit


def parse_command(command: str, custom_gates_map: Dict[str, QCircuit], qregisters: Dict[str, int]) -> QCircuit:
    """
    Parse qasm code command

    Args:
        command: open qasm code to be parsed
        custom_gates_map: map with custom gates
    """

    name, rest = command.split(" ", 1)

    if name in ("barrier", "creg", "measure", "id"):
        return None
    if name in ("opaque", "if"):
        raise TequilaException("Unsupported operation {}".format(command))

    args = [s.strip() for s in rest.split(",") if s.strip()]

    if name == "qreg":
        regname, sizep = args[0].split("[", 1)
        size = int(sizep[:-1])
        for i in range(size):
            qregisters[regname + "[" + str(i) + "]"] = len(qregisters)
        return None

    for arg in args:
        if not (arg in qregisters or arg in [key.split("[",1)[0] for key in qregisters.keys()]):
            raise TequilaException("Invalid register {}".format(arg))

    if name in custom_gates_map:
        custom_circuit = custom_gates_map[name]
        qregisters_values = []
        for a in args:
            qregisters_values.append(get_qregister(a, qregisters))
        return apply_custom_gate(custom_circuit=custom_circuit, qregisters_values=qregisters_values)

    if name in ("x", "y", "z", "h", "cx", "cy", "cz", "ch"):
        return QCircuit.wrap_gate(gates.impl.QGateImpl(name=(name[1] if name[0] == 'c' else name).upper(),
                                            control=get_qregister(args[0], qregisters) if name[0] == 'c' else None,
                                            target=get_qregister(args[1 if name[0] == 'c' else 0], qregisters)))
    if name in ("ccx", "ccy", "ccz"):
        return QCircuit.wrap_gate(gates.impl.QGateImpl(name=name.upper()[2],
                                            control=[get_qregister(args[0], qregisters), get_qregister(args[1], qregisters)],
                                            target=get_qregister(args[2], qregisters)))
    if name.startswith("rx(") or name.startswith("ry(") or name.startswith("rz(") or \
        name.startswith("crx(") or name.startswith("cry(") or name.startswith("crz("):
        return QCircuit.wrap_gate(gates.impl.RotationGateImpl(axis=name[2 if name[0] == 'c' else 1],
                                                   angle=get_angle(name)[0],
                                                   control=get_qregister(args[0], qregisters) if name[0] == 'c' else None,
                                                   target=get_qregister(args[1 if name[0] == 'c' else 0], qregisters)))
    if name.startswith("U("):
        angles = get_angle(name)
        return gates.U(theta=angles[0], phi=angles[1], lambd=angles[2],
                 control=None,
                 target=get_qregister(args[0], qregisters))
    if name.startswith("u1("):
        angles = get_angle(name)
        return gates.u1(lambd=angles[0],
                  control=None,
                  target=get_qregister(args[0], qregisters))
    if name.startswith("u2("):
        angles = get_angle(name)
        return gates.u2(phi=angles[0], lambd=angles[1],
                  control=None,
                  target=get_qregister(args[0], qregisters))
    if name.startswith("u3("):
        angles = get_angle(name)
        return gates.u3(theta=angles[0], phi=angles[1], lambd=angles[2],
                  control=None,
                  target=get_qregister(args[0], qregisters))
    if name.startswith("cu1("):
        angles = get_angle(name)
        return gates.u1(lambd=angles[0],
                  control=get_qregister(args[0], qregisters),
                  target=get_qregister(args[1], qregisters))
    if name.startswith("cu2("):
        angles = get_angle(name)
        return gates.u2(phi=angles[0], lambd=angles[1],
                  control=get_qregister(args[0], qregisters),
                  target=get_qregister(args[1], qregisters))
    if name.startswith("cu3("):
        angles = get_angle(name)
        return gates.u3(theta=angles[0], phi=angles[1], lambd=angles[2],
                  control=get_qregister(args[0], qregisters),
                  target=get_qregister(args[1], qregisters))
    if name in ("s", "t", "sdg", "tdg"):
        g = gates.Phase(pi / (2 if name.startswith("s") else 4),
                     control=None,
                     target=get_qregister(args[0], qregisters))
        if name.find("dg") != -1:
            g = g.dagger()
        return g


def apply_custom_gate(custom_circuit: QCircuit, qregisters_values: list) -> QCircuit:
    applied_custom_circuit = QCircuit()
    for gate in custom_circuit.gates:
        g = gate.copy()
        g._target = tuple([qregisters_values[i] for i in gate._target])
        g._control = tuple([qregisters_values[i] for i in gate._control]) if gate.is_controlled() else gate._control
        applied_custom_circuit += g
    return applied_custom_circuit


def get_qregister(qreg: str, qregisters: Dict[str, int]) -> typing.Union[list, int]:
    if qreg == qreg.split("[", 1)[0]:
        qreg_tequila = [qregisters[key] for key in qregisters.keys() if qreg == key.split("[", 1)[0]]
    else:
        qreg_tequila = qregisters[qreg]
    return qreg_tequila

def get_angle(name: str) -> list:
    i = name.find('(')
    j = name.find(')')
    if j == -1:
        raise TequilaException("Invalid specification {}".format(name))
    angles_str = name[i+1:j].split(',')
    angles = []
    for angle in angles_str:
        try:
            phase = float(angle)
        except ValueError:
            if angle.find('pi') == -1:
                raise TequilaException("Invalid specification {}".format(name))
            angle = angle.replace('pi', '')
            try:
                sign = 1
                div = 1
                if angle.find('-') != -1:
                    angle = angle.replace('-', '')
                    sign = -1
                if angle.find('/') != -1:
                    div = float(angle[angle.index('/')+1:])
                    angle = angle[:angle.index('/')]
                if angle.find('*') != -1:
                    angle = angle.replace('*', '')
                    phase = sign * float(angle) * pi / div
                elif len(angle) == 0:
                    phase = sign * pi / div
                else:
                    phase = sign * float(angle) / div
            except ValueError:
                raise TequilaException("Invalid specification {}".format(name))
        angles.append(phase)
    return angles

