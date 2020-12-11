"""
Export QCircuits as qpic files
https://github.com/qpic/qpic/blob/master/doc/qpic_doc.pdf
"""

from tequila.circuit import QCircuit
from tequila.circuit import gates
from tequila.tools import number_to_string
from tequila.objective.objective import FixedVariable

import subprocess, numpy
from shutil import which, move
from os import remove

system_has_qpic = which("qpic") is not None
system_has_pdflatex = which("pdflatex") is not None


def assign_name(parameter):
    if isinstance(parameter, tuple):
        return "\\theta"
    if hasattr(parameter, "extract_variables"):
        return str(parameter.extract_variables()).lstrip('[').rstrip(']')
    if isinstance(parameter, FixedVariable):
        for i in [1,2,3,4]:
            if numpy.isclose(numpy.abs(float(parameter)), numpy.pi/i, atol=1.e-4):
                if float(parameter) < 0.0:
                    return "-\\pi/{}".format(i)
                else:
                    return "+\\pi/{}".format(i)
        return "{:+2.4f}".format(float(parameter))
    return str(parameter)


def export_to_qpic(circuit: QCircuit, filename=None) -> str:
    result = ""
    # define wires
    names = dict()
    for q in circuit.qubits:
        name = "a" + str(q)
        names[q] = name
        result += name + " W " + str(q) + "\n"

    for g in circuit.gates:
        if g.is_controlled():

            for t in g.target:
                result += names[t] + " "

            if hasattr(g, "angle"):
                result += " G $R_{" + g.axis_to_string[g.axis] + "}(" + assign_name(g.parameter) + ")$ width=" + str(
                    25 + 5 * len(assign_name(g.parameter))) + " "
            elif hasattr(g, "parameter") and g.parameter is not None:
                result += " G $" + g.name + "(" + g.parameter.name + ")$ width=" + str(
                    25 + 5 * len(assign_name(g.parameter))) + " "
            elif g.name.lower() == "x":
                result += " C "
            else:
                result += g.name + " "

            for c in g.control:
                result += names[c] + " "
        else:
            for t in g.target:
                result += names[t] + " "
            if hasattr(g, "angle"):
                result += " G $R_{" + g.axis_to_string[g.axis] + "}(" + assign_name(g.parameter) + ")$ width=" + str(
                    25 + 5 * len(assign_name(g.parameter))) + " "
            elif hasattr(g, "parameter") and g.parameter is not None:
                result += " G $" + g.name + "(" + assign_name(g.parameter) + ")$ width=" + str(
                    25 + 5 * len(assign_name(g.parameter))) + " "
            else:
                result += g.name + " "

        result += "\n"

    if filename is not None:
        filenamex = filename
        if not filenamex.endswith(".qpic"):
            filenamex = filename + ".qpic"
        with open(filenamex, "w") as file:
            file.write(result)
    return result

def export_to_pdf(circuit: QCircuit, filename):
    export_to(circuit=circuit, filename=filename, filetype="pdf")

def export_to(circuit: QCircuit, filename, filetype="pdf"):
    if not system_has_qpic:
        raise Exception("You need qpic in order to export circuits to pdfs ---\n pip install qpic")
    if not system_has_pdflatex:
        raise Exception("You need pdflatex in order to export circuits to pdfs")

    filename_qpic = None
    if isinstance(circuit, str):
        filename_qpic = circuit

    if filename_qpic is None:
        export_to_qpic(circuit, filename=filename)
        filename_qpic = filename + ".qpic"

    subprocess.call(["qpic", str(filename_qpic), "-f", "{}".format(filetype)])
    return


if __name__ == "__main__":
    circuit = gates.X(0) + gates.H(1) + gates.X(target=0, control=1) + gates.Ry(target=0, control=1,
                                                                                angle="\\theta") + gates.X(target=2,
                                                                                                           control=[0,
                                                                                                                    1])
    string = export_to_qpic(circuit)

    print(string)

    export_to_pdf(circuit, filename="test", keep_qpic=True, keep_tex=True)
