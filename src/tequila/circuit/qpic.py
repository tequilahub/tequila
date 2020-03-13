"""
Export QCircuits as qpic files
https://github.com/qpic/qpic/blob/master/doc/qpic_doc.pdf
"""

from tequila.circuit import QCircuit
from tequila.circuit import gates
from tequila.tools import number_to_string

import subprocess
from shutil import which, move
from os import remove

system_has_qpic = which("qpic") is not None
system_has_pdflatex = which("pdflatex") is not None


def assign_name(parameter):
    if isinstance(parameter, tuple):
        return "\\theta"
    if hasattr(parameter, "extract_variables"):
        return str(parameter.extract_variables()).lstrip('[').rstrip(']')
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
                    25 + 5*len(assign_name(g.parameter))) + " "
            elif hasattr(g, "parameter") and g.parameter is not None:
                result += " G $" + g.name + "(" + g.parameter.name + ")$ width=" + str(
                    25 + 5*len(assign_name(g.parameter))) + " "
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
                    25 + 5*len(assign_name(g.parameter))) + " "
            elif hasattr(g, "parameter") and g.parameter is not None:
                result += " G $" + g.name + "(" + assign_name(g.parameter) + ")$ width=" + str(
                    25 + 5*len(assign_name(g.parameter))) + " "
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


def export_to_pdf(circuit: QCircuit, filename, keep_tex=True, keep_qpic=True):
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

    with open(filename + ".tex", "w") as file:
        subprocess.call(["qpic", str(filename_qpic)], stdout=file)

    latex_header = """
\\documentclass[]{standalone}
\\usepackage[english]{babel}
\\usepackage{amsmath}
\\usepackage{amssymb}
\\usepackage{graphicx}
\\usepackage{xspace}
\\usepackage{booktabs}
\\usepackage{xcolor}
\\usepackage{tikz}
\\usepackage{qcircuit}
"""

    with open(filename + ".tmp.tex", "w") as file:
        file.write(latex_header + "\n")
        file.write("\\begin{document}\n")
        file.write("\\input{" + filename + "}\n")
        file.write("\\end{document}\n")

    with open(filename + "tmp.log", "w") as file:
        subprocess.call(["pdflatex", str(filename) + ".tmp.tex"], stdout=file)

    move(filename + ".tmp.pdf", filename + ".pdf")
    remove(filename + ".tmp.aux")
    remove(filename + ".tmp.log")
    remove(filename + ".tmp.tex")
    if not keep_qpic:
        remove(filename + ".qpic")
    if not keep_tex:
        remove(filename + ".tex")


if __name__ == "__main__":
    circuit = gates.X(0) + gates.H(1) + gates.X(target=0, control=1) + gates.Ry(target=0, control=1,
                                                                                angle="\\theta") + gates.X(target=2,
                                                                                                           control=[0,
                                                                                                                    1])
    string = export_to_qpic(circuit)

    print(string)

    export_to_pdf(circuit, filename="test", keep_qpic=True, keep_tex=True)
