"""
Export QCircuits as qpic files
https://github.com/qpic/qpic/blob/master/doc/qpic_doc.pdf
"""

from tequila.objective.objective import FixedVariable

import subprocess, numpy
from shutil import which

import numbers

system_has_qpic = which("qpic") is not None
system_has_pdflatex = which("pdflatex") is not None


def assign_name(parameter):
    if isinstance(parameter, tuple):
        return "\\theta"
    if hasattr(parameter, "extract_variables"):
        return repr(parameter.extract_variables()).lstrip('[').rstrip(']')
    if isinstance(parameter, FixedVariable):
        for i in [1, 2, 3, 4]:
            if numpy.isclose(numpy.abs(float(parameter)), numpy.pi / i, atol=1.e-4):
                if float(parameter) < 0.0:
                    return "-\\pi/{}".format(i)
                else:
                    return "+\\pi/{}".format(i)
        return "{:+2.4f}".format(float(parameter))

    try:
        return repr(parameter)
    except:
        return str(parameter)


def export_to_qpic(circuit, filename=None, filepath=None, always_use_generators=True,
                   decompose_control_generators=False,
                   group_together=False, qubit_names=None, mark_parametrized_gates=True, gatecolor1="tq",
                   textcolor1="white", gatecolor2="guo", textcolor2="black", *args, **kwargs) -> str:
    result = ""

    colors = [{"name": "tq", "rgb": (0.03137254901960784, 0.1607843137254902, 0.23921568627450981)}]
    colors += [{"name": "guo", "rgb": (0.988, 0.141, 0.757)}]

    # define colors as list of dictionaries with "name":str, "rgb":tuple entries
    if "colors" in kwargs:
        colors += kwargs["colors"]

    for color in colors:
        result += "COLOR {} {} {} {}\n".format(color["name"], *tuple(color["rgb"]))

    if group_together is True:
        group_together = "TOUCH"
    # define wires
    names = dict()
    if qubit_names is None:
        qubit_names = circuit.qubits
    if isinstance(qubit_names, str):
        qubit_names = [qubit_names for i in range(len(circuit.qubits))]
    for i, q in enumerate(circuit.qubits):
        name = "a" + str(q)
        names[q] = name
        result += name + " W " + str(qubit_names[i]) + "\n"

    for g in circuit.gates:
        gcol = gatecolor1
        tcol = textcolor1
        param = None
        if hasattr(g, "parameter"):
            if not isinstance(g.parameter, numbers.Number) and mark_parametrized_gates:
                tcol = textcolor2
                gcol = gatecolor2
            param = g.parameter

        if isinstance(param, numbers.Number):
            param = "{:1.2f}".format(param)

        # special gates
        # generator decomposition of H is misleading
        if g.name in ["H", "h"]:
            for target in g.target:
                result += " a{qubit} G:fill={gcol}  \\textcolor{tcol}{{{op}}} ".format(qubit=target, gcol=gcol,
                                                                                       tcol="{" + tcol + "}", op="H")
                if g.is_controlled():
                    for c in g.control:
                        result += names[c] + " "
        elif always_use_generators and g.make_generator(include_controls=decompose_control_generators) is not None:
            for ps in g.make_generator(include_controls=decompose_control_generators).paulistrings:
                if len(ps) == 0: continue

                # if controls are not decomposed this will become a mess
                # so we will represent NOT gates as + (and not as X)
                # and will use standard notation for Y and H
                if not decompose_control_generators and g.name.upper() in ["X", "Y", "Z", "H"]:
                    if g.name.upper() == "X":
                        result += " a{qubit} P:fill={gcol}  \\textcolor{tcol}{{{op}}} ".format(qubit=g.target[0],
                                                                                               gcol=gcol,
                                                                                               tcol="{" + tcol + "}",
                                                                                               op="+")
                    else:
                        result += " a{qubit} G:fill={gcol}  \\textcolor{tcol}{{{op}}} ".format(qubit=g.target[0],
                                                                                               gcol=gcol,
                                                                                               tcol="{" + tcol + "}",
                                                                                               op=g.name.upper())
                else:
                    for k, v in ps.items():
                        result += " a{qubit} P:fill={gcol}  \\textcolor{tcol}{{{op}}} ".format(qubit=k, gcol=gcol,
                                                                                               tcol="{" + tcol + "}",
                                                                                               op=v.upper())
                if g.is_controlled() and not decompose_control_generators:
                    for c in g.control:
                        result += names[c] + " "
                result += "\n"
            if hasattr(group_together, "upper"):
                for t in circuit.qubits:
                    result += "a{} ".format(t)
                result += "{}\n".format(group_together.upper())

        else:
            if g.name.upper() in ["Exp-Pauli".upper(), "GenRot".upper()]:
                # represent ExpPaulis as generators
                for ps in g.generator.paulistrings:
                    if len(ps) == 0: continue
                    for k, v in ps.items():
                        result += " a{qubit} P:fill={gcol}  \\textcolor{tcol}{{{op}}} ".format(qubit=k, gcol=gcol,
                                                                                               tcol="{" + tcol + "}",
                                                                                               op=v.upper())
                    if g.is_controlled():
                        for c in g.control:
                            result += names[c] + " "
                    result += "\n"
            else:
                for t in g.target:
                    result += names[t] + " G:fill={gcol} ".format(gcol=gcol)
                gname=g.name
                if "R" in gname.upper():
                    gname=gname.replace("R", "R_")

                if param is not None:
                    gname="{{{x}}}({angle})".format(x=gname, angle=assign_name(g.parameter))

                result += "\\textcolor{tcol}{{${op}$}} ".format(tcol="{" + tcol + "}", op=gname)
                if hasattr(g, "parameter") and g.parameter is not None:
                    result += "width=" + str(25 + 5 * len(assign_name(g.parameter))) + " "

                if g.is_controlled():
                    for c in g.control:
                        result += names[c] + " "

        result += "\n"

    if filename is not None:
        filenamex = filename
        if not filenamex.endswith(".qpic"):
            filenamex = filename + ".qpic"
        if filepath is not None:
            filenamex = "{}/{}".format(filepath, filenamex)
        with open(filenamex, "w") as file:
            file.write(result)
    return result


def export_to(circuit,
              filename: str,
              style="tequila",
              qubit_names: list = None, *args, **kwargs):
    """
    Parameters
    ----------
    circuit:
        the tequila circuit to export
    filename:
        filename.filetype, e.g. my_circuit.pdf, my_circuit.png (everything that qpic supports)
    style:
        string keyword (tequila, standard, generators) or dictionary containing the following keys:
        always_use_generators: represent all gates with their generators
        decompose_control_generators: Decompose the controls to generators. Effective only in combination with always_use_generators=True.
        group_together: Keep PauliStrings from the same generator together. Effective only in combination with always_use_generators=True.
        possible values: False, True, 'TOUCH' and 'BARRIER'. True is the same as TOUCH.
        BARRIER will create a visible barrier in qpic
    args
    kwargs

    Returns
    -------

    """
    if not system_has_qpic:
        raise Exception("You need qpic in order to export circuits to pictures ---\n pip install qpic")
    if "." not in filename:
        raise Exception("export_to: No filetype given {}, expected something like {}.pdf".format(filename, filename))

    if style is None or style == "tequila":
        style = {
            'decompose_control_generators': False,
            'always_use_generators': True,
            'group_together': False,
            'textcolor1': "white",
            "textcolor2": "black",
            "gatecolor1": "tq",
            "gatecolor2": "guo"
        }
    elif style == "standard":
        style = {
            'decompose_control_generators': False,
            'always_use_generators': False,
            'group_together': False
        }
    elif style == "plain":
        # standard without colors
        style = {
            'decompose_control_generators': False,
            'always_use_generators': False,
            'group_together': False,
            'textcolor1': "black",
            "textcolor2": "black",
            "gatecolor1": "white",
            "gatecolor2": "white"
        }
    elif style == "generators":
        style = {
            'decompose_control_generators': True,
            'always_use_generators': True,
            'group_together': "BARRIER"
        }
    elif not hasattr("style", "items"):
        raise Exception(
            "style needs to be `tequila`, or `standard` or `generators` or a dictionary, you gave: {}".format(
                str(style)))

    pop = []
    for k,v in kwargs.items():
        if k in style:
            style[k]=kwargs[k]
            pop.append(k)
    for k in pop:
        kwargs.pop(k)

    filename_tmp = filename.split(".")
    if len(filename_tmp) == 1:
        ftype = ".pdf"
        fname = filename
    else:
        ftype = filename_tmp[-1]
        fname = "".join(filename_tmp[:-1])

    fpath = None
    tmp = fname.split("/")
    fname = tmp[-1]
    if len(tmp) > 1:
        fpath = "".join([x + "/" for x in tmp[:-1]])
        if filename[0] == "/":
            fpath = "/" + fpath

    export_to_qpic(circuit=circuit,
                   filename=fname,
                   filepath=fpath,
                   qubit_names=qubit_names, **style, **kwargs)
    if ftype != "qpic":
        subprocess.call(["qpic", "{}.qpic".format(fname), "-f", ftype], cwd=fpath)
