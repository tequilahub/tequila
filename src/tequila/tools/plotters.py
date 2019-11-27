"""
Convenience plot functions
"""
from tequila import QubitWaveFunction
from tequila.simulators import SimulatorReturnType
import typing


def plot_counts(counts: typing.Union[dict, QubitWaveFunction, SimulatorReturnType], filename: str = None,
                title: str = None, label_with_integers=False):
    from matplotlib import pyplot as plt

    data = counts
    if isinstance(counts, SimulatorReturnType):
        data = counts.counts

    if title is not None:
        plt.title(title)

    plt.ylabel("counts")
    plt.xlabel("state")

    values = []
    names = []
    for k, v in data.items():
        values.append(v)
        if label_with_integers:
            names.append(str(k.integer))
        else:
            names.append(str(k.binary))
    plt.bar(names, values)

    if filename is None:
        plt.show()
    else:
        with open(filename+"_data", 'a+') as file:
            file.write("names\tvalues\n")
            for i, v in enumerate(values):
                file.write(str(names[i])+"\t"+str(values[i])+"\n")
        plt.savefig(filename)
