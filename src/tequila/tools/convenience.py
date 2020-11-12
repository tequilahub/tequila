from numpy import isclose, pi
from cmath import polar
import numbers


def list_assignment(o):
    """
    --> moved to tools
    Helper function to make initialization with lists and single elements possible
    :param o: iterable object or single element
    :return: Gives back a list if a single element was given
    """
    if o is None:
        return []
    elif isinstance(o,tuple):
        return o
    elif hasattr(o, "__get_item__"):
        return list(o)
    elif hasattr(o, "__iter__"):
        return list(o)
    else:
        return [o]


def number_to_string(number: complex, precision: int = 4, threshold: float = 1.e-6) -> str:
    if not isinstance(number, numbers.Number):
        return str(number)

    number = complex(number)
    real = number.real
    imag = number.imag
    prec = '{:+.' + str(precision) + 'f}'

    if isclose(real, 0.0, atol=threshold):
        return prec.format(imag) + "i"
    elif isclose(imag, 0.0, atol=threshold):
        return prec.format(real)
    else:
        r, theta = polar(number)
        return prec.format(r) + ('e^(' + prec).format(theta / pi) + 'πi)'


if __name__ == "__main__":
    from numpy import sqrt

    for v in [0, 1, 1.2345, 1 + 1j, 1j, (1 + 1j) / sqrt(2)]:
        print(number_to_string(number=v), " --- ", v)
