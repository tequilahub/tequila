from numpy import isclose, pi
from math import hypot, atan2
import numbers

def list_assignement(o):
    """
    --> moved to tools
    Helper function to make initialization with lists and single elements possible
    :param o: iterable object or single element
    :return: Gives back a list if a single element was given
    """
    if o is None:
        return None
    elif hasattr(o, "__get_item__"):
        return o
    elif hasattr(o, "__iter__"):
        return o
    else:
        return [o]

def number_to_string(number: complex, precision: int=4, threshold: float = 1.e-6) -> str:
    if not isinstance(number, numbers.Number):
        return str(number)

    number = complex(number)
    real = number.real
    imag = number.imag
    prec = '{:.'+str(precision)+'f}'

    if isclose(real, 0.0, atol=threshold):
        return prec.format(imag)+"i" if imag < 0 else ('+'+prec).format(imag)+"i"
    elif isclose(imag, 0.0, atol=threshold):
        return prec.format(real) if real < 0 else ('+'+prec).format(real)
    else:
        r = hypot(real, imag)
        theta = atan2(real, imag)
        if isclose(theta, 0.5, atol=threshold):
            return "+" + prec.format(r) + "i"
        elif isclose(theta, -0.5, atol=threshold):
            return "-" + prec.format(r) + "i"
        elif isclose(theta, 1.0, atol=threshold):
            return "-" + prec.format(r) + "i"
        elif isclose(theta, -1.0, atol=threshold):
            return "-" + prec.format(r) + "i"
        else:
            return "+"+prec.format(r) + ('e^('+prec).format(theta/pi) + 'πi)'


if __name__ == "__main__":
    from numpy import sqrt
    for v in [0, 1, 1.2345, 1 + 1j, 1j, (1 + 1j) / sqrt(2)]:
        print(number_to_string(number=v), " --- ", v)
