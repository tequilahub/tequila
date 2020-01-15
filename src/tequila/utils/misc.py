from numpy import isclose


def to_float(number) -> float:
    """
    Cast numeric type to reals
    """

    if hasattr(number, "imag"):
        if isclose(number.imag, 0.0):
            return float(number.real)
        else:
            raise Exception("imaginary part detected")
    else:
        try:
            return float(number)
        except TypeError:
            raise Exception("casting number=" + str(number) + " to float failed\n")
