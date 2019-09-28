from numpy import isclose, pi
from math import hypot, atan2

def number_to_string(number: complex, precision: int=4, threshold: float = 1.e-6) -> str:
    number = complex(number)
    real = number.real
    imag = number.imag
    prec = '{:.'+str(precision)+'f}'

    if isclose(real, 0, atol=threshold):
        return prec.format(imag)+"i" if imag < 0 else ('+'+prec).format(imag)+"i"
    elif isclose(imag, 0, atol=threshold):
        return prec.format(real) if real < 0 else ('+'+prec).format(real)
    else:
        r = hypot(real, imag)
        theta = atan2(real, imag)
        return prec.format(r) + ('+e^(i'+prec).format(theta/pi) + 'π)'


if __name__ == "__main__":
    from numpy import sqrt
    for v in [0, 1, 1.2345, 1 + 1j, 1j, (1 + 1j) / sqrt(2)]:
        print(number_to_string(number=v), " --- ", v)
