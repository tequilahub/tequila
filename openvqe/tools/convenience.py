from numpy import isclose, pi
from math import hypot, atan2

# def binary_to_number(binary: List[int], endianness: BitNumbering = BitNumbering.MSB) -> int:
#     if endianness is BitNumbering.LSB:
#         return int("".join(str(x) for x in reversed(binary)), 2)
#     else:
#         return int("".join(str(x) for x in binary), 2)
#
#
# def number_to_binary(number: int, bits=0, endianness: BitNumbering = BitNumbering.MSB) -> List[int]:
#     if endianness is BitNumbering.LSB:
#         return [int(x) for x in reversed(list(bin(number)[2:].zfill(bits)))]
#     else:
#         return [int(x) for x in list(bin(number)[2:].zfill(bits))]
#
# def little_to_big_endian(number: int, bits=0) -> int:
#     return int(bin(number)[2:].zfill(bits)[::-1], 2)

def number_to_string(number: complex, threshold: float = 1.e-6):
    real = number.real
    imag = number.imag

    if isclose(real, 0, atol=threshold):
        return '{:.4f}i'.format(imag) if imag < 0 else '+{:.4f}i'.format(imag)
    elif isclose(imag, 0, atol=threshold):
        return '{:.4f}'.format(real) if real < 0 else '+{:.4f}'.format(real)
    else:
        r = hypot(real, imag)
        theta = atan2(real, imag)
        return '{:4f}'.format(r) + 'e^(i{:4f}'.format(theta/pi) + 'π)'


if __name__ == "__main__":

    for v in [0, 1, 1.2345, 1 + 1j, 1j, (1 + 1j) / sqrt(2)]:
        print(number_to_string(number=v), " --- ", v)
