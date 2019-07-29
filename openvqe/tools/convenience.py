def binary_to_number(l):
    return int("".join(str(x) for x in l), 2)


def number_to_binary(number, bits=0):
    return [int(x) for x in list(bin(number)[2:].zfill(bits))]