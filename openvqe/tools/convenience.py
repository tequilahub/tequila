def binary_to_number(l, invert=True):
    if invert:
        return int("".join(str(x) for x in reversed(l)), 2)
    else:
        return int("".join(str(x) for x in l), 2)


def number_to_binary(number, bits=0, invert=True):
    if invert:
        return ([int(x) for x in reversed(list(bin(number)[2:].zfill(bits)))])
    else:
        return ([int(x) for x in list(bin(number)[2:].zfill(bits))])
