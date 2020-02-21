from tequila import BitString, BitNumbering, BitStringLSB


def test_bitstrings():
    for i in range(15):
        bit = BitString.from_int(integer=i)
        assert (bit.integer == i)
        assert (bit.binary == format(i, 'b'))
        assert (bit.binary == bin(i)[2:])

    arrays = [[0, 0, 1], [1, 0, 0], [1, 0, 1], [1, 0, 1, 1, 1, 0], [1, 0, 0, 1, 1, 1, 0, 0, 1, 0]]
    integers = [1, 4, 5, 32 + 8 + 4 + 2, 2 + 16 + 32 + 64 + 512]

    for i, arr in enumerate(arrays):
        nbits = len(arr)
        bita = BitString.from_array(array=arr, nbits=nbits)
        bitb = BitString.from_int(integer=integers[i], nbits=nbits)
        assert (bita == bitb)
        assert (bita.integer == integers[i])
        assert (int(bita) == bita.integer)
        assert (bita.array == arr)
        assert (bitb.integer == integers[i])
        assert (bitb.array == arr)


def test_bitstring_lsb():
    for i in range(15):
        bit = BitString.from_int(integer=i)
        bit_lsb = BitStringLSB.from_int(integer=i)
        assert (bit.integer == i)
        assert (bit.binary == format(i, 'b'))
        assert (bit.binary == bin(i)[2:])
        assert (bit_lsb.integer == bit.integer)
        assert (bit != bit_lsb)

    arrays = [[0, 0, 1], [1, 0, 0], [1, 0, 1], [1, 0, 1, 1, 1, 0], [1, 0, 0, 1, 1, 1, 0, 0, 1, 0]]
    integers = [1, 4, 5, 32 + 8 + 4 + 2, 2 + 16 + 32 + 64 + 512]
    integers_lsb = [4, 1, 5, 1 + 4 + 8 + 16, 1 + 8 + 16 + 32 + 256]

    for i, arr in enumerate(arrays):
        nbits = len(arr)

        bita = BitString.from_array(array=arr, nbits=nbits)
        bitb = BitString.from_int(integer=integers[i], nbits=nbits)
        bitc = BitStringLSB.from_array(array=arr, nbits=nbits)

        assert (bita == bitb)
        assert (bita.integer == integers[i])
        assert (bita.array == arr)
        assert (bitb.integer == integers[i])
        assert (bitb.array == arr)

        assert (bitc.integer == integers_lsb[i])
        assert (bitc.array == arr)
        assert (bitc.binary == bita.binary)


def test_conversion():
    for i in range(15):
        bita = BitString.from_int(integer=i)
        bita_lsb = BitStringLSB.from_int(integer=i)
        bita_converted = BitString.from_bitstring(other=bita_lsb)
        assert (bita == bita_converted)

    arrays = [[0, 0, 1], [1, 0, 0], [1, 0, 1], [1, 0, 1, 1, 1, 0], [1, 0, 0, 1, 1, 1, 0, 0, 1, 0]]
    for i, arr in enumerate(arrays):
        nbits = len(arr)

        bita = BitString.from_array(array=arr, nbits=nbits)
        bita_lsb = BitStringLSB.from_bitstring(other=bita)
        assert (bita_lsb.array == [x for x in reversed(arr)])
        assert (bita.binary == bita_lsb.binary[::-1])
        assert (bita.integer == bita_lsb.integer)


def test_constructor():
    for i in range(15):
        bita = BitString.from_int(integer=i)
        bita_lsb = BitStringLSB.from_int(integer=i)
        bitb = BitString.from_int(integer=bita)
        bitc = BitString.from_int(integer=bita_lsb)
        bitd = BitString.from_array(array=bita)
        bite = BitString.from_array(array=bita_lsb)
        bitf = BitString.from_binary(binary=bita)
        bitg = BitString.from_binary(binary=bita_lsb)
        assert (bita == bitb)
        assert (bita == bitc)
        assert (bita == bitd)
        assert (bita == bite)
        assert (bita == bitf)
        assert (bita == bitg)
