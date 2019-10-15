import typing
from openvqe import BitNumbering, BitString, BitStringLSB


class KeyMapABC:

    @property
    def n_qubits(self):
        return None

    @property
    def numbering(self):
        return BitNumbering.MSB

    def __call__(self, input_state: BitString, initial_state: BitString = 0):
        return input_state


class KeyMapLSB2MSB(KeyMapABC):

    def __call__(self, input_state: BitStringLSB, initial_state: int = None) -> BitString:
        return BitString.from_int(integer=input_state)


class KeyMapMSB2LSB(KeyMapABC):

    @property
    def numbering(self) -> BitNumbering:
        return BitNumbering.LSB

    def __call__(self, input_state: BitString, initial_state: int = None) -> BitStringLSB:
        return BitStringLSB.from_int(integer=input_state)


class KeyMapQubitSubregister(KeyMapABC):

    @property
    def n_qubits(self):
        return len(self.register)

    @property
    def register(self):
        return self._register

    @property
    def subregister(self):
        return self._subregister

    @property
    def complement(self):
        return self.make_complement()

    def __init__(self, subregister: typing.List[int], register: typing.List[int]):
        self._subregister = subregister
        self._register = register

    def make_complement(self):
        return [i for i in self._register if i not in self._subregister]

    def __call__(self, input_state: BitString, initial_state: BitString = 0):

        input_state = BitString.from_int(integer=input_state, nbits=len(self._subregister))
        initial_state = BitString.from_int(integer=initial_state, nbits=len(self._subregister))

        output_state = BitString.from_int(integer=initial_state.integer, nbits=len(self._register))
        for k, v in enumerate((self._subregister)):
            output_state[v] = input_state[k]

        return output_state

    def inverted(self, input_state: int):
        """
        Map from register to subregister
        :param input_state:
        :return: input_state only on subregister
        """
        input_state = BitString.from_int(integer=input_state, nbits=len(self._register))
        output_state = BitString.from_int(integer=0, nbits=len(self._subregister))
        for k, v in enumerate(self._subregister):
            output_state[k] = input_state[v]
        return output_state

    def __repr__(self):
        return "keymap:\n" + "register    = " + str(self.register) + "\n" + "subregister = " + str(self.subregister)