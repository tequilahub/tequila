import typing
import numbers
from tequila import BitNumbering, BitString, BitStringLSB


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
        if isinstance(input_state, numbers.Integral):
            return BitString.from_int(integer=input_state)
        else:
            return BitString.from_int(integer=input_state.integer, nbits=input_state.nbits)


class KeyMapMSB2LSB(KeyMapABC):

    @property
    def numbering(self) -> BitNumbering:
        return BitNumbering.LSB

    def __call__(self, input_state: BitString, initial_state: int = None) -> BitStringLSB:
        if isinstance(input_state, numbers.Integral):
            return BitStringLSB.from_int(integer=input_state)
        else:
            return BitStringLSB.from_int(integer=input_state.integer, nbits=input_state.nbits)


class KeyMapSubregisterToRegister(KeyMapABC):

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
        self._subregister = sorted(subregister)
        self._register = sorted(register)

    def make_complement(self):
        return [i for i in self._register if i not in self._subregister]

    def __call__(self, input_state: BitString, initial_state: BitString = None) -> BitString:
        if initial_state is None:
            initial_state = BitString.from_int(integer=0)

        input_state = BitString.from_int(integer=input_state, nbits=len(self._subregister))
        initial_state = BitString.from_int(integer=initial_state, nbits=len(self._subregister))
        output_state = BitString.from_int(integer=initial_state.integer, nbits=len(self._register))
        for k, v in enumerate((self._subregister)):
            output_state[v] = input_state[k]

        return output_state

    def inverted(self, input_state: int) -> BitString:
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


class KeyMapRegisterToSubregister(KeyMapSubregisterToRegister):

    def __call__(self, input_state: BitString, initial_state: BitString = None) -> BitString:
        """
        Map from register to subregister
        :param input_state:
        :return: input_state only on subregister
        """
        input_state = BitString.from_int(integer=input_state.integer, nbits=len(self._register))
        output_state = BitString.from_int(integer=0, nbits=len(self._subregister))
        for k, v in enumerate(self._subregister):
            output_state[k] = input_state[v]
        return output_state

    def __repr__(self):
        return "keymap:\n" + "register    = " + str(self.register) + "\n" + "subregister = " + str(self.subregister)
