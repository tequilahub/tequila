"""
Project out undesired measurements
"""

from openvqe.qubit_wavefunction import QubitWaveFunction, BitString
from openvqe import typing
from openvqe.keymap import KeyMapRegisterToSubregister, KeyMapSubregisterToRegister, KeyMapABC
from openvqe.tools import list_assignement

class HeraldingABC:

    keymap = KeyMapABC() # has no effect, overwrite

    def __call__(self, input: QubitWaveFunction, *args, **kwargs) -> QubitWaveFunction:
        output=QubitWaveFunction()
        for k,v in input.items():
            if self.is_valid(key=self.keymap(k)):
                key, value = self.transform(key=k, value=v)
                output[key] = value
        return output

    def is_valid(self, key:BitString) -> bool:
        # has no effect, overwrite
        return True

    def transform(self, key: BitString, value: int) -> typing.Tuple[BitString, int]:
        # has no effect, overwrite
        return (key, value)

class HeraldingProjector(HeraldingABC):

    def __init__(self, register: typing.List[int],  subregister: typing.List[int], projector_space: typing.List[BitString], delete_subregister=True):
        self.keymap = KeyMapRegisterToSubregister(subregister=subregister, register=register)
        self.projector_space = self.assign_projector_space(projector_space=projector_space)
        self.delete_subregister=delete_subregister

    def transform(self, key: BitString, value: int) -> typing.Tuple[BitString, int]:
        if self.delete_subregister:
            map = KeyMapRegisterToSubregister(register=self.keymap.register, subregister=self.keymap.complement)
            return map(key), value
        else:
            return key, value

    def assign_projector_space(self, projector_space: typing.List[BitString]):
        """
        Automatic assignement of the projector space depeding on input types
        """
        if isinstance(projector_space, QubitWaveFunction):
            return projector_space.state
        projector_space = list_assignement(projector_space)
        result = []
        for i in projector_space:
            if isinstance(i, BitString):
                result.append(i)
            elif isinstance(i, str):
                result.append(BitString.from_binary(binary=i, nbits=len(self.keymap.subregister)))
            else:
                result.append(BitString.from_int(integer=i, nbits=len(self.keymap.subregister)))
        return result

    def is_valid(self, key:BitString) -> bool:
        if key in self.projector_space:
            return True
        else:
            return False




