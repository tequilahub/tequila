from dataclasses import dataclass
from typing import List, Union

from tequila.circuit.noise import NoiseModel
from tequila.circuit.noise import BitFlip, PhaseFlip, AmplitudeDamp, PhaseDamp, PhaseAmplitudeDamp, DepolarizingError

# composed noise models names
COMPOSED_BITFLIP_DEPOLARIZING = "bitflip_depolarizing"
COMPOSED_BITFLIP_PHASEDAMP = "bitflip_phasedamp"
COMPOSED_BITFLIP_AMPLITUDEDAMP = "bitflip_amplitudedamp"
COMPOSED_TWO_BITFLIPS = "two_bitflips"

DEPOLARIZING = "depolarizing"
BITFLIP = "bit_flip"

basic_noise_models = {
    "bitflip": BitFlip,
    "phaseflip": PhaseFlip,
    "amplitudedamp": AmplitudeDamp,
    "phasedamp": PhaseDamp,
    "phaseamplitudedamp": PhaseAmplitudeDamp,
    "depolarizingerror": DepolarizingError,
    "depolarizing": DepolarizingError
}


@dataclass
class NoiseConfig:
    name: str
    p: float
    level: int


class ComposedNoiseModel:
    """
    convenience class for creating compositions of noise models
    """

    def __init__(self, noise_model_name, noise_configs: List[NoiseConfig]):
        self._name = noise_model_name
        self._noise_configs = self._clean_config(noise_configs)
        self._noise_model = self._build_noise_model()

    def _build_noise_model(self) -> NoiseModel:
        cfgs = self._noise_configs
        noise_model = basic_noise_models[cfgs[0].name](p=cfgs[0].p, level=cfgs[0].level)

        for c in cfgs[1:]:
            noise_model += basic_noise_models[c.name](p=c.p, level=c.level)

        return noise_model

    @staticmethod
    def _clean_config(noises: List[NoiseConfig]) -> List[NoiseConfig]:
        noises_clean = []
        for n in noises:
            noises_clean.append(NoiseConfig(name=n.name.lower().replace('_', ''), p=n.p, level=n.level))

        return noises_clean

    @property
    def noise_model(self) -> NoiseModel:
        return self._noise_model

    @property
    def noise_configs(self):
        return self._noise_configs

    @property
    def name(self):
        return self._name


def get_noise_model(noise_model_name_or_id) -> Union[NoiseModel, str] or None:
    noise_model_name_or_id = str(noise_model_name_or_id)

    if noise_model_name_or_id in ['-1', None]:
        return None

    if noise_model_name_or_id.lower() in ['device', "0"]:
        return 'device'

    if noise_model_name_or_id.lower() in [COMPOSED_BITFLIP_DEPOLARIZING, "1"]:
        composed = ComposedNoiseModel(noise_model_name=COMPOSED_BITFLIP_DEPOLARIZING,
                                      noise_configs=[NoiseConfig(name='bit_flip', p=1e-2, level=1),
                                                     NoiseConfig(name='depolarizing', p=1e-2, level=2)])
        return composed.noise_model

    if noise_model_name_or_id.lower() in [COMPOSED_TWO_BITFLIPS, "2"]:
        composed = ComposedNoiseModel(noise_model_name=COMPOSED_TWO_BITFLIPS,
                                      noise_configs=[NoiseConfig(name='bit_flip', p=1e-2, level=1),
                                                     NoiseConfig(name='bit_flip', p=1e-3, level=2)])
        return composed.noise_model

    if noise_model_name_or_id.lower() in [COMPOSED_BITFLIP_PHASEDAMP, "3"]:
        composed = ComposedNoiseModel(noise_model_name=COMPOSED_BITFLIP_PHASEDAMP,
                                      noise_configs=[NoiseConfig(name='bit_flip', p=1e-2, level=1),
                                                     NoiseConfig(name='phasedamp', p=5e-3, level=2)])
        return composed.noise_model

    if noise_model_name_or_id.lower() in [COMPOSED_BITFLIP_AMPLITUDEDAMP, "4"]:
        composed = ComposedNoiseModel(noise_model_name=COMPOSED_BITFLIP_AMPLITUDEDAMP,
                                      noise_configs=[
                                          NoiseConfig(name='bit_flip', p=1e-2, level=1),
                                          NoiseConfig(name='amplitudedamp', p=5e-3, level=2)])
        return composed.noise_model

    if noise_model_name_or_id.lower() in [DEPOLARIZING, "5"]:
        noise_model = ComposedNoiseModel(noise_model_name=DEPOLARIZING,
                                         noise_configs=[NoiseConfig(name='depolarizing', p=5e-3, level=1)])
        return noise_model.noise_model

    if noise_model_name_or_id.lower() in [BITFLIP, "6"]:
        noise_model = ComposedNoiseModel(noise_model_name=BITFLIP,
                                         noise_configs=[NoiseConfig(name='bit_flip', p=1e-2, level=1)])
        return noise_model.noise_model

    raise NotImplementedError(f"noise_model_name {noise_model_name_or_id} not implemented")
