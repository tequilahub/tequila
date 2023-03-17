from .utils_ml import TequilaMLException
from shutil import which
from tequila.objective import Objective
SUPPORTED_PLATFORMS = ['pytorch']
CONVERTERS = {}

#HAS_TORCH = which('torch') is not None or which('pytorch') is not None
HAS_TORCH = True
try:
    import torch
except:
    HAS_TORCH = False

if HAS_TORCH:
    from .interface_torch import TorchLayer
    CONVERTERS['pytorch'] = TorchLayer

def to_platform(objective: Objective, platform: str,
                compile_args: dict = None, input_vars: list = None):
    plat = platform.lower()
    if plat == 'torch':
        # common alias.
        plat = 'pytorch'

    try:
        f = CONVERTERS[plat]
        return f(objective, compile_args, input_vars)
    except KeyError:
        raise TequilaMLException('Desired ML platform {} either not supported, or not installed.'.format(plat))
