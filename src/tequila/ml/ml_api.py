from .utils_ml import TequilaMLException
from shutil import which
from tequila.objective import Objective
SUPPORTED_PLATFORMS = ['pytorch', 'tensorflow']
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


HAS_TF = True
try:
    import tensorflow
except:
    HAS_TF = False

if HAS_TF:
    from .interface_tf import TFLayer
    CONVERTERS['tensorflow'] = TFLayer


def to_platform(objective: Objective, platform: str,
                compile_args: dict = None, input_vars: list = None):
    plat = platform.lower()
    if plat == 'torch':
        # common alias.
        plat = 'pytorch'

    if plat == 'tf':
        # common alias
        plat = 'tensorflow'

    try:
        f = CONVERTERS[plat]
        return f(objective, compile_args, input_vars)
    except KeyError:
        raise TequilaMLException('Desired ML platform {} either not supported, or not installed.'.format(plat))
