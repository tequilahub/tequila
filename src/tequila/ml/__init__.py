from .ml_api import to_platform
from .ml_api import HAS_TORCH, HAS_TF
if HAS_TORCH:
    from .interface_torch import TorchLayer
