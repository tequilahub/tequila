from tequila.optimizers.optimizer_base import OptimizerHistory
from tequila.optimizers.optimizer_scipy import OptimizerSciPy
from shutil import which
has_phoenics= which("phoenics") is not None
if has_phoenics:
    from tequila.optimizers.optimizer_phoenics import PhoenicsOptimizer

