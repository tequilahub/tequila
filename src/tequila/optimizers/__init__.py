from tequila.optimizers.optimizer_base import OptimizerHistory
from tequila.optimizers.optimizer_scipy import OptimizerSciPy
from shutil import which
has_phoenics = False
try:
    import phoenics
    has_phoenics = True
except ImportError:
    has_phoenics = False

if has_phoenics:
    from tequila.optimizers.optimizer_phoenics import PhoenicsOptimizer

