from tequila.optimizers.optimizer_base import OptimizerHistory
from tequila.optimizers.optimizer_scipy import OptimizerSciPy
from shutil import which
has_phoenics = False
try:
    import phoenics
    has_phoenics = True
except ImportError:
    has_phoenics = False

has_gpyopt = False
try:
    import GPyOpt
    has_gpyopt = True
except ImportError:
    has_gpyopt = False

if has_phoenics:
    from tequila.optimizers.optimizer_phoenics import PhoenicsOptimizer

if has_gpyopt:
    from tequila.optimizers.optimizer_gpyopt import GPyOptOptimizer
