# make sure to use the jax/autograd numpy
from tequila.utils.exceptions import TequilaException

__AUTOGRAD__BACKEND__ = None
try:
    import jax
    from jax import numpy

    __AUTOGRAD__BACKEND__ = "jax"
except ImportError:
    try:
        import autograd as jax
        from autograd import numpy

        __AUTOGRAD__BACKEND__ = "autograd"
    except ImportError:
        raise TequilaException("Neither jax nor autograd found on your system")
