# make sure to use the jax/autograd numpy
# will import either jax or autograd, depending on what is available on the system
from tequila.utils.exceptions import TequilaException

__AUTOGRAD__BACKEND__ = None
try:
    import jax
    from jax import numpy
    jax.config.update('jax_platform_name', 'cpu')
    __AUTOGRAD__BACKEND__ = "jax"
except:
    # will pick autograd if jax is not installed
    # or if there are errors on import (like on M2 chips or when jax/jaxlib are not matching
    try:
        import autograd as jax
        from autograd import numpy

        __AUTOGRAD__BACKEND__ = "autograd"
    except ImportError:
        raise TequilaException("Neither jax nor autograd found on your system")

def change_classical_differentiation_backend(name:str):
    if name.lower() == "jax":
        try:
            import jax
            from jax import numpy
        except:
            raise TequilaException("failed to load jax as classical differentiation backend")
    elif name.lower() == "autograd":
        try:
            import autograd as jax
            from autograd import numpy
        except:
            raise TequilaException("failed to load autograd as classical differentiation backend")

    else:
        raise TequilaException("unknown differentiation backend: {}, try jax or autograd".format(name))

    return True

def status():
    print("currently loaded autodiff numpy: {}, {}".format(numpy.__name__, numpy))
    print("currently loaded autodiff library: {}, {}".format(jax.__name__, jax))
