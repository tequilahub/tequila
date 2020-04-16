from tequila.utils import BitString, BitNumbering, BitStringLSB, initialize_bitstring, TequilaException
from tequila.circuit import gates, QCircuit,NoiseModel
from tequila.hamiltonian import paulis, QubitHamiltonian, PauliString
from tequila.objective import Objective, ExpectationValue, Variable, assign_variable, format_variable_dictionary
from tequila.optimizers import optimizer_scipy

from tequila.optimizers import has_phoenics
if has_phoenics:
    from tequila.optimizers import optimizer_phoenics
from tequila.optimizers import has_gpyopt
if has_gpyopt:
    from tequila.optimizers import optimizer_gpyopt

from tequila.simulators.simulator_api import simulate, compile, compile_to_function, draw, pick_backend, INSTALLED_SAMPLERS, \
    INSTALLED_SIMULATORS, SUPPORTED_BACKENDS, INSTALLED_BACKENDS, show_available_simulators
from tequila.wavefunction import QubitWaveFunction
import tequila.quantumchemistry as chemistry

# make sure to use the jax/autograd numpy
from tequila.circuit.gradient import grad
from tequila.autograd_imports import numpy, jax, __AUTOGRAD__BACKEND__

# get rid of the jax GPU/CPU warnings
import warnings

warnings.filterwarnings("ignore", module="jax")

__version__ = "BorisYeltsin"
