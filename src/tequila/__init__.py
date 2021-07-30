from tequila.utils import BitString, BitNumbering, BitStringLSB, initialize_bitstring, TequilaException, TequilaWarning
from tequila.circuit import gates, QCircuit, NoiseModel
from tequila.hamiltonian import paulis, QubitHamiltonian, PauliString
from tequila.objective import Objective, VectorObjective,\
    ExpectationValue, Variable, assign_variable, format_variable_dictionary,\
    vectorize

from tequila.optimizers import INSTALLED_OPTIMIZERS, show_available_optimizers
from tequila.optimizers import minimize, minimize_scipy, minimize_gd, optimizer_scipy

from tequila.simulators.simulator_api import simulate, compile, compile_to_function, draw, pick_backend, \
    INSTALLED_SAMPLERS, \
    INSTALLED_SIMULATORS, SUPPORTED_BACKENDS, INSTALLED_BACKENDS, show_available_simulators
from tequila.wavefunction import QubitWaveFunction
from tequila.circuit.qasm import export_open_qasm, import_open_qasm, import_open_qasm_from_file
from tequila.circuit.pyzx import convert_to_pyzx, convert_from_pyzx
import tequila.quantumchemistry as chemistry # shortcut
from tequila.quantumchemistry import Molecule, MoleculeFromOpenFermion

# make sure to use the jax/autograd numpy for objectives
from tequila.circuit.gradient import grad
from tequila.autograd_imports import numpy, jax, __AUTOGRAD__BACKEND__

# get rid of the jax GPU/CPU warnings
import warnings

warnings.filterwarnings("ignore", module="jax")
warnings.filterwarnings("ignore", module="absl")
warnings.filterwarnings("default", category=TequilaWarning)

# load applications and helpers
from tequila.apps import adapt

from .version import __version__, __author__
