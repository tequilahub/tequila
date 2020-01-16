from tequila.utils import BitString, BitNumbering, BitStringLSB, initialize_bitstring, TequilaException
from tequila.circuit import gates
from tequila.hamiltonian import paulis
from tequila.objective import Objective, ExpectationValue, Variable, assign_variable
from tequila.optimizers import optimizer_scipy
from tequila.simulators import pick_simulator
from tequila.wavefunction import QubitWaveFunction

def chemistry(backend="psi4"):
    import tequila.quantumchemistry as chem
    return chem.pick_backend(backend)

__version__ = "The Ghost of Mittens The Kitten"
