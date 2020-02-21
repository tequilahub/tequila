from tequila.utils import BitString, BitNumbering, BitStringLSB, initialize_bitstring, TequilaException
from tequila.circuit import gates, QCircuit
from tequila.hamiltonian import paulis, QubitHamiltonian, PauliString
from tequila.objective import Objective, ExpectationValue, Variable, assign_variable, format_variable_dictionary
from tequila.optimizers import optimizer_scipy
from tequila.simulators import pick_backend
from tequila.wavefunction import QubitWaveFunction
from tequila.simulators import simulate_objective, simulate_wavefunction, sample_objective, sample_wavefunction
import tequila.quantumchemistry as chemistry

from typing import Dict, Union, Hashable
from numbers import Real as RealNumber

# make sure to use the jax/autograd numpy
from tequila.autograd_imports import numpy, __AUTOGRAD__BACKEND__
from tequila.circuit.gradient import grad


# get rid of the jax GPU/CPU warnings
import warnings
warnings.filterwarnings("ignore", module="jax")


def simulate(objective: Objective,
             variables: Dict[Union[Variable, Hashable], RealNumber] = None,
             samples: int = None,
             backend: str = None,
             *args,
             **kwargs) -> Union[RealNumber, QubitWaveFunction]:
    """Simulate a tequila objective or circuit

    Parameters
    ----------
    objective :
        tequila objective or circuit
    variables :
        The variables of the objective given as dictionary
        with keys as tequila Variables/hashable types and values the corresponding real numbers
    samples : int : (Default value = None)
        if None a full wavefunction simulation is performed, otherwise a fixed number of samples is simulated
    backend : str : (Default value = None)
        specify the backend or give None for automatic assignment

    *args :
        
    **kwargs :
        

    Returns
    -------
    type
        simulated/sampled objective or simulated/sampled wavefunction

    """
    if variables is None and not (len(objective.extract_variables()) == 0):
        raise TequilaException("You called simulate for a parametrized type but forgot to pass down the variables: {}".format(objective.extract_variables()))
    elif variables is not None:
        # allow hashable types as keys without casting it to variables
        variables = {assign_variable(k): v for k, v in variables.items()}

    if isinstance(objective, Objective) or hasattr(objective, "args"):
        if samples is None:
            return simulate_objective(objective=objective, variables=variables, backend=backend)
        else:
            return sample_objective(objective=objective, variables=variables, samples=samples, backend=backend)
    elif isinstance(objective, QCircuit) or hasattr(objective, "gates"):
        if samples is None:
            return simulate_wavefunction(abstract_circuit=objective, variables=variables, backend=backend, *args, **kwargs)
        else:
            return sample_wavefunction(abstract_circuit=objective, variables=variables, samples=samples, backend=backend, *args, **kwargs)
    else:
        raise TequilaException(
            "Don't know how to simulate object of type: {type}, \n{object}".format(type=type(objective),
                                                                                   object=objective))

def draw(objective, variables=None, backend:str=None):
    """

    Pretty output (depends on installed backends)

    Parameters
    ----------
    objective :
        the tequila objective to print out
    variables :
         (Default value = None)
         Give variables if the objective is parametrized
    backend:str :
         (Default value = None)
         chose backend (of None it will be automatically picked)
    """
    if backend is None:
        if "cirq" in simulators.INSTALLED_SIMULATORS:
            backend = "cirq"
        elif "qiskit" in simulators.INSTALLED_SIMULATORS:
            backend = "qiskit"

    if isinstance(objective, Objective):
        #pretty printer not here yet
        print(objective)
    else:
        if backend is None:
            print(objective)
        else:
            if variables is None:
                variables = {}
            for k in objective.extract_variables():
                if k not in variables:
                    variables[k] = 0.0
            variables = format_variable_dictionary(variables)
            compiled=simulators.compile_circuit(abstract_circuit=objective, backend=backend, variables=variables)
            print(compiled.circuit)

def compile(objective: Objective,
             variables: Dict[Union[Variable, Hashable], RealNumber] = None,
             samples: int = None,
             backend: str = None,
             *args,
             **kwargs) -> Union[simulators.BackendCircuit]:
    """Compile a tequila objective or circuit to a backend

    Parameters
    ----------
    objective : Objective:
        tequila objective or circuit
    variables : Dict[Union[Variable :Hashable]:RealNumber]:
        The variables of the objective given as dictionary
        with keys as tequila Variables and values the corresponding real numbers
    samples : str : (Default value = None) :
        if None a full wavefunction simulation is performed, otherwise a fixed number of samples is simulated
    backend : str : (Default value = None) :
        specify the backend or give None for automatic assignment

    Returns
    -------
    simulators.BackendCircuit
        simulated/sampled objective or simulated/sampled wavefunction

    """
    if variables is None and not (len(objective.extract_variables()) == 0):
        raise TequilaException("You called simulate for a parametrized type but forgot to pass down the variables: {}".format(objective.extract_variables()))
    elif variables is not None:
        # allow hashable types as keys without casting it to variables
        variables = {assign_variable(k): v for k, v in variables.items()}

    backend = simulators.pick_backend(backend=backend, samples=samples)

    if isinstance(objective, Objective) or hasattr(objective, "args"):
        return simulators.compile_objective(objective=objective, variables=variables, backend=backend)
    elif isinstance(objective, QCircuit) or hasattr(objective, "gates"):
        return simulators.compile_circuit(abstract_circuit=objective, variables=variables, backend=backend, *args, **kwargs)
    else:
        raise TequilaException(
            "Don't know how to compile object of type: {type}, \n{object}".format(type=type(objective),
                                                                                   object=objective))

__version__ = "AndreasDorn"
