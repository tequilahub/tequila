from .circuit import QCircuit
from .noise import NoiseModel
from .qpic import export_to
from .compiler import CircuitCompiler as CircuitCompiler

def compile_circuit(U, *args, **kwargs):
    # see CircuitCompiler documentation
    c = CircuitCompiler.standard_gate_set(*args, **kwargs)
    return c(U)