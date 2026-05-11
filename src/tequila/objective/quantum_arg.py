from __future__ import annotations
import abc

class QuantumArg(abc.ABC):
    @abc.abstractmethod
    def compile(self) -> tuple:
        """
        Returns (real_obj, imag_obj) where both are Objective instances.
        imag_obj may be None for purely real results.
        """
 
    @abc.abstractmethod
    def extract_variables(self) -> list:
        """Return all Variable objects this quantum arg depends on."""
 
    @abc.abstractmethod
    def count_measurements(self) -> int:
        """Return the total number of Pauli strings / measurements needed."""
 
    @abc.abstractmethod
    def map_variables(self, variables: dict, *args, **kwargs) -> "QuantumArg":
        """Return a copy with variables substituted."""
 
    @abc.abstractmethod
    def map_qubits(self, qubit_map: dict) -> "QuantumArg":
        """Return a copy with qubits remapped."""

def is_quantum_arg(obj) -> bool:
    if isinstance(obj, QuantumArg):
        return True
    return callable(getattr(obj, "compile", None))