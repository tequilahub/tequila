from copy import deepcopy
from typing import Union
import numpy as np
import tequila as tq
from tequila import QCircuit, QubitWaveFunction, BitNumbering


class Postselection:
    """
    A class representing a postselection operation on a set of qubits,
    i.e. a projection of the wavefunction onto the subspace where all
    selected qubits are in the state |0>.
    """

    def __init__(self, qubits: list[int]):
        self._qubits = qubits

    @property
    def qubits(self):
        return self._qubits

    @qubits.setter
    def qubits(self, qubits: list[int]):
        self._qubits = qubits

    def mask(self, nbits: int, numbering: BitNumbering) -> int:
        """
        Returns a bitmask for the postselected qubits.
        """
        mask = 0
        for qubit in self.qubits:
            if numbering == BitNumbering.LSB:
                mask |= 1 << qubit
            else:
                mask |= 1 << (nbits - qubit - 1)
        return mask


class PostselectionCircuit:
    """
    An extended circuit class that supports Postselection operations
    which project the wavefunction onto the subspace where a specified
    set of qubits are all in the state |0>.

    This works by storing a list of fragments which can be either QCircuits
    or Postselection objects. These fragments are processed in order by
    passing the result from the previous fragment as the initial state of its
    successor. When a postselection is encountered, the bitmask is used to
    check which amplitudes belong to the projected subspace, and the rest
    is set to 0.
    """

    def __init__(self, circuit: tq.QCircuit = None):
        self._fragments: list[Union[QCircuit, Postselection]] = []
        if circuit is not None:
            self._fragments = [circuit]

    def simulate(
        self,
        backend: str = None,
        initial_wfn: Union[QubitWaveFunction, int] = 0,
        repetitions: int = 1,
        optimize_circuit: bool = True,
    ):
        backend = tq.pick_backend(backend=backend)
        numbering = tq.INSTALLED_SIMULATORS[tq.pick_backend(backend)].CircType.numbering

        compiled = {}
        for i, fragment in enumerate(self._fragments):
            if isinstance(fragment, QCircuit):
                # TODO: Handle empty qubits properly instead of doing this
                for j in range(self.n_qubits):
                    fragment += tq.gates.X(target=j)
                    fragment += tq.gates.X(target=j)
                compiled[i] = tq.compile(fragment, backend=backend, optimize_circuit=optimize_circuit)

        wfn = initial_wfn
        for _ in range(repetitions):
            for i, fragment in enumerate(self._fragments):
                if isinstance(fragment, QCircuit):
                    wfn = compiled[i](initial_state=wfn)
                elif isinstance(fragment, Postselection):
                    amplitudes = wfn.to_array(numbering, copy=False)
                    mask = fragment.mask(self.n_qubits, numbering)
                    indices = np.arange(2**self.n_qubits) & mask != 0
                    amplitudes[indices] = 0
                    wfn = QubitWaveFunction.from_array(amplitudes, numbering, copy=False)
        norm = np.linalg.norm(wfn.to_array(numbering))
        # TODO: Reconsider how to handle norm == 0.0
        normalized_wfn = (1.0 / norm) * wfn if norm != 0.0 else 0.0 * wfn
        return normalized_wfn, norm

    def __iadd__(self, other: Union[QCircuit, Postselection, "PostselectionCircuit"]):
        if isinstance(other, QCircuit) or isinstance(other, Postselection):
            self.add_fragment(other)
        elif isinstance(other, PostselectionCircuit):
            for fragment in other._fragments:
                self.add_fragment(fragment)
        return self

    def __add__(self, other: Union[QCircuit, Postselection, "PostselectionCircuit"]):
        result = deepcopy(self)
        result += other
        return result

    def add_fragment(self, fragment: Union[QCircuit, Postselection]):
        if self._fragments and isinstance(self._fragments[-1], QCircuit) and isinstance(fragment, QCircuit):
            self._fragments[-1] += fragment
        elif self._fragments and isinstance(self._fragments[-1], Postselection) and isinstance(fragment, Postselection):
            self._fragments[-1].qubits += fragment.qubits
        else:
            self._fragments.append(fragment)
        return self

    @property
    def n_qubits(self):
        if self._fragments:
            return self._fragments[0].n_qubits
        else:
            return 0
