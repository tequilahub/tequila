from tequila.simulators.simulator_base import BackendCircuit, QCircuit, BackendExpectationValue
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila import TequilaException
from tequila import BitString, BitNumbering
from tequila.utils.keymap import KeyMapRegisterToSubregister
from tequila.utils import to_float
from typing import Union

import numpy as np
from mqt.ddsim.pyddsim import CircuitSimulator
from mqt.core.ir import QuantumComputation
from mqt.core.ir.operations import StandardOperation, SymbolicOperation, NonUnitaryOperation, Control, OpType
from mqt.core.ir.symbolic import Variable, Expression, Term

import numbers
from tequila.circuit.compiler import change_basis


def set_computational_basis(state: int, n_qubits: int) -> QuantumComputation:
    qc = QuantumComputation(n_qubits)
    bitstring = BitString.from_int(state, n_qubits)
    for i, bit in enumerate(bitstring):
        if bit:
            qc.x(i)
    return qc


class TequilaDDSimException(TequilaException):
    def __str__(self):
        return "Error in DDSim backend:" + self.message


class BackendCircuitDDSim(BackendCircuit):
    """
    Class representing circuits compiled to DDSim.
    See BackendCircuit for documentation of features and methods inherited therefrom

    Attributes
    ----------
    counter:
        counts how many distinct ddsim Variables are employed in the circuit.
    op_lookup: dict:
        dictionary mapping strings (tequila gate names) to ddsim objects (mqt.core.ir).
    resolver:
        dictionary for resolving parameters at runtime for circuits.
    tq_to_ddsim: dict:
        dictionary mapping tequila Variables and Objectives to ddsim Variables, for parameter resolution.
    """

    numbering = BitNumbering.LSB

    quantum_state_class = QuantumComputation

    # TODO: Set to true and use mqt.core.dd to simulate (only supports unitary operations i.e no sampling or reset)
    supports_sampling_initialization = False
    supports_generic_initialization = False

    def __init__(self, abstract_circuit: QCircuit, variables: dict, *args, **kwargs):
        """

        Parameters
        ----------
        abstract_circuit: QCircuit:
            the circuit to compile to ddsim
        variables: dict:
            values of all variables in the circuit, to compile with.
        args
        kwargs
        """
        self.op_lookup = {
            "I": OpType.i,
            "X": OpType.x,
            "Y": OpType.y,
            "Z": OpType.z,
            "H": OpType.h,
            "Rx": OpType.rx,
            "Ry": OpType.ry,
            "Rz": OpType.rz,
            "SWAP": OpType.swap,
        }

        self.counter = 0
        self.tq_to_ddsim = {}

        self.resolver = None
        qubit_map = {q: i for i, q in enumerate(abstract_circuit.qubits)}

        super().__init__(abstract_circuit=abstract_circuit, variables=variables, qubit_map=qubit_map, *args, **kwargs)

    def initialize_state(self, n_qubits: int = None, initial_state: Union[int, QubitWaveFunction] = None):
        if n_qubits is None:
            n_qubits = self.n_qubits

        state = self.quantum_state_class(n_qubits)

        if isinstance(initial_state, int):
            state = set_computational_basis(initial_state, n_qubits)
        elif isinstance(initial_state, QubitWaveFunction):
            raise TequilaDDSimException("backend does not support arbitrary initial states")

        return state

    def update_variables(self, variables: dict):
        """
        set new variable values for the circuit.
        Parameters
        ----------
        variables: dict:
            the variables to supply to the circuit.

        Returns
        -------
        None
        """
        if isinstance(variables, dict):
            variables = {k: to_float(v) for k, v in variables.items()}

        if len(self.tq_to_ddsim.keys()) > 0:
            self.resolver = {k: v(variables) for v, k in self.tq_to_ddsim.items()}

    def do_simulate(self, variables, initial_state: Union[int, QubitWaveFunction], *args, **kwargs):
        """
        Helper function to perform simulation.

        Parameters
        ----------
        variables: dict:
            variables to supply to the circuit.
        initial_state:
            information indicating the initial state on which the circuit should act.
        args
        kwargs

        Returns
        -------
        QubitWaveFunction:
            QubitWaveFunction representing result of the simulation.
        """
        state = self.initialize_state(self.n_qubits, initial_state).to_operation()
        if state is not None:
            self.circuit.insert(idx=0, op=state)
        sim_kwargs = {
            "approximation_step_fidelity": kwargs.get("approximation_step_fidelity", 1),
            "approximation_steps": kwargs.get("approximation_steps", 1),
            "approximation_strategy": kwargs.get("approximation_strategy", "fidelity"),
            "seed": kwargs.get("seed", -1),
        }
        if self.resolver is not None:
            circuit = self.circuit.instantiate(self.resolver)
        else:
            circuit = self.circuit
        sim = CircuitSimulator(circuit, **sim_kwargs)
        sim.simulate(shots=0)
        vec = sim.get_constructed_dd().get_vector()
        wfn = QubitWaveFunction.from_array(array=np.array(vec, copy=False), numbering=self.numbering)
        return wfn

    def convert_measurements(self, backend_result, target_qubits=None) -> QubitWaveFunction:
        """
        Transform backend evaluation results into QubitWaveFunction
        Parameters
        ----------
        backend_result:
            the return value of backend simulation.

        Returns
        -------
        QubitWaveFunction
            results transformed to tequila native QubitWaveFunction
        """
        result = QubitWaveFunction(self.n_qubits, self.numbering)

        for k, v in backend_result.items():
            # ddsim uses LSB bitstrings, but from_binary expects MSB
            converted_key = BitString.from_binary(k[::-1])
            result[converted_key] = v

        if target_qubits is not None:
            mapped_target = [self.qubit_map[q].number for q in target_qubits]
            mapped_full = [self.qubit_map[q].number for q in self.abstract_qubits]
            keymap = KeyMapRegisterToSubregister(subregister=mapped_target, register=mapped_full)
            result = QubitWaveFunction.from_wavefunction(result, keymap, n_qubits=len(target_qubits))

        return result

    def do_sample(self, samples, circuit, read_out_qubits, initial_state=0, *args, **kwargs) -> QubitWaveFunction:
        """
        Helper function for performing sampling.

        Parameters
        ----------
        samples: int:
            the number of samples to be taken.
        circuit:
            the circuit to sample from.
        initial_state:
            initial state to apply the circuit to.
        args
        kwargs

        Returns
        -------
        QubitWaveFunction:
            the results of sampling, as a Qubit Wave Function.
        """
        state = self.initialize_state(self.n_qubits, initial_state).to_operation()
        if state is not None:
            circuit.insert(idx=0, op=state)
        sim = CircuitSimulator(circuit)
        sampled = sim.simulate(samples)
        return self.convert_measurements(backend_result=sampled, target_qubits=read_out_qubits)

    def initialize_circuit(self, *args, **kwargs) -> QuantumComputation:
        """
        return an empty circuit.
        Parameters
        ----------
        args
        kwargs

        Returns
        -------
        QuantumComputation
        """
        return QuantumComputation(self.n_qubits, self.n_qubits)

    def add_parametrized_gate(self, gate, circuit, *args, **kwargs):
        """
        add a parametrized gate.
        Parameters
        ----------
        gate: QGateImpl:
            the gate to add to the circuit.
        circuit:
            the circuit to which the gate is to be added
        args
        kwargs

        Returns
        -------
        None
        """
        op = self.op_lookup[gate.name]
        parameter = gate.parameter

        if isinstance(parameter, float):
            par = parameter
        else:
            try:
                par = self.tq_to_ddsim[parameter]
            except Exception:
                var = Variable(
                    "{}_{}".format(
                        self._name_variable_objective(parameter),
                        str(self.counter),
                    )
                )
                par = Expression([Term(var, 1)])
                self.tq_to_ddsim[parameter] = var
                self.counter += 1

        ddsim_gate = SymbolicOperation(
            controls=set(Control(self.qubit(c)) for c in gate.control),
            targets=[self.qubit(t) for t in gate.target],
            op_type=op,
            params=[par],
        )
        circuit.append(ddsim_gate)

    def add_basic_gate(self, gate, circuit, *args, **kwargs):
        """
        add an unparametrized gate to the circuit.
        Parameters
        ----------
        gate: QGateImpl:
            the gate to be added to the circuit.
        circuit:
            the circuit, to which a gate is to be added.
        args
        kwargs

        Returns
        -------
        None
        """
        op = self.op_lookup[gate.name]
        ddsim_gate = StandardOperation(
            controls={Control(self.qubit(c)) for c in gate.control},
            targets=[self.qubit(t) for t in gate.target],
            op_type=op,
        )
        circuit.append(ddsim_gate)

    def add_measurement(self, circuit, target_qubits, *args, **kwargs):
        """
        Add a measurement operation to a circuit.
        Parameters
        ----------
        circuit:
            a circuit, to which the measurement is to be added.
        target_qubits: List[int]
            abstract target qubits
        args
        kwargs

        Returns
        -------
        circuit with measurements

        """
        tq = [self.qubit(t) for t in target_qubits]
        meas = NonUnitaryOperation(targets=tq, classics=tq)
        circuit.append(meas)
        return circuit

    # Overwriting `sample_paulistring` since mqt.ir QuantumComputation object is not pickable:
    # copy.deepcopy fails.
    def sample_paulistring(
        self, samples: int, paulistring, variables, initial_state: Union[int, QubitWaveFunction] = 0, *args, **kwargs
    ) -> numbers.Real:
        """
        Sample an individual pauli word (pauli string) and return the average result thereof.
        Parameters
        ----------
        samples: int:
            how many samples to evaluate.
        paulistring:
            the paulistring to be sampled.
        args
        kwargs

        Returns
        -------
        float:
            the average result of sampling the chosen paulistring
        """
        not_in_u = [q for q in paulistring.qubits if q not in self.abstract_qubits]
        reduced_ps = paulistring.trace_out_qubits(qubits=not_in_u)
        if reduced_ps.coeff == 0.0:
            return 0.0
        if len(reduced_ps._data.keys()) == 0:
            return reduced_ps.coeff

        # make basis change and translate to backend
        basis_change = QCircuit()
        qubits = []
        for idx, p in reduced_ps.items():
            qubits.append(idx)
            basis_change += change_basis(target=idx, axis=p)

        # Simple fix to the `copy` problem
        if self.resolver is not None:
            _circuit = self.circuit.instantiate(self.resolver)
        else:
            r = {Variable("_"): 0}  # Dummy variable
            _circuit = self.circuit.instantiate(r)

        # add basis change to the circuit
        # can be circumvented by optimizing the measurements
        # on construction: tq.ExpectationValue(H=H, U=U, optimize_measurements=True)
        circuit = self.create_circuit(circuit=_circuit, abstract_circuit=basis_change)
        # run simulators
        counts = self.sample(
            samples=samples,
            circuit=circuit,
            read_out_qubits=qubits,
            variables=variables,
            initial_state=initial_state,
            *args,
            **kwargs,
        )
        # compute energy
        E = 0.0
        n_samples = 0
        for key, count in counts.items():
            parity = key.array.count(1)
            sign = (-1) ** parity
            E += sign * count
            n_samples += count
        assert n_samples == samples
        E = E / samples * paulistring.coeff
        return E


class BackendExpectationValueDDSim(BackendExpectationValue):
    """
    Class representing Expectation Values compiled for DDSim.
    """

    BackendCircuitType = BackendCircuitDDSim
