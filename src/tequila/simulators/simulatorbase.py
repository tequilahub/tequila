from tequila import TequilaException, BitNumbering
from tequila.circuit.circuit import QCircuit
from tequila.utils.keymap import KeyMapSubregisterToRegister
from tequila.utils.misc import to_float
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila.circuit.compiler import change_basis
from tequila.circuit.gates import Measurement
from tequila.circuit.gates import RotationGateImpl, PowerGateImpl, ExponentialPauliGateImpl
from tequila import BitString
from tequila.objective import Objective
from tequila.objective.objective import Variable, assign_variable
from tequila.simulators.heralding import HeraldingABC
from tequila.circuit import compiler
from tequila.circuit._gates_impl import MeasurementImpl

import numpy, numbers, typing, copy
from dataclasses import dataclass


@dataclass
class SimulatorReturnType:
    abstract_circuit: QCircuit = None
    circuit: int = None
    wavefunction: QubitWaveFunction = None
    measurements: typing.Dict[str, QubitWaveFunction] = None
    backend_result: int = None

    @property
    def counts(self, key: str = None):
        if self.measurements is not None:
            if key is None:
                keys = [k for k in self.measurements.keys()]
                return self.measurements[keys[0]]
            else:
                return self.measurements[key]
        elif self.wavefunction is not None:
            measurement = copy.deepcopy(self.wavefunction)
            for k, v in measurement.items():
                measurement[k] = numpy.abs(v) ** 2
            return measurement


class BackendCircuit:
    """
    Functions in the end need to be overwritten by specific backend implementation
    Other functions can be overwritten to improve performance
    self.circuit : translated circuit
    self.abstract_circuit: compiled tequila circuit
    """

    # compiler instructions
    recompile_trotter = True
    recompile_swap = False
    recompile_multitarget = True
    recompile_controlled_rotation = False
    recompile_exponential_pauli = True

    def __init__(self, abstract_circuit: QCircuit, variables):

        # compile the abstract_circuit
        c = compiler.Compiler(multitarget=self.recompile_multitarget,
                              multicontrol=False,
                              trotterized=self.recompile_trotter,
                              exponential_pauli=self.recompile_exponential_pauli,
                              controlled_exponential_pauli=True,
                              controlled_rotation=self.recompile_controlled_rotation,
                              swap=self.recompile_swap)

        self.qubit_map = self.make_qubit_map(abstract_circuit=abstract_circuit)
        self.qubits = abstract_circuit.qubits
        self.n_qubits = abstract_circuit.n_qubits
        compiled = c(abstract_circuit)
        self.abstract_circuit = compiled
        # translate into the backend object
        self.circuit = self.create_circuit(abstract_circuit=compiled, variables=variables)

    def create_circuit(self, abstract_circuit: QCircuit, variables: typing.Dict[typing.Hashable, numbers.Real] = None):
        """
        Translates abstract circuits into the specific backend type
        :param abstract_circuit: Abstract circuit to be translated
        :return: translated circuit
        """

        if self.fast_return(abstract_circuit):
            return abstract_circuit

        result = self.initialize_circuit()

        for g in abstract_circuit.gates:
            if isinstance(g, MeasurementImpl):
                self.add_measurement(gate=g, circuit=result)
            elif g.is_controlled():
                if isinstance(g, RotationGateImpl):
                    self.add_controlled_rotation_gate(gate=g, variables=variables, circuit=result)
                elif isinstance(g, PowerGateImpl) and g.power != 1:
                    self.add_controlled_power_gate(gate=g, variables=variables, qubit_map=qubit_map, circuit=result)
                else:
                    self.add_controlled_gate(gate=g, circuit=result)
            else:
                if isinstance(g, RotationGateImpl):
                    self.add_rotation_gate(gate=g, variables=variables, circuit=result)
                elif isinstance(g, PowerGateImpl) and g.power != 1:
                    self.add_power_gate(gate=g, variables=variables, circuit=result)
                elif isinstance(g, ExponentialPauliGateImpl):
                    self.add_exponential_pauli_gate(gate=g, variables=variables, circuit=result)
                else:
                    self.add_gate(gate=g, circuit=result)

        return result

    def update_variables(self, variables):
        """
        This is the default which just translates the circuit again
        Overwrite in backend if parametrized circuits are supported
        """
        self.circuit = self.create_circuit(abstract_circuit=self.abstract_circuit, variables=variables)

    def simulate(self, variables, initial_state=0) -> QubitWaveFunction:
        """
        Simulate the wavefunction
        :param returntype: specifies how the result should be given back
        :param initial_state: The initial state of the simulation,
        if given as an integer this is interpreted as the corresponding multi-qubit basis state
        :return: The resulting state
        """

        if isinstance(initial_state, BitString):
            initial_state = initial_state.integer
        if isinstance(initial_state, QubitWaveFunction):
            if len(initial_state.keys()) != 1:
                raise TequilaException("only product states as initial states accepted")
            initial_state = list(initial_state.keys())[0].integer

        active_qubits = self.abstract_circuit.qubits
        all_qubits = [i for i in range(self.abstract_circuit.n_qubits)]

        # maps from reduced register to full register
        keymap = KeyMapSubregisterToRegister(subregister=active_qubits, register=all_qubits)

        result = self.do_simulate(variables=variables, initial_state=keymap.inverted(initial_state).integer)
        result.apply_keymap(keymap=keymap, initial_state=initial_state)
        return result

    # Those functions need to be overwritten:

    def do_simulate(self, variables, initial_state):
        TequilaException("Backend Handler needs to be overwritten for supported simulators")

    def fast_return(self, abstract_circuit):
        return True

    def add_exponential_pauli_gate(self, gate, circuit, variables, *args, **kwargs):
        TequilaException("Backend Handler needs to be overwritten for supported simulators")

    def add_controlled_gate(self, gate, circuit, *args, **kwargs):
        return self.add_gate(gate, circuit, args, kwargs)

    def add_rotation_gate(self, gate, circuit, variables, *args, **kwargs):
        return self.add_gate(gate, circuit, args, kwargs)

    def add_controlled_rotation_gate(self, gate, variables, circuit, *args, **kwargs):
        return self.add_gate(gate, circuit, args, kwargs)

    def add_power_gate(self, gate, variables, circuit, *args, **kwargs):
        return self.add_gate(gate, circuit, args, kwargs)

    def add_controlled_power_gate(self, gate, variables, circuit, *args, **kwargs):
        return self.add_gate(gate, circuit, args, kwargs)

    def initialize_circuit(self, *args, **kwargs):
        TequilaException("Backend Handler needs to be overwritten for supported simulators")

    def add_gate(self, gate, circuit, *args, **kwargs):
        TequilaException("Backend Handler needs to be overwritten for supported simulators")

    def add_measurement(self, gate, circuit, *args, **kwargs):
        TequilaException("Backend Handler needs to be overwritten for supported simulators")

    def make_qubit_map(self, abstract_circuit: QCircuit):
        return [i for i in range(len(abstract_circuit.qubits))]


class BackendExpectationValue:
    BackendCircuitType = BackendCircuit

    @property
    def H(self):
        return self._H

    @property
    def U(self):
        return self._U

    def __init__(self, E, variables):
        self._H = self.initialize_hamiltonian(E.H)
        self._U = self.initialize_unitary(E.U, variables)

    def initialize_hamiltonian(self, H):
        return H

    def initialize_unitary(self, U, variables):
        return self.BackendCircuitType(abstract_circuit=U, variables=variables)

    def update_variables(self, variables):
        self._U.update_variables(variables=variables)

    def simulate(self, variables):
        self.update_variables(variables)
        final_E = 0.0
        # The hamiltonian can be defined on more qubits as the unitaries
        qubits_h = self.H.qubits
        qubits_u = self.U.qubits
        all_qubits = list(set(qubits_h) | set(qubits_u))
        keymap = KeyMapSubregisterToRegister(subregister=qubits_u, register=all_qubits)
        simresult = self.simulate_wavefunction(abstract_circuit=self.U, variables=variables)
        wfn = simresult.wavefunction.apply_keymap(keymap=keymap)
        # TODO inefficient, let the backend do it if possible or interface some library
        final_E += wfn.compute_expectationvalue(operator=self.H)

        return to_float(final_E)


class SimulatorBase:
    """
    Abstract Base Class for OpenVQE interfaces to simulators
    """
    numbering: BitNumbering = BitNumbering.MSB
    backend_hanlder = BackendCircuit

    def __init__(self, heralding: HeraldingABC = None, ):
        self._heralding = heralding
        self.__do_recompilation = True

    def __call__(self, objective: typing.Union[QCircuit, Objective],
                 variables: typing.Dict[typing.Hashable, numbers.Real] = None, samples: int = None, *args,
                 **kwargs) -> numbers.Real:
        """
        :param objective: Objective or simple QCircuit
        :param samples: Number of Samples to evaluate, None means full wavefunction simulation
        :param kwargs: keyword arguments to further pass down
        :return: Energy, or simulator return type depending on what was passed down as objective
        """

        # user convenience
        formatted_variables = variables
        if formatted_variables is not None:
            formatted_variables = {assign_variable(k): v for k, v in variables.items()}

        if isinstance(objective, QCircuit):
            if samples is None:
                return self.simulate_wavefunction(abstract_circuit=objective, variables=formatted_variables, *args,
                                                  **kwargs).wavefunction
            else:
                return self.run(abstract_circuit=objective, samples=samples, variables=formatted_variables, *args,
                                **kwargs).wavefunction
        else:
            if samples is None:
                return to_float(
                    self.simulate_objective(objective=objective, variables=formatted_variables, *args, **kwargs))
            else:
                return to_float(
                    self.measure_objective(objective=objective, samples=samples, variables=formatted_variables, *args,
                                           **kwargs))

    def run(self, abstract_circuit: QCircuit, variables: typing.Dict[typing.Hashable, numbers.Real] = None,
            samples: int = 1) -> SimulatorReturnType:
        circuit = self.create_circuit(abstract_circuit=abstract_circuit, variables=variables).circuit
        backend_result = self.do_run(circuit=circuit, samples=samples)
        return SimulatorReturnType(circuit=circuit,
                                   abstract_circuit=abstract_circuit,
                                   backend_result=backend_result,
                                   measurements=self.postprocessing(self.convert_measurements(backend_result)))

    def do_run(self, circuit, samples: int = 1):
        raise TequilaException("do_run needs to be overwritten by corresponding backend")

    def set_compile_flag(self, b):
        self.__do_recompilation = b

    def simulate_wavefunction(self, abstract_circuit: QCircuit,
                              variables: typing.Dict[typing.Hashable, numbers.Real] = None,
                              initial_state: int = 0) -> SimulatorReturnType:
        """
        Simulates an abstract circuit with the backend specified by specializations of this class
        :param abstract_circuit: The abstract circuit
        :param returntype: specifies how the result should be given back
        :param initial_state: The initial state of the simulation,
        if given as an integer this is interpreted as the corresponding multi-qubit basis state
        :return: The resulting state
        """

        if isinstance(initial_state, BitString):
            initial_state = initial_state.integer
        if isinstance(initial_state, QubitWaveFunction):
            if len(initial_state.keys()) != 1:
                raise TequilaException("only product states as initial states accepted")
            initial_state = list(initial_state.keys())[0].integer

        active_qubits = abstract_circuit.qubits
        all_qubits = [i for i in range(abstract_circuit.n_qubits)]

        # maps from reduced register to full register
        keymap = KeyMapSubregisterToRegister(subregister=active_qubits, register=all_qubits)

        result = self.do_simulate_wavefunction(abstract_circuit=abstract_circuit, variables=variables,
                                               initial_state=keymap.inverted(initial_state).integer)
        result.wavefunction.apply_keymap(keymap=keymap, initial_state=initial_state)
        return result

    def do_simulate_wavefunction(self, circuit, variables, initial_state=0) -> SimulatorReturnType:
        raise TequilaException(
            "called from base class of simulators, or non-supported operation for this backend")

    def create_circuit(self, abstract_circuit: QCircuit, variables: typing.Dict[typing.Hashable, numbers.Real] = None):
        """
        Translates abstract circuits into the specific backend type
        Overwrite the BackendHandler Class to implement new backends
        :param abstract_circuit: Abstract circuit to be translated
        :return: translated circuit
        """
        return self.backend_handler(abstract_circuit=abstract_circuit, variables=variables)

    def convert_measurements(self, backend_result) -> typing.Dict[str, QubitWaveFunction]:
        raise TequilaException(
            "called from base class of simulators, or non-supported operation for this backend")

    def measure_expectationvalue(self, E, variables, samples: int,
                                 return_simulation_data: bool = False) -> numbers.Real:
        H = E.H
        U = E.U
        # The hamiltonian can be defined on more qubits as the unitaries
        result_data = {}
        final_E = 0.0
        for ps in H.paulistrings:
            Etmp, tmp = self.measure_paulistring(abstract_circuit=U, variables=variables, paulistring=ps,
                                                 samples=samples)
            final_E += Etmp
            result_data[str(ps)] = tmp

        # type conversion to not confuse optimizers
        if hasattr(final_E, "imag"):
            assert (numpy.isclose(final_E.imag, 0.0))
            final_E = float(final_E.real)

        if return_simulation_data:
            return float(final_E), result_data
        else:
            return float(final_E)

    def measure_objective(self, objective, samples: int, variables=None, return_simulation_data: bool = False) -> float:
        elist = []
        data = []
        for ex in objective.args:
            if hasattr(ex, 'U'):
                result_data = {}
                evalue = 0.0
                for ps in ex.H.paulistrings:
                    Etmp, tmp = self.measure_paulistring(abstract_circuit=ex.U, variables=variables, paulistring=ps,
                                                         samples=samples)
                    evalue += Etmp
                    result_data[str(ps)] = tmp
                elist.append(evalue)
                if return_simulation_data:
                    data.append(tmp)
            elif hasattr(ex, 'name'):
                elist.append(ex(variables))

        # in principle complex weights are allowed, but it probably will never occur
        # however, for now here is the type conversion to not confuse optimizers
        final_E = objective.transformation(*elist)
        if hasattr(final_E, "imag") and numpy.isclose(final_E.imag, 0.0):
            final_E = float(final_E.real)

        if return_simulation_data:
            return float(final_E), data
        else:
            return float(final_E)

    def simulate_expectationvalue(self, E,
                                  variables: typing.Dict[typing.Hashable, numbers.Real] = None) -> numbers.Real:
        final_E = 0.0
        data = []
        H = E.H
        U = E.U
        # The hamiltonian can be defined on more qubits as the unitaries
        qubits_h = H.qubits
        qubits_u = U.qubits
        all_qubits = list(set(qubits_h) | set(qubits_u))
        keymap = KeyMapSubregisterToRegister(subregister=qubits_u, register=all_qubits)
        simresult = self.simulate_wavefunction(abstract_circuit=U, variables=variables)
        wfn = simresult.wavefunction.apply_keymap(keymap=keymap)
        # TODO inefficient, let the backend do it if possible or interface some library
        final_E += wfn.compute_expectationvalue(operator=H)

        return to_float(final_E)

    def simulate_objective(self, objective: Objective,
                           variables: typing.Dict[typing.Hashable, numbers.Real] = None) -> numbers.Real:
        # simulate all expectation values
        # TODO easy to parallelize
        E = []
        for Ei in objective.args:
            if hasattr(Ei, "U") and hasattr(Ei, "H"):
                E.append(self.simulate_expectationvalue(E=Ei, variables=variables))
            else:
                E.append(Ei(variables=variables))
        # return evaluated result
        return objective.transformation(*E)

    def measure_paulistring(self, abstract_circuit: QCircuit, variables: typing.Dict[typing.Hashable, numbers.Real],
                            paulistring, samples: int = 1):
        # make basis change
        basis_change = QCircuit()
        for idx, p in paulistring.items():
            basis_change += change_basis(target=idx, axis=p)
        # make measurement instruction
        measure = QCircuit()
        qubits = [idx[0] for idx in paulistring.items()]
        if len(qubits) == 0:
            # no measurement instructions for a constant term as paulistring
            return (paulistring.coeff, SimulatorReturnType())
        else:
            measure += Measurement(name=str(paulistring), target=qubits)
            circuit = abstract_circuit + basis_change + measure
            # run simulators
            sim_result = self.run(abstract_circuit=circuit, variables=variables, samples=samples)

            # compute energy
            counts = sim_result.counts
            E = 0.0
            n_samples = 0
            for key, count in counts.items():
                parity = key.array.count(1)
                sign = (-1) ** parity
                E += sign * count
                n_samples += count
            if self._heralding is None:
                assert (n_samples == samples)  # failsafe
            E = E / samples * paulistring.coeff
            return (E, sim_result)

    def postprocessing(self, measurements: typing.Dict[str, QubitWaveFunction]) -> typing.Dict[str, QubitWaveFunction]:
        # fast return
        if self._heralding is None:
            return measurements

        result = dict()
        for k, v in measurements.items():
            result[k] = self._heralding(input=v)
        return result

    def draw(self, objective: typing.Union[QCircuit, Objective], *args, **kwargs):

        if isinstance(objective, Objective):
            self.draw_objective(objective=objective, *args, **kwargs)
        else:
            self.draw_circuit(abstract_circuit=objective, *args, **kwargs)

    def draw_objective(self, objective: Objective, *args, **kwargs):
        """
        Default drawing is just the printout of the stringification ... feel free to add a nice representation
        """

        transform = objective.transformation
        objective_string = "f("
        exp_ind = []
        for i, arg in enumerate(objective._args):
            if isinstance(arg, Variable):
                objective_string += str(arg) + ", "
            elif hasattr(arg, "U"):
                objective_string += "E_" + str(i) + ", "
                exp_ind.append(i)
        objective_string += ")"

        print("Tequila Objective:")
        print("O = ", objective_string)
        print("f = ", transform)

        for i in exp_ind:
            print("E_", i, " expectation value with:")
            print("H = ", objective._args[i].H, ", U=")
            self.draw_circuit(abstract_circuit=objective._args[i].U)

    def draw_circuit(self, abstract_circuit: QCircuit, *args, **kwargs):
        """
        Default drawing is just the printout of the stringification ... feel free to add a nice representation
        """
        print(abstract_circuit)
