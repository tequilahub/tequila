import qtensor, qtree
import numbers, numpy
from tequila import TequilaException
from tequila.utils.bitstrings import BitNumbering, BitString, BitStringLSB
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila.simulators.simulator_base import BackendCircuit, BackendExpectationValue, QCircuit, change_basis
from tequila.utils.keymap import KeyMapRegisterToSubregister


class TequilaQtensorException(TequilaException):
    def __str__(self):
        return "Error in qtensor backend:" + self.message


class BackendCircuitQtensor(BackendCircuit):
    compiler_arguments = {
        "trotterized": True,
        "swap": True,
        "multitarget": True,
        "controlled_rotation": True,  # needed for gates depending on variables
        "generalized_rotation": True,
        "exponential_pauli": True,
        "controlled_exponential_pauli": True,
        "phase": True,
        "power": True,
        "hadamard_power": True,
        "controlled_power": True,
        "controlled_phase": True,
        "toffoli": True,
        "phase_to_z": True,
        "cc_max": True
    }

    def __init__(self, abstract_circuit, noise=None, *args, **kwargs):
        """

        Parameters
        ----------
        abstract_circuit: QCircuit:
            the circuit to compile to qulacs
        noise: optional:
            noise to apply to the circuit.
        args
        kwargs
        """
        self.op_lookup = {
            'I': None,
            'X': qtree.operators.X,
            'Y': qtree.operators.Y,
            'Z': qtree.operators.Z,
            'H': qtree.operators.H,
            'Rx': qtree.operators.rx,
            'Ry': qtree.operators.ry,
            'Rz': qtree.operators.rz,
            'Measure': qtree.operators.M
        }
        self.measurements = None
        self.variables = []
        super().__init__(abstract_circuit=abstract_circuit, noise=noise, *args, **kwargs)
        self.has_noise = False
        if noise is not None:
            raise TequilaQtensorException("Noisy simulation not supported in Qtensor backend!")

    def initialize_circuit(self, *args, **kwargs):
        return []

    def do_sample(self, samples, circuit, noise, abstract_qubits=None, *args, **kwargs) -> QubitWaveFunction:
        raise TequilaQtensorException("Sampling not supported in Qtensor backend!")

    def do_simulate(self, variables, initial_state, *args, **kwargs) -> QubitWaveFunction:
        raise TequilaQtensorException("Wavefunction simulation not supported in Qtensor interface!")

    def add_basic_gate(self, gate, circuit, *args, **kwargs):
        assert(len(gate.target)==1)
        target = gate.target[0]
        if gate.is_controlled():
            failed = False
            if gate.name == "X":
                if len(gate.control) == 1:
                    qtree_gate = qtree.operators.cX(target, gate.control[0])
                elif len(gate.control) == 2:
                    qtree_gate = qtree.operators.cX(target, gate.control[0], gate.control[1])
                else:
                    failed = True

            elif gate.name == "Y" and len(control) == 1:
                qtree_gate = qtree.operators.cY(*targets)
            elif gate.name == "Z" and len(control) == 1:
                qtree_gate = qtree.operators.cZ(*targets)
            else:
                failed = True

            if failed:
                raise TequilaQtensorException(
                    "Multi-Controlled, except CCX, gates not yet supported in Qtensor backend! Can be circumvented by compiling")
        else:
            op = self.op_lookup[gate.name]
            qtree_gate = op(target)

        circuit.append(qtree_gate)

    def add_parametrized_gate(self, gate, circuit, *args, **kwargs):
        if gate.is_controlled() or gate.name not in ["Rx", "Ry", "Rz"]:
            raise TequilaQtensorException("Only non-controlled rotations are supported as parametrized gates! Received gate={}".format(gate))
        assert len(gate.target) == 1
        assert "variables" in kwargs

        op = self.op_lookup[gate.name]
        angle = gate.parameter(kwargs["variables"])
        qtree_gate = op([angle], gate.target[0])
        circuit.append(qtree_gate)



class BackendExpectationValueQtensor(BackendExpectationValue):
    use_mapping = True
    BackendCircuitType = BackendCircuitQtensor

    def do_sample(self, variables, samples, *args, **kwargs) -> numpy.array:
        raise TequilaQtensorException("Sampling not supported in QTensor backend")

    def simulate(self, variables, *args, **kwargs):
        self.update_variables(variables)
        Hv = self.H
        import copy
        Ud = [copy.copy(gate).dagger_me() for gate in reversed(self.U.circuit)]
        results = []
        for H in Hv:
            result = 0.0
            for ps in H.paulistrings:
                contracted = self.contract_paulistring(paulistring=ps, ket=self.U.circuit, bra=Ud)
                result += ps.coeff * contracted
            results.append(result)
        return results

    def contract_paulistring(self, paulistring, ket=None, bra=None):

        pauli_unitary = [self.U.op_lookup[name.upper()](target) for target, name in paulistring.items()]
        if ket is None:
            ket = self.U.circuit
        if bra is None:
            bra = [gate.dagger() for gate in reversed(U)]

        sim = qtensor.QtreeSimulator()
        operators = ket + pauli_unitary + bra
        print(operators)
        result = sim.simulate(operators)
        return result
