from openvqe.simulator.simulator import Simulator, QCircuit, SimulatorReturnType, MeasurementResultType
from openvqe import OpenVQEException
from openvqe.circuit.compiler import compile_multitarget, compile_controlled_rotation_gate
from openvqe.circuit.gates import MeasurementImpl
import qiskit


class OpenVQEQiskitException(OpenVQEException):
    def __str__(self):
        return "Error in qiskit backend:" + self.message


class SimulatorQiskit(Simulator):

    def create_circuit(self, abstract_circuit: QCircuit, name="q", cname="c") -> qiskit.QuantumCircuit:

        n_qubits = abstract_circuit.n_qubits
        q = qiskit.QuantumRegister(n_qubits, name)
        c = qiskit.ClassicalRegister(n_qubits, cname)
        result = qiskit.QuantumCircuit(q, c)
        result.rx

        for g in abstract_circuit.gates:

            if isinstance(g, MeasurementImpl):
                for t in g.target:
                    result.measure(q[t], c[t])
                continue

            if g.control is not None and len(g.control) > 2:
                # do that automatically at some point
                # same as below with multitarget
                raise OpenVQEQiskitException("More than two controls not supported by Qiskit -> Recompile")

            if g.control is not None and g.name.lower() in ["rx", "ry"]:
                # recompile controled rotations, this is what qiskit does anyway, but it somehow has no Rx
                result.barrier(q)  # better visibility for recompilation
                result += self.create_circuit(abstract_circuit=compile_controlled_rotation_gate(gate=g))
                result.barrier(q)  # better visibility for recompilation
                continue

            if len(g.target) > 1:
                # multi targets need to be explicitly recompiled for Qiskit
                result.barrier(q)  # better visibility for recompilation
                result += self.create_circuit(abstract_circuit=compile_multitarget(gate=g))
                result.barrier(q)  # better visibility for recompilation
                continue

            assert (len(g.target) == 1)
            if g.control is None or len(g.control) == 0:
                gfunc = getattr(result, g.name.lower())
                if g.is_parametrized():
                    gfunc(g.parameter, q[g.target[0]])
                else:
                    gfunc(q[g.target[0]])
            elif len(g.control) == 1:
                gfunc = getattr(result, "c" + g.name.lower())
                if g.is_parametrized():
                    gfunc(g.parameter, q[g.control[0]], q[g.target[0]])
                else:
                    gfunc(q[g.control[0]], q[g.target[0]])
            elif len(g.control) == 1:
                gfunc = getattr(result, "cc" + g.name.lower())
                if g.is_parametrized():
                    gfunc(g.parameter, q[g.control[0]], q[g.control[1]], q[g.target[0]])
                else:
                    gfunc(q[g.control[0]], q[g.control[1]], q[g.target[0]])
            else:
                raise OpenVQEQiskitException("Not supported")

        return result


if __name__ == "__main__":

    from openvqe.circuit import gates

    simulator = SimulatorQiskit()
    testc = [gates.X(target=0), gates.X(target=1), gates.X(target=1, control=0),
             gates.Rx(target=1, control=0, angle=2.0), gates.Ry(target=1, control=0, angle=2.0),
             gates.Rz(target=1, control=0, angle=2.0)]

    for c in testc:
        c_qiskit = simulator.create_circuit(abstract_circuit=c)
        print(c_qiskit.draw())
