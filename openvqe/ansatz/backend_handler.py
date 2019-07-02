"""
Backend Handler:
handles the construction of quantum circuits for different simulation backends
"""

from openvqe.exceptions import OVQEException



class BackendHandlerABC:
    # gates which can not be handled in the standard way
    exceptions = []
    backend = None

    def __init__(self, n_qubits, qubits=None):
        self.n_qubits = n_qubits
        try:
            self.import_backend()
        except ImportError as e:
            print(e)
            raise OVQEException("failed to import backed for " + type(self).__name__,
                                " make sure it is available in the current environment")
        self.qubits = self.initialize_qubits(n_qubits, qubits)


    def import_backend(self):
        """
        Overwrite this in specializations
        """
        pass

    def __call__(self, name, targets: list, controls: list = None, angle=None):
        gate = None
        c = None

        if controls is not None:
            c = [self.qubits[i] for i in controls]

        t = [self.qubits[i] for i in targets]

        if name in self.exceptions:
            if controls is None:
                if angle is None:
                    gate = getattr(self, name)(targets=t)
                else:
                    gate = getattr(self, name)(targets=t, angle=angle)
            else:
                if angle is None:
                    gate = getattr(self, name)(targets=t, controls=c)
                else:
                    gate = getattr(self, name)(targets=t, controls=c, angle=angle)
        elif angle is None:
            gate = self.non_parametrized_gate(name=name, targets=t, controls=c)
        else:
            gate = self.parametrized_gate(name=name, targets=t, controls=c, angle=angle)

        return self.wrap_gate(gate)

    def wrap_gate(self, gate):
        """
        Sometimes we need to wrap the gate objects into other objects in order to
        use the += operator
        The default here does nothing
        """
        return gate

    def initialize_qubits(self, n_qubits: int, qubits=None):
        """
        Define the Qubit Map, by default just a string topology, can be overwritten
        :param qubits: the qubit map
        :return:
        """
        if qubits is None:
            return [i for i in range(n_qubits)]
        else:
            return qubits

    def init_circuit(self):
        """
        Initialize an circuit object
        :return: Empty Circuit for Backend
        """
        raise OVQEException("Do not call functions BackendHandlerABC directly")

    def non_parametrized_gate(self, name, targets, controls):
        """
        :param name: Name of the Gate
        :param targets: Target qubits (will be translated as self.qubits[target]
        :param controls: control qubits (will be translated as self.qubits[target]
        :return: unparametrized one_qubit gates in the backend format
        """
        raise OVQEException("Do not call functions BackendHandlerABC directly")

    def parametrized_gate(self, name, targets, controls, angle):
        """
        :param name: Name of the Gate
        :param targets: Target qubits (will be translated as self.qubits[target]
        :param controls: control qubits (will be translated as self.qubits[target]
        :param angle: angle or exponent which parametrizes the gate
        :return: unparametrized one_qubit gates in the backend format
        """
        raise OVQEException("Do not call functions BackendHandlerABC directly")

    def verify(self):
        raise OVQEException("Do not call functions BackendHandlerABC directly")


class BackendHandlerCirq(BackendHandlerABC):
    exceptions = ["I", "CNOT"]

    def I(self, targets, controls=None, angle=None):
        """
        Exception necessary since cirq does not have an identity gate on specific qubits
        """
        gate=self.backend.Z ** 0.0
        return gate.on(targets[0])

    def CNOT(self, targets: list, controls: list):
        assert(len(targets)==1)
        assert(len(controls)==1)
        return self.backend.CNOT(target=targets[0], control=controls[0])

    def initialize_qubits(self, n_qubits: int, qubits=None):
        """
        Initialize cirq qubits on a line (default) or as given by qubits
        :param n_qubits: number of qubits
        :param qubits: cirq qubits to use
        :return: the given cirq qubits or n_qubits on a line
        """
        if qubits is None:
            return [self.backend.GridQubit(i, 0) for i in range(n_qubits)]
        else:
            assert(isinstance(qubits, cirq.GridQubit))
            return qubits

    def init_circuit(self):
        return self.backend.Circuit()

    def non_parametrized_gate(self, name, targets, controls):
        gate = getattr(self.backend, name)
        if controls is not None:
            gate.controlled_by(controls)
        return gate.on(targets[0])

    def parametrized_gate(self, name, targets, controls, angle):
        gate = getattr(self.backend, name)(angle)
        if controls is not None:
            gate.controlled_by(controls[0])
        return gate.on(targets[0])

    def import_backend(self):
        import cirq
        self.backend = cirq

    def wrap_gate(self, gate):
        c = self.init_circuit()
        c.append(gate)
        return c

    def test_import(self):
        if self.backend is None:
            print(type(self).__name__, " backend import failed, testing again ...")
            try:
                import cirq as test_cirq
            except ImportError:
                raise OVQEException(
                    "Could not import cirq, make sure the module is available in the current environment")


def get_backend_hander(backend: str, n_qubits: int, qubits=None):
    if backend == "cirq":
        return BackendHandlerCirq(n_qubits=n_qubits, qubits=qubits)
    else:
        raise NotImplementedError("Unknown backend:", str(backend))
