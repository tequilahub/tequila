"""
Generate a circuit which can generate any state similar to a unary representation
i.e number of basis states == number of qubits

The whole thing is currently not very stable
.... proceed with caution
"""

from openvqe.circuit import QCircuit
from openvqe import BitString
from openvqe import typing, numpy, copy, OpenVQEException
from openvqe.apps._unary_state_prep_impl import UnaryStatePrepImpl, sympy
from openvqe.simulator.simulator_symbolic import SimulatorSymbolic
from openvqe.ansatz import AnsatzBase


class UnaryStatePrep(AnsatzBase):

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def target_space(self) -> typing.List[BitString]:
        return self._target_space

    @property
    def circuit(self) -> QCircuit:
        return self._abstract_circuit

    def __init__(self, target_space: typing.List[BitString], max_repeat: int = 20, use_symbolic_solution: bool = False):
        """
        :param target_space: Give the target space by its basis functions
        e.g. target_space = ['00', '11']
        :param use_symbolic_solution: Solve for angles purely symbolic (fails a lot of times due to sign errors)
        the sympy interface is probably not used correctly. If False numerical solvers are used which work better
        :param max_repeat: maximum numbers of repetitions for numerical solver
        the target space can be given as list of strings (binaries) or BitString objects
        """

        self._use_symbolic_solver = use_symbolic_solution
        self.max_repeat = max_repeat

        if isinstance(target_space[0], str):
            target_space = [BitString.from_binary(binary=i) for i in target_space]

        IMPL = UnaryStatePrepImpl()

        self._n_qubits = target_space[0].nbits

        abstract_coefficients = [sympy.var("c" + str(i)) for i in range(len(target_space))]
        abstract_angles = [sympy.var(IMPL.alphabets[i]) for i in range(len(target_space) - 1)]

        # code is currenty unstable but can still succeed if input is shuffled
        # todo: that should not happen, but untill it is fixed I will increase success probability
        count = 0
        success = False
        # todo need to change input format that reshuffling does not destroy it
        # setting count to 1 to avoid it
        while(not success and count <1):
            try:
                count += 1
                self._abstract_circuit = IMPL.get_circuit(s=[i.binary for i in target_space])
                success = True
            except:
                packed = list(zip(target_space, abstract_angles, abstract_coefficients))
                numpy.random.shuffle(packed)
                target_space, abstract_angles, abstract_coefficients = zip(*packed)

        if not success: raise OpenVQEException("Could not disentangle the given state after " + str(count) + " restarts")

        # get the equations to determine the angles
        simulator = SimulatorSymbolic()
        wfn = simulator.simulate_wavefunction(abstract_circuit=self._abstract_circuit,
                                              initial_state=BitString.from_int(0, nbits=self.n_qubits)).wavefunction

        print("wfn=", wfn)
        print(self._abstract_circuit)
        equations = []
        for i, k in enumerate(target_space):
            equations.append(wfn[k] - abstract_coefficients[i])

        normeq = -sympy.Integer(1)
        for c in abstract_coefficients:
            normeq += c ** 2
        equations.append(normeq)

        # this gives back multiple solutions and some of them sometimes have sign errors ...
        if (self._use_symbolic_solver):
            solutions = sympy.solve(equations, *tuple(abstract_angles), check=True, dict=True)[1]
            if len(abstract_angles) != len(solutions):
                raise OpenVQEException("Could definetely not solve for the angles in UnaryStatePrep!")
            self._angle_solutions = solutions

        self._target_space = target_space
        self._abstract_angles = abstract_angles
        self._equations = equations
        self._abstract_coeffs = abstract_coefficients

        if self._n_qubits < len(self._target_space):
            print("WARNING: UnaryStatePrep initialized with more basis states than qubits -> that might not work")

    def __repr__(self):
        result = "UnaryStatePrep:\n"
        result += "angles         : " + str(self._abstract_angles) + "\n"
        result += "coeff to angles:\n"
        if self._use_symbolic_solver:
            result += str(self._angle_solutions) + "\n"
        result += "abstract circuit:\n"
        result += str(self._abstract_circuit)
        return result

    def evaluate_angles(self, coeffs: list) -> typing.Dict[sympy.Symbol, float]:
        # coeffs need to be normalized
        coeffs = numpy.asarray(coeffs, dtype=numpy.float)
        norm2 = numpy.linalg.norm(coeffs)
        coeffs *= 1.0 / norm2

        # initialize as sympy floats
        c = [sympy.Float(coeff) for coeff in coeffs]

        subs = dict()
        for i, ac in enumerate(self._abstract_coeffs):
            subs[ac] = c[i]

        result = dict()
        if (self._use_symbolic_solver):
            # fails a lot of times
            # better don't use
            # get the angle variables from the symbolic equations for the angles
            for symbol, eq in self._angle_equations.items():
                result[symbol] = float(eq.evalf(subs=subs))
        else:
            count = 0
            solutions = []
            while (count < self.max_repeat and len(solutions) == 0):
                try:
                    # numeric solution
                    subsx = [(ac, c[i]) for i, ac in enumerate(self._abstract_coeffs)]
                    equations = [eq.subs(subsx) for eq in self._equations]
                    guess = numpy.random.uniform(0.1, 0.9 * 2 * numpy.pi, len(self._abstract_angles))
                    solutions = sympy.nsolve(equations, self._abstract_angles, guess)
                    count += 1
                except:
                    count += 1
            if (len(solutions) == 0):
                raise Exception("Failed to numerically solve for angles")

            for i, symbol in enumerate(self._abstract_angles):
                result[symbol] = float(solutions[i].evalf(subs=subs))

        return result

    def __call__(self, coeffs: list, guess:  list = None) -> QCircuit:
        """
        :param coeffs: give a dictionary of the coeffs where the key is the corresponding basis state in target_space
        :param guess: list of guess values for the angles, if not given a random guess is initialized
        :return:
        """
        assert (len(coeffs) == len(self._target_space))
        angles = self.evaluate_angles(coeffs=coeffs)

        # construct new circuit with evaluated angles
        result = QCircuit()
        for g in self._abstract_circuit:
            g2 = copy.deepcopy(g)
            if hasattr(g, "angle"):
                symbol = g.angle
                g2.angle = -angles[
                    -symbol]  # the minus follows mahas convention since the circuits are daggered in the end
            result += g2

        return result


if __name__ == "__main__":
    # test and play around
    USP = UnaryStatePrep(target_space=['001', '010', '100'])
    print(USP)

    print("angles=", USP.evaluate_angles(coeffs=[1, 1, 1]))
    # print(U.circuit)
    U = USP(coeffs=[1, 1, 1])

    print(SimulatorSymbolic().simulate_wavefunction(abstract_circuit=U).wavefunction)
    print(SimulatorCirq().simulate_wavefunction(abstract_circuit=U).wavefunction)
    # print(SimulatorPyquil().simulate_wavefunction(abstract_circuit=U).wavefunction)
