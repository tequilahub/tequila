"""
Generate a circuit which can generate any state similar to a unary representation
i.e number of basis states == number of qubits

The whole thing is currently not very stable
.... proceed with caution
"""

from tequila.circuit import QCircuit
from tequila import BitString
import typing, numpy, copy
from tequila import TequilaException
from tequila.apps._unary_state_prep_impl import UnaryStatePrepImpl, sympy
from tequila.simulators.simulator_symbolic import BackendCircuitSymbolic
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila.objective.objective import assign_variable


class TequilaUnaryStateException(TequilaException):
    """
    Exception will be thrown when UnaryStatePrep Fails (which can happen for some instances)
    Those are the cases which we currently can not prevent
    Failed assertions which should not happen in all cases will throw a regular OpenVQEException
    Class is just there to skip tests
    """

    pass

class UnaryStatePrep:

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def target_space(self) -> typing.List[BitString]:
        return self._target_space

    @property
    def circuit(self) -> QCircuit:
        return self._abstract_circuit

    def __init__(self, target_space: typing.List[BitString], max_repeat: int = 100,
                 use_symbolic_solution: bool = False):
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

        if isinstance(target_space, QubitWaveFunction):
            target_space = [f for f in target_space.keys()]
        if isinstance(target_space[0], str):
            target_space = [BitString.from_binary(binary=i) for i in target_space]

        self._n_qubits = target_space[0].nbits

        abstract_coefficients = dict()
        for bf in target_space:
            abstract_coefficients[bf] = sympy.var("c(" + bf.binary + ")")

        # code is currenty unstable but can still succeed if input is shuffled
        # todo: that should not happen, but untill it is fixed I will increase success probability
        count = 0
        success = False

        while (not success and count < self.max_repeat):
            try:
                count += 1
                IMPL = UnaryStatePrepImpl()
                abstract_angles = [sympy.var(IMPL.alphabet(i)) for i in range(len(target_space) - 1)]
                self._abstract_circuit = IMPL.get_circuit(s=[i.binary for i in target_space])
                self._abstract_circuit.n_qubits = self._n_qubits
                success = True
            except NotImplementedError:
                numpy.random.shuffle(target_space)

        if not success: raise TequilaUnaryStateException(
            "Could not disentangle the given state after " + str(count) + " restarts")

        # get the equations to determine the angles
        simulator = BackendCircuitSymbolic(abstract_circuit=self._abstract_circuit, variables={})
        simulator.convert_to_numpy = False
        variables = None # {k:k.name.evalf() for k in self._abstract_circuit.extract_variables()}
        wfn = simulator.simulate(initial_state=BitString.from_int(0, nbits=self.n_qubits), variables=variables)
        wfn.n_qubits = self._n_qubits
        equations = []
        for k in target_space:
            equations.append(wfn[k] - abstract_coefficients[k])

        normeq = -sympy.Integer(1)
        for c in abstract_coefficients.values():
            normeq += c ** 2
        equations.append(normeq)

        # this gives back multiple solutions and some of them sometimes have sign errors ...
        if (self._use_symbolic_solver):
            solutions = sympy.solve(equations, *tuple(abstract_angles), check=True, dict=True)[1]
            if len(abstract_angles) != len(solutions):
                raise TequilaUnaryStateException("Could definetely not solve for the angles in UnaryStatePrep!")
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

    def _evaluate_angles(self, wfn: QubitWaveFunction) -> typing.Dict[sympy.Symbol, float]:
        # coeffs need to be normalized
        wfn = wfn.normalize()

        # initialize the map that will substitute the abstract_coefficients with the QubitWaveFunction coefficients
        subs = dict()
        for key, ac in self._abstract_coeffs.items():
            coeff = 0.0
            if key in wfn:
                coeff = wfn[key]
            if coeff.imag != 0.0:
                raise TequilaException("UnaryStatePrep currently only possible for real coefficients")
            subs[ac] = sympy.Float(float(coeff.real))
        result = dict()
        if (self._use_symbolic_solver):
            # fails a lot of times
            # better don't use
            # get the angle variables from the symbolic equations for the angles
            for symbol, eq in self._angle_equations.items():
                result[symbol] = float(eq.evalf(subs=subs))
        else:
            # numeric solution (more stable)
            # integrated repeat loop for the case that the randomly generated guess is especially bad
            count = 0
            solutions = []
            while (count < self.max_repeat and len(solutions) == 0):
                try:
                    # same substitution map as before, but initialized as list of tuples
                    subsx = [x for x in subs.items()]
                    equations = [eq.subs(subsx) for eq in self._equations]
                    guess = numpy.random.uniform(0.1, 0.9 * 2 * numpy.pi, len(self._abstract_angles))
                    solutions = sympy.nsolve(equations, self._abstract_angles, guess)
                    count += 1
                except:
                    count += 1

            if (len(solutions) == 0):
                raise TequilaUnaryStateException("Failed to numerically solve for angles")

            for i, symbol in enumerate(self._abstract_angles):
                result[symbol] = float(solutions[i].evalf(subs=subs))

        return result

    def __call__(self, wfn: QubitWaveFunction) -> QCircuit:
        """
        :param coeffs: The QubitWaveFunction you want to initialize
        :return:
        """
        try:
            assert (len(wfn) == len(self._target_space))
            for key in wfn.keys():
                try:
                    assert (key in self._target_space)
                except AssertionError:
                    print("key=", key.binary, " not found in target space")
        except AssertionError:
            raise TequilaException("UnaryStatePrep was not initialized for the basis states in your wavefunction\n"
                                   "You gave:\n" + str(wfn) + "\n"
                                                              "But the target_space is " + str(
                [k.binary for k in self._target_space]) + "\n")

        angles = self._evaluate_angles(wfn=wfn)

        # construct new circuit with evaluated angles
        result = QCircuit()
        for g in self._abstract_circuit.gates:
            g2 = copy.deepcopy(g)
            if hasattr(g, "parameter"):
                symbol = g.parameter
                # the module needs repairing ....
                g2._parameter = assign_variable(-angles[-symbol()])  # the minus follows mahas convention since the circuits are daggered in the end
            result += g2

        return result

    def get_circuit(self):
        """
        :return: Return the abstract circuit with tequila parameters
        """
        result = QCircuit()
        for g in self._abstract_circuit.gates:
            g2 = copy.deepcopy(g)
            if hasattr(g, "parameter"):
                symbol = g.parameter
                name = str(-symbol) # kill the minus from the dagger
                g2._parameter = assign_variable(name)
            result += g2

        return result

    def angles(self, wfn: QubitWaveFunction) -> typing.Dict[typing.Hashable, float]:
        sympy_angles = self._evaluate_angles(wfn=wfn)
        angles = {assign_variable(str(key)):value for key, value in sympy_angles.items()}
        return angles