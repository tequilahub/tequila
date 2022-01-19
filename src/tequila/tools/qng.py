from tequila import TequilaException
from tequila.hamiltonian import paulis
from tequila.objective.objective import Objective, ExpectationValueImpl, ExpectationValue
from tequila.circuit.circuit import QCircuit
from tequila.simulators.simulator_api import compile_objective
from tequila.circuit.gradient import __grad_inner
from tequila.autograd_imports import jax
from tequila.circuit.compiler import compile_controlled_rotation, compile_power_gate, \
    compile_trotterized_gate, compile_controlled_phase, compile_multitarget
import typing
import numpy
import copy


class QngMatrix:

    """
    A callable class which is meant to be used for calculating the inverse qgt of an expectationvalue.


    takes a single init: blocks, a list of lists of lists, which each list of lists being a block.
    This makes block-diagonal approximations easier to calculate. In the event that a method
    which is not block diagonal is to be used, then there is only one block: the whole matrix.

    Attributes
    ----------
    blocks:
        the list of lists of lists of (nonzero) qgt terms.
    """

    @property
    def dim(self):
        """
        obtain the dimensions of the inverse qgt.
        Returns
        -------
        tuple
            the shape of the square matrix returned by call.
        """
        d = 0
        for block in self.blocks:
            d += len(block)
        return (d,d)

    def __init__(self,blocks):
        """
        Parameters
        ----------
        blocks: list:
            list of list of lists. the blocks of the qgt.
        """
        self.blocks= blocks

    def __call__(self, variables,samples=None) -> numpy.ndarray:
        """
        from the blocks provided, evaluate all qgt terms, then calculate the pseudo-inverse, and return it.
        Parameters
        ----------
        variables: dict:
            the parameters to evaluate terms which are callable, i.e, objectives
        samples: int, optional:
            the number of samples with which to evaluate callable (objective) qgt terms.
        Returns
        -------
        numpy array representing the inverse qgt of some expectationvalue.
        """

        output = numpy.zeros(self.dim)
        d_v = 0
        # blocks are square, so if we are using a block diagonal approx, we need to
        # displace our running index of position, as we enumerate through a block
        # d_v does this. If you only provide one block (the whole QGT), this won't matter
        for block in self.blocks:

            d_v_temp = 0 # how much to increment d_v by when done with the block at hand.
            for i, row in enumerate(block):
                for j, term in enumerate(row):
                    if i <= j:
                        # if its an objective, call it. Else, it is a float.
                        try:
                            output[i + d_v][j + d_v] = term(variables=variables,samples=samples)
                        except:
                            output[i + d_v][j + d_v] = term
                    else:
                        output[i + d_v][j + d_v] = output[j + d_v][i + d_v]
                d_v_temp += 1
            d_v += d_v_temp

        back = numpy.linalg.pinv(output) # return the pseudo_inverse of the matrix!
        return back


class CallableVector:
    """
    class representng a generic, callable vector; to be used in vector representations of gradients.

    Attributes
    ----------
    vector:
        list of terms, implicitly floats or objectives, to be called and/or forwarded into a numpy array.
    """

    @property
    def dim(self):
        return (len(self._vector),)

    def __init__(self,vector):
        """
        init.
        Parameters
        ----------
        vector:
            a list of terms to return as a vector when called.
        """
        self._vector = vector

    def __call__(self, variables, samples=None) -> numpy.ndarray:
        """
        return a vector from a list of Objectives and numbers, calling all callables.

        Parameters
        ----------
        variables: dict:
            the parameters to evaluate terms which are callable, i.e, objectives
        samples: int: (Default None):
            the number of samples with which to evaluate callable (objective) qgt terms.

        Returns
        -------
        numpy.ndarray
            result of evaluating a vector of objectives
        """

        output = numpy.empty(self.dim)
        for i, entry in enumerate(self._vector):
            if hasattr(entry, '__call__'):
                output[i] = entry(variables, samples=samples)
            else:
                output[i] = entry
        return output


def get_generator(gate) -> paulis.QubitHamiltonian:
    """
    get the generator of a gaussian gate as a Qubit hamiltonian. Relies on the name of the gate.
    Parameters
    ----------
    gate: QGateImpl:
        QGateImpl object or inheritor thereof, with name corresponding to its generator in some fashion.

    Returns
    -------
    QubitHamiltonian:
        the generator of the gate acting, on the gate's target.

    """

    if gate.name.lower() == 'rx':
        gen = paulis.X(gate.target[0])
    elif gate.name.lower() == 'ry':
        gen = paulis.Y(gate.target[0])
    elif gate.name.lower() == 'rz':
        gen = paulis.Z(gate.target[0])
    elif gate.name.lower() == 'phase':
        gen = paulis.Qm(gate.target[0])
    else:
        print(gate.name.lower())
        raise TequilaException('cant get the generator of a non Gaussian gate, you fool!')
    return gen


def stokes_block(expectation, initial_values=None, samples=None, device=None,
                 backend=None,
                 noise=None) -> typing.List[typing.List[typing.List[typing.Union[float,Objective]]]]:

    """
    returns the blocks of the layerwise block-diagonal approximation to the qgt.
    The default for all qng-based optimizations, as a method for obtaining the qgt.
    See: Stokes et. al, https://arxiv.org/abs/1909.02108

    Parameters
    ----------
    expectation: ExpectationValueImpl:
        the expectation value whose qgt is to be built.
    initial_values: dict, optional:
        a dictionary of initial values with which Objectives in the qgt should be compiled
    samples: int, optional:
        the number of samples with which Objectives in the qgt should be compiled
    device: optional:
        the device (real, or simulated, and specific to the backend) with which Objectives in the qgt
        should be compiled. Generally, a string, but backend specific types also accepted.
    backend: str, optional:
        the backend with which Objectives in the qgt should be compiled
    noise: str or NoiseModel, optional:
        the noise model  with which Objectives in the qgt should be compiled

    Returns
    -------
    list of list of lists:
        list of list of lists representing the blocks of the block diagonal layerwise approx to the qgt.

    """

    U = expectation.U
    # orders the circuit into alternating layer ansatz, where a moment is all simultaneous gates
    moments = U.canonical_moments
    # rebuild the sub circuits used in the expectation values that populate the QGT
    sub = [QCircuit.from_moments(moments[:i]) for i in range(1, len(moments), 2)]
    # this is the list of just the moments which are parametrized.
    parametric_moms = [moments[i] for i in range(1, len(moments)+1, 2)]
    generators = []
    # generators is a list of lists, ultimately, where each sublist is all the generators in order
    # for a given parametric layer (if said layer is
    # occupied, which it might not be! a layer can be nothing, I.E, the identity.)
    for pm in parametric_moms:
        g_set = []
        if len(pm.gates) != 0:
            for gate in pm.gates:
                # get_generator takes a gaussian gate, and returns the pauli that is its generator.
                # See that function for detail.
                gen = get_generator(gate)
                g_set.append(gen)
        if len(g_set) != 0:
            generators.append(g_set)
        else:
            # blank sets get passed over
            generators.append(None)
    blocks = []
    for i, g_set in enumerate(generators):
        if g_set is None:
            pass
        else:
            # a block is a list of lists, and indexing it should correspond to indexing a matrix in
            # A[row][column] fashion. alternate functions could have the whole QGT be a single block,
            # but you need to have as a return a List of (List of Lists)!!!!
            block = [[0 for _ in range(len(g_set))] for _ in range(len(g_set))]
            for k, gen1 in enumerate(g_set):
                for q, gen2 in enumerate(g_set):
                    ### make sure you compile the objectives! otherwise this bad boy will not run
                    if k == q:
                        arg = (ExpectationValue(U=sub[i], H=gen1 * gen1) - ExpectationValue(U=sub[i], H=gen1)**2)/4
                    else:
                        arg = (ExpectationValue(U=sub[i], H=gen1 * gen2) - ExpectationValue(U=sub[i], H=gen1) *
                               ExpectationValue(U=sub[i], H=gen2)) / 4
                    block[k][q] = compile_objective(arg, variables=initial_values, samples=samples,
                                                    backend=backend, device=device,
                                                    noise=noise)
            blocks.append(block)
    return blocks


def qng_circuit_grad(E: ExpectationValueImpl) -> typing.List[Objective]:
    """
    tool for constructing the gradient of an expectationvalue,
    but without digging deeper into the underlying parameters; Unlike it's cousin in the usual gradient,
    this version never extracts the variables from a gate's parameter.

    Parameters
    ----------
    E: ExpectationValueImpl:
        the expectation value whose gradient is to be returned.

    Returns
    -------
    list of objectives:
        list of dU/dp_i, with each p_i being the parameter of an individual gate.

    """
    hamiltonian = E.H
    unitary = E.U

    # fast return if possible
    out=[]
    for i, g in enumerate(unitary.gates):
        if g.is_parametrized():
            if g.is_controlled():
                raise TequilaException("controlled gate in qng circuit gradient: Compiler was not called")
            if hasattr(g, "eigenvalues_magnitude"):
                if hasattr(g._parameter,'extract_variables'):
                    shifter = qng_grad_gaussian(unitary, g, i, hamiltonian)
                    out.append(shifter)
            else:
                print(g, type(g))
                raise TequilaException('No shift found for gate {}'.format(g))
    if out is None:
        raise TequilaException("caught a dead circuit in qng gradient")
    return out


def qng_grad_gaussian(unitary, g, i, hamiltonian) -> Objective:
    """
    get the gradient of an expectationvalue of a unitary and a hamiltonian with respect to gaussian gate g.
    THIS variant of the function does not seek out underlying gate parameters; it treats each variable 'as is'.
    This treatment is necessary for the QNG but is incorrect elsewhere.

    Parameters
    ----------
    unitary: QCircuit:
        the QCircuit object containing the gate to be differentiated
    g: parametrized gate:
        the gate being differentiated
    i: int:
        the position in unitary at which g appears.
    hamiltonian: QubitHamiltonian:
        the hamiltonian with respect to which unitary is to be measured, in the case that unitary
        is contained within an ExpectationValue
    Returns
    -------
    Objective:
        the analytical gradient of  <U,H> w.r.t g=g(theta_g)
    """

    # unlike grad_gaussian, this doesn't dig below, into a gate's underlying parametrization.
    # In other words, if a gate is Rx(y), y=f(x), this gives you back d Rx / dy, not d Rx/dy * dy/dx

    if not hasattr(g, "eigenvalues_magnitude"):
        raise TequilaException("No shift found for gate {}".format(g))

    # neo_a and neo_b are the shifted versions of gate g needed to evaluate its gradient
    shift_a = g._parameter + numpy.pi / (4 * g.eigenvalues_magnitude)
    shift_b = g._parameter - numpy.pi / (4 * g.eigenvalues_magnitude)
    neo_a = copy.deepcopy(g)
    neo_a._parameter = shift_a
    neo_b = copy.deepcopy(g)
    neo_b._parameter = shift_b

    U1 = unitary.replace_gates(positions=[i], circuits=[neo_a])
    w1 = g.eigenvalues_magnitude

    U2 = unitary.replace_gates(positions=[i], circuits=[neo_b])
    w2 = -g.eigenvalues_magnitude

    Oplus = ExpectationValueImpl(U=U1, H=hamiltonian)
    Ominus = ExpectationValueImpl(U=U2, H=hamiltonian)
    dOinc = w1 * Objective(args=[Oplus]) + w2 * Objective(args=[Ominus])
    return dOinc


def subvector_procedure(e_val, initial_values=None, samples=None, device=None,
                        backend=None, noise=None) -> CallableVector:
    """
    take an expectation value and return its (qng style) gradient as a CallableVector.

    Parameters
    ----------
    e_val: ExpectationValueImpl:
        the expectation value whose gradient is to be obtained
    initial_values: dict, optional:
        a dictionary of initial values with which Objectives in the qgt should be compiled
    samples: int, optional:
        the number of samples with which Objectives in the qgt should be compiled
    device: typing depends on backend, but in general, str (Default value = None):
        the device (real, or simulated, and specific to the backend) with which Objectives in the qgt should be compiled
    backend: str, optional:
        the backend with which Objectives in the qgt should be compiled
    noise: str or NoiseModel, optional:
        the noise model  with which Objectives in the qgt should be compiled
    Returns
    -------
    CallableVector, the gradient of an ExpectationValue in callable format.

    """

    vect = qng_circuit_grad(e_val)
    out = []
    for entry in vect:
        out.append(compile_objective(entry, variables=initial_values, samples=samples, device=device,
                                     backend=backend,
                                     noise=noise))
    return CallableVector(out)


def get_self_pars(U) -> typing.List:
    """
    get the parameters of circuit U, without extracting underlying variables.
    Parameters
    ----------
    U: QCircuit:
        the circuit whose 'self' parameters are to be extracted.

    Returns
    -------
    list:
        list eg. of Objectives and Variables; the self-parameters of a circuit.
    """

    out=[]
    for g in U.gates:
        if g.is_parametrized():
            if hasattr(g._parameter,'extract_variables'):
                out.append(g._parameter)
    return out


def qng_dict(argument, matrix, subvector, mapping, positional) -> typing.Dict:
    """
    helper function to obtain a formatted dictionary,
    for the easy transport of objects and their qng-related structures.

    Parameters
    ----------
    argument:
        the object whose qgt, euclidean gradient, etc. are contained herein.
    matrix: QngMatrix:
        the callable that obtains the QGT^-1 of argument.
    subvector: CallableVector:
        the CallableVector of the (self-parameter) gradient of argument.
    mapping: dict:
        a dictionary mapping the self parameters of argument to the real parameters of an objective.
    positional: Objective:
        the positional derivative of the objective from whence argument came w.r.t to argument.

    Returns
    -------
    dict:
        dict containing information used to obtain the qng of some argument of an objective.

    """
    return {'arg': argument, 'matrix': matrix, 'vector': subvector, 'mapping': mapping, 'positional': positional}


def get_qng_combos(objective, func=stokes_block,
                   initial_values=None, samples=None,
                   backend=None, device=None, noise=None) -> typing.List[typing.Dict]:

    """
    get all the objects needed to evaluate the qng for some objective; return them in a list of dictionaries.

    Parameters
    ----------
    objective: Objective:
        the Objective whose qng is sought.
    func: callable: (Default = stokes_block):
        the function used to obtain the (blocks of) the qgt. Default uses stokes_block, defined above.
    initial_values: dict, optional:
        a dictionary indicating the intial parameters with which to compile all objectives appearing in the qng.
    samples: int, optional:
        the number of samples with which to compile all objectives appearing in the qng. Default: none.
    backend: str, optional:
        the backend with which to compile all objectives appearing in the qng. default: pick for you.
    device: optional:
        the device with which to compile all objectives appearing in the qng. Default: no device use or emulation.
    noise: str or NoiseModel, optional:
        the noise model with which to compile all objectives appearing in the qng. Default: no noise.

    Returns
    -------
    list of dicts:
        a list of dictionaries, each entry corresponding to the qng for 1 argument of objective, in the order
        of said objectives.

    """

    combos = []
    var_list = objective.extract_variables()
    objective.contract()
    compiled = compile_multitarget(gate=objective)
    compiled = compile_trotterized_gate(gate=compiled)
    compiled = compile_power_gate(gate=compiled)
    compiled = compile_controlled_phase(gate=compiled)
    compiled = compile_controlled_rotation(gate=compiled)
    for i,arg in enumerate(compiled.args):
        if not isinstance(arg, ExpectationValueImpl):
            # this is a variable, no QNG involved
            mat = QngMatrix([[[1]]])
            vec = CallableVector([__grad_inner(arg, arg)])
            mapping = {0: {v: __grad_inner(arg, v) for v in var_list}}
        else:
            # if the arg is an expectationvalue, we need to build some qngs and mappings!
            blocks = func(arg, initial_values=initial_values, samples=samples, device=device,
                          backend=backend, noise=noise)

            mat = QngMatrix(blocks)

            vec = subvector_procedure(arg, initial_values=initial_values, samples=samples, device=device,
                                      backend=backend, noise=noise)

            mapping = {}
            self_pars = get_self_pars(arg.U)
            for j, p in enumerate(self_pars):
                indict = {}
                for v in p.extract_variables():
                    gi = __grad_inner(p, v)
                    if isinstance(gi, Objective):
                        g = compile_objective(gi, variables=initial_values, samples=samples, device=device,
                                              backend=backend, noise=noise)
                    else:
                        g = gi
                    indict[v] = g
                mapping[j] = indict


        pos_arg = jax.grad(compiled.transformation, i)
        p = Objective(compiled.args, transformation=pos_arg)

        pos = compile_objective(p, variables=initial_values, samples=samples, device=device,
                                backend=backend, noise=noise)
        combos.append(qng_dict(arg, mat, vec, mapping, pos))
    return combos


def evaluate_qng(combos, variables, samples=None) -> list:

    """
    actually evaluate the terms of a qng.
    Parameters
    ----------
    combos: list of dicts:
        the dictionaries of qng evaluables of some object.
    variables: dict:
        the variables, with which to evaluate the qng.
    samples: int, optional:
        the number of samples, with which to evaluate the qng.

    Returns
    -------
    list:
        list of floats: the evaluated gradient of the qng.

    """
    gd = {v: 0 for v in variables.keys()}
    for c in combos:
        qgt = c['matrix']
        vec = c['vector']
        m = c['mapping']
        pos = c['positional']
        marco = qgt(variables, samples=samples)
        polo = vec(variables, samples=samples)
        ev = numpy.dot(marco, polo)
        for i, val in enumerate(ev):
            maps = m[i]
            for k in maps.keys():
                gd[k] += (val*maps[k]*pos)(variables=variables, samples=samples)

    out = [v for v in gd.values()]
    return out


class QNGVector:
    """
    the class, similar to CallableVector, that returns the qng of some object as a vector.

    Attributes
    ----------
    combos:
        list of qng dicts for getting a qng.

    """

    def __init__(self,combos):
        """
        init.
        Parameters
        ----------
        combos:
            combos the dicts to get the QNG with.
        """
        self.combos = combos

    def __call__(self, variables, samples=None):
        """
        return the QNG of something.

        Parameters
        ----------
        variables: dict:
            the variables with which to evaluate the qng
        samples: int (Default value = None):
            the samples with which to evaluate the qng

        Returns
        -------
        numpy.ndarray
            a qng.
        """
        return numpy.asarray(evaluate_qng(self.combos, variables=variables, samples=samples))
