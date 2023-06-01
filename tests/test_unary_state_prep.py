import tequila.simulators.simulator_api
from tequila.apps import UnaryStatePrep
from tequila.apps.unary_state_prep import TequilaUnaryStateException
import numpy
from tequila import BitString
from tequila.simulators.simulator_api import BackendCircuitSymbolic
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
import pytest

@pytest.mark.parametrize("target_space", [['00', '11'], ['01', '11'], ['001', '010', '100'], ['0011', '1010'],
                                          QubitWaveFunction.from_string("1.0|00> - 2.0|11>")])
def test_construction(target_space: list):
    print("ts=", target_space)
    UPS = UnaryStatePrep(target_space=target_space)


@pytest.mark.parametrize("target_space", [['01', '10'], ['001', '010', '100'], ['0011', '0110', '1100', '1001']])
def test_unary_states(target_space: list):
    UPS = UnaryStatePrep(target_space=target_space)
    qubits = len(target_space)
    coeff = 1.0 / numpy.sqrt(qubits)  # fails for the 3-Qubit Case because the wrong sign is picked in the solution
    coeffs = [coeff for i in range(qubits)]

    wfn = QubitWaveFunction()
    for i, c in enumerate(coeffs):
        wfn += c * QubitWaveFunction.from_string("1.0|" + target_space[i] + ">")

    U = UPS(wfn=wfn)
    wfn = BackendCircuitSymbolic(abstract_circuit=U, variables=None).simulate(variables=None)

    checksum = 0.0
    for k, v in wfn.items():
        assert (v.imag == 0.0)
        vv = float(v.real)
        cc = float(coeff.real)
        assert (numpy.isclose(vv, cc, atol=1.e-4))
        checksum += vv

    assert (numpy.isclose(checksum, qubits * coeff, atol=1.e-4))


def get_random_target_space(n_qubits):
    result = []
    while (len(result) < n_qubits):
        i = numpy.random.randint(0, 2 ** n_qubits)
        if i not in result:
            result.append(i)

    return [BitString.from_int(i, nbits=n_qubits).binary for i in result]


@pytest.mark.xfail(reason="module is far from perfect")
@pytest.mark.parametrize("target_space", [get_random_target_space(n_qubits=qubits) for qubits in range(2, 5)])
def test_random_instances(target_space):

    # can happen that a tests fails, just start again ... if all tests fail: start to worry
    # it happens from time to time that UPS can not disentangle
    # it will throw the error/
    # OpenVQEException: Could not disentangle the given state after 100 restarts
    qubits = len(target_space)
    coeffs = numpy.random.uniform(0, 1, qubits)

    wfn = QubitWaveFunction()
    for i, c in enumerate(coeffs):
        wfn += c * QubitWaveFunction.from_string("1.0|" + target_space[i] + ">")
    wfn = wfn.normalize()

    try:
        UPS = UnaryStatePrep(target_space=target_space)
        U = UPS(wfn=wfn)

        # now the coeffs are normalized
        bf2c = dict()
        for i, c in enumerate(coeffs):
            bf2c[target_space[i]] = coeffs[i]

        wfn2 = tequila.simulators.simulator_api.simulate(U, initial_state=0, variables=None)

        for k, v in wfn.items():
            assert (numpy.isclose(wfn2[k], v))

    except TequilaUnaryStateException:
        print("caught a tolerated excpetion")
