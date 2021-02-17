import numpy as np
import pytest
import tequila as tq
from tequila.ml import HAS_TORCH

if HAS_TORCH:
    import torch
    from torch import optim


@pytest.mark.dependencies
def test_dependencies():
    assert HAS_TORCH


@pytest.mark.skipif(condition=not HAS_TORCH, reason="you don't have PyTorch")
@pytest.mark.parametrize("angles", [np.random.uniform(0, np.pi*2, 3)])
def test_calls_correctly(angles):
    U1 = tq.gates.Rx(angle='a', target=0)
    H1 = tq.paulis.Y(0)
    U2 = tq.gates.Ry(angle='b', target=0)
    H2 = tq.paulis.X(0)
    U3 = tq.gates.H(0) + tq.gates.Rz(angle='c',target=0) + tq.gates.H(0)
    H3 = tq.paulis.Y(0)

    evals = [tq.ExpectationValue(U1, H1), tq.ExpectationValue(U2, H2), tq.ExpectationValue(U3, H3)]
    stacked = tq.vectorize(evals)
    torched = tq.ml.to_platform(stacked, platform='torch', input_vars=['a', 'b', 'c'])
    inputs = torch.from_numpy(angles)
    output = torched(inputs)
    summed = output.sum().detach().numpy()
    analytic = -np.sin(angles[0]) + np.sin(angles[1]) - np.sin(angles[2])
    assert np.isclose(summed, analytic, atol=1.e-3)


@pytest.mark.skipif(condition=not HAS_TORCH, reason="you don't have PyTorch")
def test_example_training():
    U = tq.gates.Rx('a', 0) + tq.gates.Rx('b', 1) + tq.gates.CNOT(1, 3) + tq.gates.CNOT(0, 2) + tq.gates.CNOT(0, 1)
    H1 = tq.paulis.Qm(1)
    H2 = tq.paulis.Qm(2)
    H3 = tq.paulis.Qm(3)

    stackable = [tq.ExpectationValue(U, H1), tq.ExpectationValue(U, H2), tq.ExpectationValue(U, H3)]
    stacked = tq.vectorize(stackable)

    initial_values = {'a': 1.5, 'b': 2.}
    cargs = {'samples': None, 'backend': 'random', 'initial_values': initial_values}
    torched = tq.ml.to_platform(stacked, platform='torch', compile_args=cargs)
    optimizer = optim.SGD(torched.parameters(), lr=.1, momentum=0.9)

    for i in range(80):
        optimizer.zero_grad()
        out = torched()
        loss = out.sum()
        loss.backward()
        optimizer.step()

    called = torched().sum().detach().numpy()
    assert np.isclose(called, 0.0, atol=1e-3)
