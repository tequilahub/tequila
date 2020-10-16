"""Simulators for parametrized testing."""
import pytest
from typing import Sequence, List, Any
import tequila.simulators.simulator_api


def get(sampler: bool = False, skip: Sequence[str] = ["symbolic"]) -> List[Any]:
    """Returns a pytest-parametrized list of simulators.

    Returns a list of simulators that includes the default one + any simulator
    that is installed. The non-default ones are returned wrapped in
    pytest.param(, marks=pytest.mark.slow)
    """
    if sampler:
        installed = tequila.simulators.simulator_api.INSTALLED_SAMPLERS
        samples = 1
    else:
        installed = tequila.simulators.simulator_api.INSTALLED_SIMULATORS
        samples = None

    # Default simulator
    simulators = [tequila.simulators.simulator_api.pick_backend(samples=samples)]
    # The other ones
    for k in installed.keys():
        if (k not in skip) and (k not in simulators):
            simulators.append(pytest.param(k, marks=pytest.mark.slow))

    return simulators

