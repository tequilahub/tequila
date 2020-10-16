"""Pytest configuration."""
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="Run every test, even slow ones.",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: Mark a test or option as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--slow"):
        # --slow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="skipped, --slow not selected")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
