"""Shared pytest configuration for the crc-metagenomics test suite.

Currently this file does one job: register the ``slow`` marker used by
``tests/test_orchestration_scripts.py`` so pytest does not emit
``PytestUnknownMarkWarning`` for tests that take >1 second (e.g. the
subprocess invocations of ``scripts/bootstrap_ci.py``).

Run only the fast subset with:
    pytest tests/ -m "not slow"
"""
from __future__ import annotations


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests that take more than ~1 second to run "
        "(typically subprocess invocations of full orchestration scripts). "
        "Skip with ``pytest -m 'not slow'``.",
    )
