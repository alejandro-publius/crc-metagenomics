from __future__ import annotations

import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from verify_intervention_readiness import verify  # noqa: E402


def test_committed_intervention_claims_verify():
    root = Path(__file__).resolve().parents[1]
    results = root / "results" / "intervention_readiness"
    report = verify(results, results / "dossiers")
    assert report["status"] == "pass"


def test_missing_results_are_not_silently_accepted(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify(tmp_path, tmp_path / "dossiers")
