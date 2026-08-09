from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from validate_known_target_evidence import validate_and_summarize  # noqa: E402


def _registry() -> pd.DataFrame:
    return pd.DataFrame({"target_id": ["colibactin"]})


def _evidence() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "evidence_id": "a",
                "target_id": "colibactin",
                "source_url": "https://example.org/a",
                "publication_status": "peer_reviewed",
                "primary_source": True,
                "evidence_model": "animal",
                "perturbation": "deletion",
                "main_observation": "changed phenotype",
                "causal_tier": "C2",
                "editability_tier": "E1",
                "key_limitation": "not delivered",
            },
            {
                "evidence_id": "b",
                "target_id": "colibactin",
                "source_url": "https://example.org/b",
                "publication_status": "preprint",
                "primary_source": True,
                "evidence_model": "animal",
                "perturbation": "delivered CRISPRi",
                "main_observation": "changed phenotype",
                "causal_tier": "C2",
                "editability_tier": "E3",
                "key_limitation": "preprint",
            },
        ]
    )


def test_summary_keeps_causal_and_editability_tiers_separate():
    summary = validate_and_summarize(_evidence(), _registry())
    assert summary.loc[0, "highest_causal_tier"] == "C2"
    assert summary.loc[0, "highest_editability_tier"] == "E3"
    assert not summary.loc[0, "experiment_ready"]


def test_non_primary_row_cannot_set_a_tier():
    evidence = _evidence()
    evidence.loc[0, "primary_source"] = False
    with pytest.raises(ValueError, match="primary research"):
        validate_and_summarize(evidence, _registry())
