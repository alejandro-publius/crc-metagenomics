from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from summarize_intervention_funnel import (  # noqa: E402
    build_funnel,
    build_taxon_threshold_sensitivity,
)


def test_funnel_preserves_all_gate_counts():
    parent = pd.DataFrame(
        {"parent_adjustment_gate": ["pass", "pass", "not_passed"]}
    )
    taxon = pd.DataFrame(
        {"taxonomic_resolution_status": ["dominant_source_observed", "mixed"]}
    )
    funnel = build_funnel(100, parent, taxon)
    assert funnel["n_candidates"].tolist() == [100, 3, 2, 1]


def test_threshold_sensitivity_uses_greater_than_or_equal():
    taxon = pd.DataFrame({"dominant_taxon_fraction": [0.5, 0.8]})
    sensitivity = build_taxon_threshold_sensitivity(taxon, (0.5, 0.8, 0.9))
    assert sensitivity["n_candidates_passing"].tolist() == [2, 1, 0]
    assert sensitivity["prespecified_primary_threshold"].tolist() == [False, True, False]
