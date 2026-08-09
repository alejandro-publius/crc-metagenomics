from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from select_gene_family_manifests import (  # noqa: E402
    build_training_cohort_map,
    select_fold_features,
)


def test_country_aware_training_map_excludes_same_country_cohort():
    metadata = pd.DataFrame(
        {
            "sample_id": ["a0", "a1", "b0", "b1", "c0", "c1"],
            "study_name": ["A", "A", "B", "B", "C", "C"],
            "country": ["US", "US", "US", "US", "DE", "DE"],
            "label": [0, 1, 0, 1, 0, 1],
        }
    )

    folds = build_training_cohort_map(metadata)

    assert folds["A"] == ["C"]
    assert folds["B"] == ["C"]
    assert folds["C"] == ["A", "B"]


def test_feature_selection_uses_training_cohorts_only():
    summaries = pd.DataFrame(
        [
            {"cohort": "A", "gene_id": "shared", "n_samples": 50,
             "n_nonzero": 10, "total_abundance": 4.0},
            {"cohort": "B", "gene_id": "shared", "n_samples": 50,
             "n_nonzero": 10, "total_abundance": 4.0},
            {"cohort": "C", "gene_id": "test_only", "n_samples": 50,
             "n_nonzero": 50, "total_abundance": 40.0},
            {"cohort": "A", "gene_id": "one_cohort", "n_samples": 50,
             "n_nonzero": 30, "total_abundance": 20.0},
        ]
    )

    selected = select_fold_features(
        summaries,
        training_cohorts=["A", "B"],
        cohort_sample_counts={"A": 50, "B": 50, "C": 50},
        min_prevalence=0.05,
        min_cohorts=2,
        max_features=100,
    )

    assert selected["gene_id"].tolist() == ["shared"]
    assert selected.loc[0, "prevalence"] == 0.2


def test_feature_selection_ranking_and_cap_are_deterministic():
    summaries = pd.DataFrame(
        [
            {"cohort": cohort, "gene_id": gene, "n_samples": 50,
             "n_nonzero": nonzero, "total_abundance": abundance}
            for cohort in ["A", "B"]
            for gene, nonzero, abundance in [
                ("high", 20, 10.0),
                ("medium", 10, 20.0),
                ("low", 5, 100.0),
            ]
        ]
    )

    selected = select_fold_features(
        summaries,
        training_cohorts=["A", "B"],
        cohort_sample_counts={"A": 50, "B": 50},
        min_prevalence=0.05,
        min_cohorts=2,
        max_features=2,
    )

    assert selected["gene_id"].tolist() == ["high", "medium"]
