from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crc_lodo_bench.readiness import (
    CrossPopulationThresholds,
    compute_cohort_target_associations,
    summarize_cross_population_evidence,
    validate_target_registry,
)


def _registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "target_id": "toxin",
                "display_name": "Example toxin",
                "score_prefix": "example_toxin",
                "parent_taxon": "Example bacterium",
                "prespecified_genes": "toxA;toxB",
                "target_role": "known_target_benchmark",
                "selection_basis": "external evidence",
                "anchor_evidence_url": "https://example.org/evidence",
                "causal_evidence_status": "pending_structured_review",
                "editability_evidence_status": "pending_structured_review",
                "registry_status": "frozen_benchmark",
            }
        ]
    )


def _scores(n_cohorts: int = 5) -> pd.DataFrame:
    rows = []
    for cohort_index in range(n_cohorts):
        cohort = f"cohort_{cohort_index}"
        for sample_index, (label, abundance) in enumerate(
            [(0, 0.0), (0, 0.0), (1, 1.0), (1, 2.0)]
        ):
            rows.append(
                {
                    "sample_id": f"{cohort}_{sample_index}",
                    "study_name": cohort,
                    "country": f"country_{cohort_index}",
                    "label": label,
                    "example_toxin__abundance": abundance,
                    "example_toxin__completeness": 0.0 if abundance == 0 else 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_registry_rejects_duplicate_target_ids():
    registry = pd.concat([_registry(), _registry()], ignore_index=True)
    with pytest.raises(ValueError, match="target_id"):
        validate_target_registry(registry)


def test_cohort_evidence_keeps_populations_separate():
    evidence = compute_cohort_target_associations(_scores(), _registry())

    assert len(evidence) == 5
    assert evidence["evaluable"].all()
    assert evidence["direction"].eq("crc_enriched").all()
    assert np.allclose(evidence["crc_prevalence"], 1.0)
    assert np.allclose(evidence["control_prevalence"], 0.0)
    assert np.allclose(evidence["association_auc"], 1.0)


def test_cross_population_gate_has_frozen_minimums():
    registry = _registry()
    evidence = compute_cohort_target_associations(_scores(), registry)
    summary = summarize_cross_population_evidence(evidence, registry)

    assert summary.loc[0, "cross_population_gate"] == "pass"
    assert summary.loc[0, "overall_readiness"] == "not_yet_assessable"
    assert summary.loc[0, "conservation_status"] == "not_yet_assessed"


def test_missing_assay_is_not_interpreted_as_absence():
    registry = _registry().assign(score_prefix="not_measured")
    evidence = compute_cohort_target_associations(_scores(), registry)

    assert not evidence["assay_available"].any()
    assert not evidence["evaluable"].any()
    assert evidence["direction"].eq("not_assayed").all()
    assert evidence["association_auc"].isna().all()


def test_threshold_validation_rejects_permissive_direction_rule():
    with pytest.raises(ValueError, match="min_crc_enriched_fraction"):
        CrossPopulationThresholds(min_crc_enriched_fraction=0.49)
