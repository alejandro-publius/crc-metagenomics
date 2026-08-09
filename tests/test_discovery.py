from __future__ import annotations

import numpy as np
import pandas as pd

from crc_lodo_bench.discovery import (
    DiscoveryThresholds,
    build_cross_fitted_evidence,
    compute_gene_cohort_statistics,
    select_training_candidates,
    summarize_cross_fitted_candidates,
)


def _synthetic_data():
    metadata_rows = []
    matrix_rows = []
    for cohort_index in range(8):
        cohort = f"cohort_{cohort_index}"
        for sample_index, label in enumerate([0, 0, 0, 1, 1, 1]):
            metadata_rows.append(
                {
                    "sample_id": f"{cohort}_{sample_index}",
                    "study_name": cohort,
                    "country": f"country_{cohort_index}",
                    "label": label,
                }
            )
            robust = 1.0 + sample_index if label else 0.0
            heldout_only = 2.0 if (cohort_index == 7 and label) else 0.0
            noise = float((sample_index + cohort_index) % 2)
            matrix_rows.append([robust, heldout_only, noise])
    return (
        np.asarray(matrix_rows),
        ["robust_gene", "heldout_only_gene", "noise_gene"],
        pd.DataFrame(metadata_rows),
    )


def test_vectorized_auc_and_prevalence_are_correct():
    matrix, genes, metadata = _synthetic_data()
    stats = compute_gene_cohort_statistics(matrix, genes, metadata)
    robust = stats[stats["gene_id"].eq("robust_gene")]

    assert len(robust) == 8
    assert np.allclose(robust["association_auc"], 1.0)
    assert np.allclose(robust["prevalence_difference"], 1.0)
    assert robust["direction"].eq("crc_enriched").all()


def test_heldout_outcome_cannot_select_itself():
    matrix, genes, metadata = _synthetic_data()
    stats = compute_gene_cohort_statistics(matrix, genes, metadata)
    training = [f"cohort_{index}" for index in range(7)]
    thresholds = DiscoveryThresholds(
        min_training_cohorts=3,
        max_candidates_per_fold=10,
        min_outer_selections=3,
        min_heldout_evaluable=3,
    )
    selected = select_training_candidates(stats, training, thresholds=thresholds)

    assert "robust_gene" in set(selected["gene_id"])
    assert "heldout_only_gene" not in set(selected["gene_id"])


def test_cross_fitted_summary_nominates_only_repeated_outer_evidence():
    matrix, genes, metadata = _synthetic_data()
    stats = compute_gene_cohort_statistics(matrix, genes, metadata)
    training_map = {
        f"cohort_{heldout}": [
            f"cohort_{training}" for training in range(8) if training != heldout
        ]
        for heldout in range(8)
    }
    thresholds = DiscoveryThresholds(
        min_training_cohorts=3,
        max_candidates_per_fold=10,
        min_outer_selections=7,
        min_heldout_evaluable=7,
    )
    _selections, evidence = build_cross_fitted_evidence(
        stats, training_map, thresholds=thresholds
    )
    summary = summarize_cross_fitted_candidates(evidence, thresholds=thresholds)

    robust = summary.set_index("gene_id").loc["robust_gene"]
    assert bool(robust["internal_nomination"])
    assert robust["n_outer_selections"] == 8
    assert robust["external_confirmation_status"] == "not_yet_assessed"
