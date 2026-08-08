from __future__ import annotations

import pandas as pd
from sklearn.metrics import roc_auc_score


def test_species_aware_results_match_predictions_and_audit():
    results = pd.read_csv("results/species_aware_correction/lodo_results.csv")
    predictions = pd.read_csv("results/species_aware_correction/predictions.csv")
    audit = pd.read_csv("results/species_aware_correction/correction_audit.csv")

    assert len(results) == 30
    assert len(predictions) == 3 * 1339
    assert len(audit) == 10
    assert not audit["target_distribution_used_for_source_only"].any()
    assert audit["target_distribution_used_for_adaptive"].all()
    assert audit["train_pathway_parent_coverage"].mean() > 0.80

    recomputed = {
        (model, cohort): roc_auc_score(frame["y_true"], frame["y_prob"])
        for (model, cohort), frame in predictions.groupby(["model", "cohort"])
    }
    for row in results.itertuples(index=False):
        assert abs(row.auc - recomputed[(row.model, row.cohort)]) < 1e-12

    means = results.groupby("model")["auc"].mean()
    assert abs(means["species_source_only"] - 0.8139672003914212) < 1e-12
    assert abs(means["stratified_source_only"] - 0.772825119468387) < 1e-12
    assert abs(means["stratified_target_adaptive"] - 0.7765138403257461) < 1e-12
