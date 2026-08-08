from __future__ import annotations

import pandas as pd
from sklearn.metrics import roc_auc_score


EXPECTED_COHORTS = {
    "FengQ_2015",
    "GuptaA_2019",
    "ThomasAM_2018a",
    "ThomasAM_2018b",
    "ThomasAM_2019_c",
    "VogtmannE_2016",
    "WirbelJ_2018",
    "YachidaS_2019",
    "YuJ_2015",
    "ZellerG_2014",
}


def test_committed_gene_family_results_match_predictions():
    results = pd.read_csv("results/gene_family_lodo_results.csv")
    predictions = pd.read_csv("results/preds_gene_family_elastic_net.csv")

    assert set(results["cohort"]) == EXPECTED_COHORTS
    assert results["cohort"].is_unique
    assert predictions["sample_id"].is_unique
    assert len(predictions) == 1339
    assert (results["n_features"] == 5000).all()

    recomputed = {
        cohort: roc_auc_score(frame["y_true"], frame["y_prob"])
        for cohort, frame in predictions.groupby("cohort")
    }
    for row in results.itertuples(index=False):
        assert abs(row.auc - recomputed[row.cohort]) < 1e-12

    assert abs(results["auc"].mean() - 0.6932728184043169) < 1e-12
