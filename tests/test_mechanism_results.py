from __future__ import annotations

import hashlib
import json

import pandas as pd
from sklearn.metrics import roc_auc_score


def test_frozen_manifest_matches_recorded_checksum():
    payload = open("results/mechanism_panel/frozen_manifest.csv", "rb").read()
    freeze = json.load(open("results/mechanism_panel/freeze.json"))

    assert hashlib.sha256(payload).hexdigest() == freeze["manifest_sha256"]
    assert freeze["selection_used_outcome_labels"] is False
    assert freeze["status"] == "frozen_before_outcome_modeling"


def test_mechanism_results_match_held_out_predictions():
    results = pd.read_csv("results/mechanism_panel/lodo_results.csv")
    predictions = pd.read_csv("results/mechanism_panel/predictions.csv")

    assert len(results) == 30
    assert len(predictions) == 3 * 1339
    assert predictions.groupby("model")["sample_id"].nunique().eq(1339).all()

    recomputed = {
        (model, cohort): roc_auc_score(frame["y_true"], frame["y_prob"])
        for (model, cohort), frame in predictions.groupby(["model", "cohort"])
    }
    for row in results.itertuples(index=False):
        assert abs(row.auc - recomputed[(row.model, row.cohort)]) < 1e-12

    means = results.groupby("model")["auc"].mean()
    assert abs(means["mechanism_only"] - 0.569408802057239) < 1e-12
    assert abs(means["parent_species_only"] - 0.6555342142066598) < 1e-12
    assert abs(means["mechanism_plus_parent"] - 0.6552173873703397) < 1e-12
