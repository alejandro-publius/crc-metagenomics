import numpy as np
import pandas as pd
import pytest

from scripts.generalization_risk import (
    historical_feature,
    outer_cohort_evaluation,
    prediction_features,
)


def test_prediction_features_do_not_use_labels():
    frame = pd.DataFrame({"y_prob": [0.1, 0.8], "y_true": [0, 1]})
    first = prediction_features(frame)
    frame["y_true"] = [1, 0]
    assert first == prediction_features(frame)


def test_outer_predictions_cover_every_model_cohort_pair():
    rows = []
    for cohort_i, cohort in enumerate(["a", "b", "c"]):
        for model_i, model in enumerate(["m1", "m2"]):
            rows.append({
                "cohort": cohort, "model": model,
                "observed_auc": 0.6 + cohort_i * 0.03 + model_i * 0.02,
                "n_target": 20, "mean_probability": 0.5,
                "sd_probability": 0.2, "mean_confidence": 0.3,
                "mean_entropy": 0.7, "fraction_extreme": 0.1,
                "species_mean_shift": 0.2, "species_max_shift": 1.0,
                "species_prevalence_shift": 0.1,
                "domain_classifier_auc": 0.7,
            })
    frame = pd.DataFrame(rows)
    predictions = outer_cohort_evaluation(frame)
    assert len(predictions) == len(frame)
    assert np.isfinite(predictions.unlabeled_risk_estimate).all()


# ---------------------------------------------------------------------------
# Leakage-safety regression tests for the label-free risk model.
#
# generalization_risk.py's whole claim to being "label-free" and honestly
# evaluated rests on two exclusions that outer_cohort_evaluation must apply
# together: (1) the outer split holds an entire target cohort out of
# training, and (2) within training, each meta-row's "historical_auc"
# feature excludes that row's own cohort so the meta-model never partly
# regresses a cohort's AUC on itself. A bug in either exclusion would let
# the risk model "cheat" and would invalidate the MAE 0.094 vs. 0.062
# comparison the README reports as a negative result. These tests are
# mutation-checked: reverting either exclusion in generalization_risk.py
# (see report) makes them fail.
# ---------------------------------------------------------------------------


def _obs_row(cohort, model, observed_auc):
    return {
        "cohort": cohort, "model": model, "observed_auc": observed_auc,
        "n_target": 20, "mean_probability": 0.5, "sd_probability": 0.2,
        "mean_confidence": 0.3, "mean_entropy": 0.7, "fraction_extreme": 0.1,
        "species_mean_shift": 0.2, "species_max_shift": 1.0,
        "species_prevalence_shift": 0.1, "domain_classifier_auc": 0.7,
    }


def test_historical_feature_excludes_the_named_cohort():
    train = pd.DataFrame([
        {"cohort": "X", "model": "m", "observed_auc": 0.9},
        {"cohort": "Y", "model": "m", "observed_auc": 0.5},
        {"cohort": "Z", "model": "m", "observed_auc": 0.5},
    ])
    row_x = train.iloc[[0]]

    excluding_x = historical_feature(train, row_x, excluded_cohort="X")
    assert excluding_x[0] == pytest.approx(0.5)  # mean of Y, Z only

    including_x = historical_feature(train, row_x, excluded_cohort=None)
    assert including_x[0] == pytest.approx((0.9 + 0.5 + 0.5) / 3)


def test_outer_cohort_evaluation_historical_estimate_excludes_held_cohort():
    """The held-out cohort's own observed_auc must never leak into its own
    historical_mean_estimate, even though that cohort's other-model rows
    remain in the frame for every other cohort's training fold."""
    observations = pd.DataFrame([
        _obs_row("A", "m", 0.9),
        _obs_row("B", "m", 0.5),
        _obs_row("C", "m", 0.5),
    ])
    predictions = outer_cohort_evaluation(observations)
    held_a = predictions.loc[predictions.cohort == "A"].iloc[0]
    # Correct: mean of B, C (0.5). A leak would pull this toward
    # mean(0.9, 0.5, 0.5) = 0.633.
    assert held_a.historical_mean_estimate == pytest.approx(0.5)
    assert held_a.historical_mean_estimate != pytest.approx((0.9 + 0.5 + 0.5) / 3)
