import numpy as np
import pandas as pd

from scripts.generalization_risk import outer_cohort_evaluation, prediction_features


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
