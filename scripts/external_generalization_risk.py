#!/usr/bin/env python3
"""Estimate external species-model AUC using target data but no target labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from generalization_risk import (
    build_observations,
    historical_feature,
    prediction_features,
)
from score_external_species import harmonize_gmrepo


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "generalization_risk"


def external_shift_features() -> dict[str, float]:
    species = pd.read_csv(ROOT / "data/processed/species_filtered.csv")
    metadata = pd.read_csv(ROOT / "data/processed/metadata_clean.csv")
    retained = [column for column in species.columns if column != "sample_id"]
    source_ids = set(metadata.loc[metadata.label.isin([0, 1]), "sample_id"])
    source = species[species.sample_id.isin(source_ids)].set_index("sample_id")[retained]

    profiles = pd.read_csv(ROOT / "results/external_cohort/gmrepo_species_long.csv.gz")
    external = harmonize_gmrepo(profiles, retained)
    frozen_runs = set(
        pd.read_csv(
            ROOT / "results/external_cohort/manifest.csv", usecols=["run_accession"]
        ).run_accession
    )
    if set(external.index) != frozen_runs:
        raise ValueError("External features do not cover the exact frozen run set")

    source_x = source.to_numpy(float)
    target_x = external.to_numpy(float)
    source_mean = source_x.mean(axis=0)
    source_sd = source_x.std(axis=0) + 1e-8
    standardized = np.abs((target_x.mean(axis=0) - source_mean) / source_sd)
    source_prev = (source_x > -6).mean(axis=0)
    target_prev = (target_x > -6).mean(axis=0)

    domain_x = np.vstack([source_x, target_x])
    domain_y = np.concatenate([np.zeros(len(source_x)), np.ones(len(target_x))])
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=20260808)
    domain_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000),
    )
    domain_p = cross_val_predict(
        domain_model, domain_x, domain_y, cv=folds, method="predict_proba"
    )[:, 1]
    return {
        "species_mean_shift": float(standardized.mean()),
        "species_max_shift": float(standardized.max()),
        "species_prevalence_shift": float(np.abs(target_prev - source_prev).mean()),
        "domain_classifier_auc": float(roc_auc_score(domain_y, domain_p)),
    }


def main() -> None:
    observations = build_observations()
    observations["historical_auc"] = [
        historical_feature(observations, row.to_frame().T, row.cohort)[0]
        for _, row in observations.iterrows()
    ]
    numeric = [
        "historical_auc", "n_target", "mean_probability", "sd_probability",
        "mean_confidence", "mean_entropy", "fraction_extreme",
        "species_mean_shift", "species_max_shift", "species_prevalence_shift",
        "domain_classifier_auc",
    ]
    design = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("model", OneHotEncoder(handle_unknown="ignore"), ["model"]),
    ])
    estimator = make_pipeline(design, Ridge(alpha=10.0))
    estimator.fit(observations[numeric + ["model"]], observations.observed_auc)

    # Only the accession and probability columns are read. Target outcome and
    # age labels are intentionally not available to this estimate.
    probabilities = pd.read_csv(
        ROOT / "results/external_cohort/predictions.csv",
        usecols=["run_accession", "y_prob"],
    )
    features = {"model": "species_rf"}
    features.update(prediction_features(probabilities))
    features.update(external_shift_features())
    features["historical_auc"] = float(
        observations.loc[observations.model == "species_rf", "observed_auc"].mean()
    )
    row = pd.DataFrame([features])
    estimate = float(np.clip(estimator.predict(row[numeric + ["model"]])[0], 0, 1))
    artifact = {
        "status": "completed_without_external_outcome_labels",
        "external_labels_accessed": False,
        "model": "species_rf",
        "historical_model_mean_estimate": features["historical_auc"],
        "unlabeled_risk_estimate": estimate,
        "features": {key: float(features[key]) for key in numeric},
        "interpretation": (
            "Prospective estimate from the internally developed risk model; "
            "not a confidence interval and not used to alter external scoring."
        ),
    }
    (OUT / "external_risk_estimate.json").write_text(
        json.dumps(artifact, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
