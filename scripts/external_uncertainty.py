#!/usr/bin/env python3
"""Quantify uncertainty, age-stratum performance, and profile coverage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from score_external_species import normalize_species_name


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "external_cohort"
N_BOOTSTRAP = 10_000
SEED = 20260808


def metric_pair(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    return roc_auc_score(y, probability), average_precision_score(y, probability)


def stratified_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample cases and controls separately, preserving class counts."""
    groups = [np.flatnonzero(y == label) for label in (0, 1)]
    return np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])


def summarize_scope(frame: pd.DataFrame, replicates: pd.DataFrame, scope: str) -> dict:
    y = frame.label.to_numpy(int)
    p = frame.y_prob.to_numpy(float)
    auc, average_precision = metric_pair(y, p)
    predicted = p >= 0.5
    return {
        "scope": scope,
        "n": len(frame),
        "n_crc": int(y.sum()),
        "n_control": int((y == 0).sum()),
        "auc": auc,
        "auc_ci_low": replicates[f"{scope}_auc"].quantile(0.025),
        "auc_ci_high": replicates[f"{scope}_auc"].quantile(0.975),
        "average_precision": average_precision,
        "average_precision_ci_low": replicates[f"{scope}_ap"].quantile(0.025),
        "average_precision_ci_high": replicates[f"{scope}_ap"].quantile(0.975),
        "brier_score": brier_score_loss(y, p),
        "sensitivity_at_0_5": float(predicted[y == 1].mean()),
        "specificity_at_0_5": float((~predicted[y == 0]).mean()),
    }


def bootstrap_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return deterministic label-stratified replicates for all scopes."""
    rng = np.random.default_rng(SEED)
    scopes = {
        "overall": predictions,
        "older": predictions[predictions.age_group == "older"],
        "younger": predictions[predictions.age_group == "younger"],
    }
    rows = []
    for replicate in range(N_BOOTSTRAP):
        row = {"replicate": replicate}
        for name, frame in scopes.items():
            y = frame.label.to_numpy(int)
            p = frame.y_prob.to_numpy(float)
            idx = stratified_indices(y, rng)
            row[f"{name}_auc"], row[f"{name}_ap"] = metric_pair(y[idx], p[idx])
        row["older_minus_younger_auc"] = row["older_auc"] - row["younger_auc"]
        rows.append(row)
    return pd.DataFrame(rows)


def profile_coverage() -> tuple[pd.DataFrame, dict]:
    profiles = pd.read_csv(OUT / "gmrepo_species_long.csv.gz")
    retained = [
        column
        for column in pd.read_csv(ROOT / "data/processed/species_filtered.csv", nrows=1).columns
        if column != "sample_id"
    ]
    locked_names = {
        normalize_species_name(feature.rsplit("s__", 1)[-1]) for feature in retained
    }
    profiles["matched_locked_feature"] = profiles.scientific_name.map(
        normalize_species_name
    ).isin(locked_names)
    coverage = profiles.groupby("run_accession").apply(
        lambda group: pd.Series({
            "matched_abundance_percent": group.loc[
                group.matched_locked_feature, "relative_abundance"
            ].sum(),
            "matched_detected_species": int(group.matched_locked_feature.sum()),
            "total_detected_species": len(group),
        }),
        include_groups=False,
    ).reset_index()
    summary = {
        "locked_features": len(retained),
        "locked_features_observed_any_sample": len(
            set(profiles.loc[profiles.matched_locked_feature, "scientific_name"].map(
                normalize_species_name
            ))
        ),
        "mean_matched_abundance_percent": coverage.matched_abundance_percent.mean(),
        "median_matched_abundance_percent": coverage.matched_abundance_percent.median(),
        "min_matched_abundance_percent": coverage.matched_abundance_percent.min(),
        "max_matched_abundance_percent": coverage.matched_abundance_percent.max(),
        "mapping_rule": "normalized exact terminal species name; no synonym guessing",
    }
    return coverage, summary


def main() -> None:
    predictions = pd.read_csv(OUT / "predictions.csv")
    replicates = bootstrap_predictions(predictions)
    metrics = pd.DataFrame([
        summarize_scope(predictions, replicates, "overall"),
        summarize_scope(predictions[predictions.age_group == "older"], replicates, "older"),
        summarize_scope(predictions[predictions.age_group == "younger"], replicates, "younger"),
    ])
    difference = replicates.older_minus_younger_auc
    age_comparison = {
        "older_minus_younger_auc": float(
            metrics.loc[metrics.scope == "older", "auc"].iloc[0]
            - metrics.loc[metrics.scope == "younger", "auc"].iloc[0]
        ),
        "ci_low": float(difference.quantile(0.025)),
        "ci_high": float(difference.quantile(0.975)),
        "bootstrap_two_sided_p": float(
            min(1.0, 2 * min((difference <= 0).mean(), (difference >= 0).mean()))
        ),
        "bootstrap_replicates": N_BOOTSTRAP,
        "resampling": "within outcome class and age stratum",
    }
    coverage, coverage_summary = profile_coverage()

    metrics.to_csv(OUT / "uncertainty_metrics.csv", index=False)
    replicates.to_csv(OUT / "bootstrap_replicates.csv.gz", index=False, compression="gzip")
    coverage.to_csv(OUT / "profile_coverage.csv", index=False)
    (OUT / "age_comparison.json").write_text(json.dumps(age_comparison, indent=2) + "\n")
    (OUT / "profile_coverage_summary.json").write_text(
        json.dumps(coverage_summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
