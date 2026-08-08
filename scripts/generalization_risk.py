#!/usr/bin/env python3
"""Estimate labeled target performance from signals available before labels.

The unit of evaluation is model x held-out cohort. Risk features use only the
model's predictions and unlabeled species composition in the target cohort.
The risk estimator itself is evaluated by leaving an entire cohort out.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "generalization_risk"


def prediction_tables() -> list[pd.DataFrame]:
    files = {
        "species_rf": "results/preds_species_rf.csv",
        "joint_rf": "results/preds_joint_rf.csv",
        "joint_xgb": "results/preds_joint_xgb.csv",
        "pathway_rf": "results/preds_bio_pathway_rf.csv",
        "stratified_joint_rf": "results/preds_stratified_joint_rf.csv",
        "stratified_joint_xgb": "results/preds_stratified_joint_xgb.csv",
        "gene_family_enet": "results/preds_gene_family_elastic_net.csv",
    }
    tables = []
    for model, path in files.items():
        frame = pd.read_csv(ROOT / path)
        frame["model"] = model
        tables.append(frame)

    mechanism = pd.read_csv(ROOT / "results/mechanism_panel/predictions.csv")
    tables.extend(g for _, g in mechanism.groupby("model", sort=False))
    corrected = pd.read_csv(ROOT / "results/species_aware_correction/predictions.csv")
    corrected = corrected[corrected.model.isin(["species_source_only", "stratified_source_only"])]
    tables.extend(g for _, g in corrected.groupby("model", sort=False))
    return tables


def prediction_features(frame: pd.DataFrame) -> dict[str, float]:
    p = np.clip(frame.y_prob.to_numpy(float), 1e-6, 1 - 1e-6)
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return {
        "n_target": len(frame),
        "mean_probability": float(p.mean()),
        "sd_probability": float(p.std()),
        "mean_confidence": float(np.mean(2 * np.abs(p - 0.5))),
        "mean_entropy": float(entropy.mean()),
        "fraction_extreme": float(np.mean((p < 0.1) | (p > 0.9))),
    }


def species_shift_features() -> pd.DataFrame:
    species = pd.read_csv(ROOT / "data/processed/species_filtered.csv")
    metadata = pd.read_csv(ROOT / "data/processed/metadata_clean.csv")
    metadata = metadata[metadata.label.isin([0, 1])][["sample_id", "study_name"]]
    data = metadata.merge(species, on="sample_id", how="inner")
    feature_cols = [c for c in species.columns if c != "sample_id"]
    # The committed table is already log10 transformed with -6 as the
    # zero/pseudocount floor; applying log1p again would be invalid.
    X_all = data[feature_cols].to_numpy(float)
    rows = []
    for cohort in sorted(data.study_name.unique()):
        target = data.study_name.eq(cohort).to_numpy()
        source = ~target
        source_mean = X_all[source].mean(axis=0)
        source_sd = X_all[source].std(axis=0) + 1e-8
        standardized = np.abs((X_all[target].mean(axis=0) - source_mean) / source_sd)
        source_prev = (X_all[source] > -6).mean(axis=0)
        target_prev = (X_all[target] > -6).mean(axis=0)

        domain_y = target.astype(int)
        folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=20260808)
        domain_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000),
        )
        domain_p = cross_val_predict(
            domain_model, X_all, domain_y, cv=folds, method="predict_proba"
        )[:, 1]
        rows.append({
            "cohort": cohort,
            "species_mean_shift": float(standardized.mean()),
            "species_max_shift": float(standardized.max()),
            "species_prevalence_shift": float(np.abs(target_prev - source_prev).mean()),
            "domain_classifier_auc": float(roc_auc_score(domain_y, domain_p)),
        })
    return pd.DataFrame(rows)


def build_observations() -> pd.DataFrame:
    shift = species_shift_features()
    rows = []
    for table in prediction_tables():
        model = str(table.model.iloc[0])
        for cohort, fold in table.groupby("cohort"):
            if fold.y_true.nunique() < 2:
                continue
            row = {"model": model, "cohort": cohort,
                   "observed_auc": roc_auc_score(fold.y_true, fold.y_prob)}
            row.update(prediction_features(fold))
            rows.append(row)
    observations = pd.DataFrame(rows).merge(shift, on="cohort", how="left")
    if observations.groupby("cohort").size().nunique() != 1:
        raise ValueError("Every cohort must have the same frozen model set")
    return observations.sort_values(["cohort", "model"]).reset_index(drop=True)


def historical_feature(train: pd.DataFrame, rows: pd.DataFrame,
                       excluded_cohort: str | None = None) -> np.ndarray:
    values = []
    global_mean = float(train.observed_auc.mean())
    for row in rows.itertuples(index=False):
        eligible = train[train.model == row.model]
        if excluded_cohort is not None:
            eligible = eligible[eligible.cohort != excluded_cohort]
        values.append(float(eligible.observed_auc.mean()) if len(eligible) else global_mean)
    return np.asarray(values)


def outer_cohort_evaluation(observations: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "historical_auc", "n_target", "mean_probability", "sd_probability",
        "mean_confidence", "mean_entropy", "fraction_extreme",
        "species_mean_shift", "species_max_shift", "species_prevalence_shift",
        "domain_classifier_auc",
    ]
    predictions = []
    for held in sorted(observations.cohort.unique()):
        train = observations[observations.cohort != held].copy()
        test = observations[observations.cohort == held].copy()

        # For each meta-training row, historical performance excludes both
        # that row's cohort and the outer held-out cohort.
        train["historical_auc"] = [
            historical_feature(train, row.to_frame().T, row.cohort)[0]
            for _, row in train.iterrows()
        ]
        test["historical_auc"] = historical_feature(train, test)

        design = ColumnTransformer([
            ("numeric", StandardScaler(), numeric),
            ("model", OneHotEncoder(handle_unknown="ignore"), ["model"]),
        ])
        estimator = make_pipeline(design, Ridge(alpha=10.0))
        estimator.fit(train[numeric + ["model"]], train.observed_auc)
        estimated = np.clip(estimator.predict(test[numeric + ["model"]]), 0, 1)
        for (_, row), estimate in zip(test.iterrows(), estimated):
            predictions.append({
                "cohort": held,
                "model": row.model,
                "observed_auc": row.observed_auc,
                "historical_mean_estimate": row.historical_auc,
                "unlabeled_risk_estimate": estimate,
            })
    return pd.DataFrame(predictions)


def metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, column in [
        ("historical_model_mean", "historical_mean_estimate"),
        ("unlabeled_risk_model", "unlabeled_risk_estimate"),
    ]:
        y = predictions.observed_auc
        p = predictions[column]
        rows.append({
            "method": method,
            "mae": mean_absolute_error(y, p),
            "rmse": mean_squared_error(y, p) ** 0.5,
            "spearman_r": spearmanr(y, p).statistic,
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    observations = build_observations()
    predictions = outer_cohort_evaluation(observations)
    metrics = metric_rows(predictions)
    observations.to_csv(OUT / "observations.csv", index=False)
    predictions.to_csv(OUT / "outer_cohort_predictions.csv", index=False)
    metrics.to_csv(OUT / "metrics.csv", index=False)

    by_cohort = predictions.assign(
        historical_error=lambda x: abs(x.observed_auc - x.historical_mean_estimate),
        risk_error=lambda x: abs(x.observed_auc - x.unlabeled_risk_estimate),
    ).groupby("cohort")[["historical_error", "risk_error"]].mean().reset_index()
    by_cohort.to_csv(OUT / "cohort_errors.csv", index=False)
    statistic, p_value = wilcoxon(
        by_cohort.risk_error, by_cohort.historical_error,
        alternative="less", zero_method="zsplit",
    )
    audit = {
        "status": "completed_internal_pilot",
        "unit": "model x target cohort",
        "n_cohorts": int(observations.cohort.nunique()),
        "n_models": int(observations.model.nunique()),
        "n_observations": len(observations),
        "outer_split": "leave-one-entire-target-cohort-out",
        "features_require_target_labels": False,
        "cohort_level_one_sided_wilcoxon_risk_better_p": float(p_value),
        "interpretation_boundary": (
            "Internal method-development evidence only; an untouched external "
            "cohort is required before claiming prospective risk prediction."
        ),
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
