"""Train joint species + stratified pathway LODO classifier.

This is the gene/species-resolved-pathway variant the project has not yet
exploited. Uses HUMAnN species-stratified pathway abundances pulled from
cMD 3.20 (data/raw/pathway_stratified_v320.csv) alongside the existing
filtered species table.

Output:
    results/stratified_joint_results.csv     (per-fold RF + XGB AUC)
    results/preds_stratified_joint_rf.csv    (per-sample LODO predictions, RF)
    results/preds_stratified_joint_xgb.csv   (per-sample LODO predictions, XGB)

Independent of any external lab / non-Pasolli method.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from crc_lodo_bench.filters import per_fold_pathway_filter  # noqa: E402
from crc_lodo_bench.lodo import run_lodo_cv  # noqa: E402


def load_data(prevalence_threshold: float, mean_threshold: float) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str], list[str]]:
    """Load species + stratified pathway features merged on sample_id.

    Returns (X, y, metadata, species_cols, pathway_cols).
    """
    species = pd.read_csv(REPO / "data/processed/species_filtered.csv")
    metadata = pd.read_csv(REPO / "data/processed/metadata_clean.csv")
    stratified = pd.read_csv(REPO / "data/raw/pathway_stratified_v320.csv")

    # Drop study_name from stratified — already in metadata
    if "study_name" in stratified.columns:
        stratified = stratified.drop(columns=["study_name"])

    species_cols = [c for c in species.columns if c not in ("sample_id",)]
    pathway_cols = [c for c in stratified.columns if c not in ("sample_id",)]

    print(f"Species features: {len(species_cols)}")
    print(f"Stratified pathway features (raw): {len(pathway_cols)}")

    # Apply same log10 transform as community pathway pipeline for stratified
    # (stratified abundances are also relative — use log10(x + 1e-6))
    stratified[pathway_cols] = np.log10(stratified[pathway_cols] + 1e-6).astype(np.float32)

    # Inner-merge on sample_id; metadata defines the case/control universe
    df = metadata.merge(species, on="sample_id", how="inner").merge(
        stratified, on="sample_id", how="inner"
    )
    df = df[df["study_condition"].isin(["CRC", "control"])].copy()
    df["label"] = (df["study_condition"] == "CRC").astype(int)
    print(f"After CRC/control filter + species join: {len(df)} samples")

    X = df[species_cols + pathway_cols].astype(np.float32)
    y = df["label"]
    md = df[["sample_id", "study_name", "country", "label"]].copy()
    return X, y, md, species_cols, pathway_cols


def make_rf_factory(seed: int):
    def _rf():
        return RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    return _rf


def make_xgb_factory(seed: int):
    def _xgb():
        return XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
            eval_metric="logloss",
            verbosity=0,
        )
    return _xgb


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prevalence-threshold", type=float, default=0.05,
                   help="Stratified features filter: min prevalence (default 5%, lower than the 10% used for community pathways because stratified features are sparser)")
    p.add_argument("--mean-threshold", type=float, default=1e-7,
                   help="Stratified features filter: min mean abundance (default 1e-7)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    X, y, md, species_cols, pathway_cols = load_data(
        args.prevalence_threshold, args.mean_threshold
    )

    print(f"Joint feature space before per-fold filter: {X.shape}")

    # Per-fold filter: filter pathway features only (species pass through)
    feat_filter = per_fold_pathway_filter(
        pathway_cols,
        passthrough_cols=species_cols,
        prevalence_threshold=args.prevalence_threshold,
        mean_threshold=args.mean_threshold,
    )

    results_rows = []
    pred_rows_rf = []
    pred_rows_xgb = []

    print("\nRunning RF...")
    rf_results = run_lodo_cv(
        make_rf_factory(args.seed), X, y, md,
        country_col="country",
        cohort_col="study_name",
        feature_filter_fn=feat_filter,
    )
    rf_per_cohort = pd.DataFrame(rf_results["per_cohort"])
    print(f"RF mean per-cohort AUC: {rf_per_cohort['auc'].mean():.4f}")
    for r in rf_results["per_cohort"]:
        results_rows.append({"model": "rf", **r})
    for r in rf_results["predictions"]:
        pred_rows_rf.append(r)

    print("\nRunning XGB...")
    xgb_results = run_lodo_cv(
        make_xgb_factory(args.seed), X, y, md,
        country_col="country",
        cohort_col="study_name",
        feature_filter_fn=feat_filter,
    )
    xgb_per_cohort = pd.DataFrame(xgb_results["per_cohort"])
    print(f"XGB mean per-cohort AUC: {xgb_per_cohort['auc'].mean():.4f}")
    for r in xgb_results["per_cohort"]:
        results_rows.append({"model": "xgb", **r})
    for r in xgb_results["predictions"]:
        pred_rows_xgb.append(r)

    # Save results
    out_dir = REPO / "results"
    pd.DataFrame(results_rows).to_csv(out_dir / "stratified_joint_results.csv", index=False)
    pd.DataFrame(pred_rows_rf).to_csv(out_dir / "preds_stratified_joint_rf.csv", index=False)
    pd.DataFrame(pred_rows_xgb).to_csv(out_dir / "preds_stratified_joint_xgb.csv", index=False)

    # Pooled AUC
    rf_preds = pd.DataFrame(pred_rows_rf)
    xgb_preds = pd.DataFrame(pred_rows_xgb)
    pooled_rf = roc_auc_score(rf_preds["y_true"], rf_preds["y_prob"])
    pooled_xgb = roc_auc_score(xgb_preds["y_true"], xgb_preds["y_prob"])
    print(f"\n=== POOLED AUC ===")
    print(f"  Species + stratified pathway RF:  {pooled_rf:.4f}")
    print(f"  Species + stratified pathway XGB: {pooled_xgb:.4f}")
    print(f"\nCompare to existing canonical:")
    print(f"  Species only RF:  0.7812  (results/bootstrap_ci.csv)")
    print(f"  Joint (community pathway) RF:  0.7561")
    print(f"  Joint (community pathway) XGB: 0.7660")


if __name__ == "__main__":
    main()
