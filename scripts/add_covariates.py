"""
add_covariates.py — Add age, sex, BMI as covariates to the classification pipeline.

This script:
1. Audits how many samples have non-null age, gender, BMI
2. Runs species-only LODO and species+covariates LODO with per-fold
   covariate imputation (no cross-fold leakage)
3. Reports whether covariates change LODO performance

Note: this overlaps with confounder_adjustment.py, which additionally tests
residualization and XGBoost. We keep both scripts: this one as a focused
"add covariates as features" check with a paired t-test, and
confounder_adjustment.py as the broader four-cell direct/residualize x
RF/XGB grid. Both use train-fold-only imputation.

Authors: Alex Velazquez, Rachel Selbrede
"""

import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from lodo_cv import get_lodo_splits, run_lodo_cv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def make_rf():
    return RandomForestClassifier(
        n_estimators=500, max_features="sqrt",
        min_samples_leaf=5, n_jobs=-1, random_state=42,
        class_weight="balanced",
    )


def run_species_plus_cov_lodo(X_species, y, meta, md_full):
    """Per-fold-imputed LODO for species + (age, gender, BMI)."""
    results = {"cohort": [], "auc": []}
    for cohort, train_idx, test_idx in get_lodo_splits(meta):
        X_tr = X_species.iloc[train_idx].copy()
        X_te = X_species.iloc[test_idx].copy()
        md_tr = md_full.iloc[train_idx].copy()
        md_te = md_full.iloc[test_idx].copy()

        for col in ["age", "BMI"]:
            if col in md_tr.columns:
                med = md_tr[col].median()
                X_tr[col] = md_tr[col].fillna(med).values
                X_te[col] = md_te[col].fillna(med).values
        if "gender" in md_tr.columns:
            mode = md_tr["gender"].mode()
            mode = mode.iloc[0] if len(mode) > 0 else "female"
            X_tr["gender_num"] = (md_tr["gender"].fillna(mode) == "male").astype(float).values
            X_te["gender_num"] = (md_te["gender"].fillna(mode) == "male").astype(float).values

        model = make_rf()
        model.fit(X_tr, y.iloc[train_idx])
        y_prob = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y.iloc[test_idx], y_prob)
        results["cohort"].append(cohort)
        results["auc"].append(auc)
        print(f"  {cohort:25s}  AUC={auc:.3f}")
    results["mean_auc"] = float(np.mean(results["auc"]))
    results["std_auc"] = float(np.std(results["auc"]))
    print(f"\n  Mean AUC: {results['mean_auc']:.3f} +/- {results['std_auc']:.3f}")
    return results


def main():
    print("=" * 60)
    print("CRC Metagenomics — Confounder Adjustment Analysis")
    print("=" * 60)

    species = pd.read_csv("data/processed/species_filtered.csv")
    metadata = pd.read_csv("data/processed/metadata_clean.csv")

    print("\n--- Covariate Completeness Audit ---")
    print(f"Total samples in metadata: {len(metadata)}")
    for col in ["age", "gender", "BMI"]:
        if col in metadata.columns:
            n_valid = metadata[col].notna().sum()
            n_missing = metadata[col].isna().sum()
            pct = 100 * n_valid / len(metadata)
            print(f"  {col:8s}: {n_valid:4d} valid, {n_missing:4d} missing ({pct:.1f}% complete)")
            if col == "gender":
                print(f"            Values: {dict(metadata[col].value_counts())}")
        else:
            print(f"  {col:8s}: COLUMN NOT FOUND in metadata")

    merged = metadata.merge(species, on="sample_id", how="inner")
    feat_cols = [c for c in species.columns if c != "sample_id"]
    mask = merged["label"].isin([0, 1])
    df = merged[mask].reset_index(drop=True)

    X_species = df[feat_cols].reset_index(drop=True)
    y = df["label"].reset_index(drop=True)
    meta = df[["sample_id", "study_name", "study_condition", "label"]].reset_index(drop=True)
    md_full = df.reset_index(drop=True)

    print(f"\n  Species-only features:     {X_species.shape[1]}")
    print(f"  Samples:                   {len(y)} (CRC={int(y.sum())}, control={int((y==0).sum())})")

    print("\n" + "=" * 60)
    print("=== LODO: Species Only (baseline) ===")
    res_species = run_lodo_cv(make_rf, X_species, y, meta)

    print("\n" + "=" * 60)
    print("=== LODO: Species + Clinical Covariates (per-fold imputed) ===")
    res_cov = run_species_plus_cov_lodo(X_species, y, meta, md_full)

    print("\n" + "=" * 60)
    print("=== COMPARISON ===")
    print(f"  Species only:        AUC = {res_species['mean_auc']:.3f} +/- {res_species['std_auc']:.3f}")
    print(f"  Species + covariates: AUC = {res_cov['mean_auc']:.3f} +/- {res_cov['std_auc']:.3f}")

    from scipy import stats
    t, p = stats.ttest_rel(res_species["auc"], res_cov["auc"])
    print(f"\n  Paired t-test: t={t:.3f}, p={p:.4f}")
    if p < 0.05:
        print("  Result: Covariates SIGNIFICANTLY change performance")
    else:
        print("  Result: No significant difference (covariates do not change performance)")
    print("  Interpretation: If not significant, species features already capture")
    print("  the same variance as age/sex/BMI, which is expected for tree-based models.")

    print("\n  Per-cohort AUC differences (species+cov minus species-only):")
    for i, cohort in enumerate(res_species["cohort"]):
        diff = res_cov["auc"][i] - res_species["auc"][i]
        print(f"    {cohort:25s}  {diff:+.3f}")

    os.makedirs("results", exist_ok=True)
    pd.DataFrame({
        "cohort": res_species["cohort"],
        "species_auc": res_species["auc"],
        "species_cov_auc": res_cov["auc"],
        "difference": [res_cov["auc"][i] - res_species["auc"][i] for i in range(len(res_species["auc"]))],
    }).to_csv("results/covariate_comparison.csv", index=False)
    print("\n  Saved results/covariate_comparison.csv")

    print("\n" + "=" * 60)
    print("Done. Suggested Methods note:")
    print('  "We assessed whether clinical covariates (age, sex, BMI)')
    print('   improved CRC classification when added to species features.')
    print('   Covariate medians/modes were computed per LODO fold using')
    print('   training-cohort samples only.')
    print(f'   The addition of covariates did {"" if p < 0.05 else "not "}significantly')
    print(f'   alter LODO performance (paired t-test, p={p:.3f}), consistent with')
    print('   prior findings that tree-based models on taxonomic profiles')
    print('   implicitly capture demographic-associated variance."')
    print("=" * 60)


if __name__ == "__main__":
    main()
