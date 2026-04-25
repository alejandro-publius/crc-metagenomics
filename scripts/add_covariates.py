"""
add_covariates.py — Add age, sex, BMI as covariates to the classification pipeline.

This script:
1. Checks how many samples have non-null age, gender, BMI
2. Creates an augmented feature matrix with clinical covariates
3. Re-runs baseline LODO with and without covariates
4. Reports whether covariates improve CRC classification

Authors: Alex Velazquez, Rachel Selbrede
"""

import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from lodo_cv import run_lodo_cv
from sklearn.ensemble import RandomForestClassifier


def main():
    print("=" * 60)
    print("CRC Metagenomics — Confounder Adjustment Analysis")
    print("=" * 60)

    # Load data
    species = pd.read_csv("data/processed/species_filtered.csv")
    metadata = pd.read_csv("data/processed/metadata_clean.csv")

    # ── Step 1: Audit covariate completeness ──
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

    # ── Step 2: Encode covariates ──
    print("\n--- Encoding Covariates ---")

    # Merge species + metadata
    merged = metadata.merge(species, on="sample_id", how="inner")
    feat_cols = [c for c in species.columns if c != "sample_id"]

    # Only keep CRC and control for this analysis
    mask = merged["label"].isin([0, 1])
    df = merged[mask].reset_index(drop=True)

    # Encode gender as numeric (0/1)
    if "gender" in df.columns:
        df["gender_encoded"] = df["gender"].map({"female": 0, "male": 1, "F": 0, "M": 1})
        n_encoded = df["gender_encoded"].notna().sum()
        print(f"  Gender encoded: {n_encoded} samples")
    else:
        df["gender_encoded"] = np.nan

    # Fill missing covariates with median (standard practice for tree-based models)
    covariate_cols = []
    for col, source in [("age", "age"), ("gender_num", "gender_encoded"), ("BMI", "BMI")]:
        if source in df.columns:
            values = df[source].copy()
            n_missing = values.isna().sum()
            if n_missing > 0 and n_missing < len(values):
                median_val = values.median()
                values = values.fillna(median_val)
                print(f"  {col}: filled {n_missing} missing values with median ({median_val:.1f})")
            elif n_missing == len(values):
                print(f"  {col}: ALL values missing — skipping this covariate")
                continue
            else:
                print(f"  {col}: no missing values")
            df[col + "_cov"] = values
            covariate_cols.append(col + "_cov")

    print(f"\n  Usable covariates: {covariate_cols}")

    if len(covariate_cols) == 0:
        print("\n  ERROR: No usable covariates found. Check metadata columns.")
        print("  Available columns:", list(metadata.columns))
        return

    # ── Step 3: Build feature matrices ──
    # Species only (baseline)
    X_species = df[feat_cols].reset_index(drop=True)

    # Species + covariates
    X_species_cov = pd.concat([
        df[feat_cols].reset_index(drop=True),
        df[covariate_cols].reset_index(drop=True)
    ], axis=1)

    y = df["label"].reset_index(drop=True)
    meta = df[["sample_id", "study_name", "study_condition", "label"]].reset_index(drop=True)

    print(f"\n  Species-only features:     {X_species.shape[1]}")
    print(f"  Species + covariates:      {X_species_cov.shape[1]}")
    print(f"  Samples:                   {len(y)} (CRC={int(y.sum())}, control={int((y==0).sum())})")

    # ── Step 4: Run LODO with and without covariates ──
    def make_rf():
        return RandomForestClassifier(
            n_estimators=500, max_features="sqrt",
            min_samples_leaf=5, n_jobs=-1, random_state=42,
            class_weight="balanced"
        )

    print("\n" + "=" * 60)
    print("=== LODO: Species Only (baseline) ===")
    res_species = run_lodo_cv(make_rf, X_species, y, meta)

    print("\n" + "=" * 60)
    print("=== LODO: Species + Clinical Covariates ===")
    res_cov = run_lodo_cv(make_rf, X_species_cov, y, meta)

    # ── Step 5: Compare ──
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

    # ── Step 6: Per-cohort comparison ──
    print("\n  Per-cohort AUC differences (species+cov minus species-only):")
    for i, cohort in enumerate(res_species["cohort"]):
        diff = res_cov["auc"][i] - res_species["auc"][i]
        print(f"    {cohort:25s}  {diff:+.3f}")

    # ── Step 7: Save results ──
    os.makedirs("results", exist_ok=True)
    pd.DataFrame({
        "cohort": res_species["cohort"],
        "species_auc": res_species["auc"],
        "species_cov_auc": res_cov["auc"],
        "difference": [res_cov["auc"][i] - res_species["auc"][i] for i in range(len(res_species["auc"]))]
    }).to_csv("results/covariate_comparison.csv", index=False)
    print("\n  Saved results/covariate_comparison.csv")

    print("\n" + "=" * 60)
    print("Done. Include these results in the Methods section:")
    print('  "We assessed whether clinical covariates (age, sex, BMI)')
    print('   improved CRC classification when added to species features.')
    print(f'   The addition of covariates did {"" if p < 0.05 else "not "}significantly')
    print(f'   alter LODO performance (p={p:.3f}), consistent with prior')
    print('   findings that tree-based models on taxonomic profiles')
    print('   implicitly capture demographic-associated variance."')
    print("=" * 60)


if __name__ == "__main__":
    main()
