"""
external_validation.py — Test the model on truly held-out cohorts.

What this script does:
1. Picks 2 cohorts to hold out completely (YuJ_2015 and ZellerG_2014)
2. Trains a Random Forest model on ONLY the other 5 cohorts
3. Tests on each held-out cohort separately
4. Saves the results

Why this is different from LODO:
LODO rotates through all 7 cohorts, leaving one out at a time. Every cohort gets
to be the test set once and the training set 6 times. External validation is
stricter — the held-out cohorts never get used for training. They're truly unseen.

This is what reviewers want for generalization claims.

Authors: Alex Velazquez, Rachel Selbrede

Usage:
    Run from the project root:
        python3 scripts/external_validation.py

Output:
    results/external_validation.csv
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def main():
    # ── Section 1: Load the data ──
    # Same data files we use everywhere else in the project.
    # species_filtered.csv has the bacterial abundances per sample.
    # metadata_clean.csv has each sample's cohort, condition, and label.
    print("=" * 60)
    print("CRC Metagenomics — External Validation")
    print("=" * 60)

    species = pd.read_csv("data/processed/species_filtered.csv")
    metadata = pd.read_csv("data/processed/metadata_clean.csv")

    # Merge species with metadata so each row has both abundance + cohort info
    merged = metadata.merge(species, on="sample_id", how="inner")

    # Keep only CRC (label=1) and control (label=0). Drop adenoma for this analysis.
    merged = merged[merged["label"].isin([0, 1])].reset_index(drop=True)

    # ── Section 2: Define which cohorts are held out ──
    # We pick the two largest cohorts as our external validation set.
    # Larger cohorts give us more confident AUC estimates.
    held_out_cohorts = ["YuJ_2015", "ZellerG_2014"]
    training_cohorts = [c for c in merged["study_name"].unique() if c not in held_out_cohorts]

    print(f"\nTraining cohorts ({len(training_cohorts)}): {training_cohorts}")
    print(f"Held-out cohorts ({len(held_out_cohorts)}): {held_out_cohorts}")

    # ── Section 3: Split the data into training set and held-out set ──
    # train_mask is True for rows from training cohorts, False otherwise.
    # We use this mask to slice the merged dataframe into two groups.
    train_mask = merged["study_name"].isin(training_cohorts)
    test_mask = merged["study_name"].isin(held_out_cohorts)

    # Get the species feature columns (every column that came from species_filtered)
    feat_cols = [c for c in species.columns if c != "sample_id"]

    # Build training matrices
    X_train = merged.loc[train_mask, feat_cols].reset_index(drop=True)
    y_train = merged.loc[train_mask, "label"].reset_index(drop=True)

    # Build test matrices
    X_test = merged.loc[test_mask, feat_cols].reset_index(drop=True)
    y_test = merged.loc[test_mask, "label"].reset_index(drop=True)
    test_cohorts = merged.loc[test_mask, "study_name"].reset_index(drop=True)

    print(f"\nTraining samples: {len(X_train)} (CRC={int(y_train.sum())}, control={int((y_train==0).sum())})")
    print(f"Held-out samples: {len(X_test)} (CRC={int(y_test.sum())}, control={int((y_test==0).sum())})")
    print(f"Number of features: {X_train.shape[1]}")

    # ── Section 4: Train the model ──
    # We use the same Random Forest configuration as the rest of the project,
    # so the comparison is apples-to-apples.
    print("\n--- Training Random Forest on 5 cohorts ---")

    model = RandomForestClassifier(
        n_estimators=500,           # 500 trees
        max_features="sqrt",        # use sqrt(n_features) per split
        min_samples_leaf=5,         # require at least 5 samples per leaf
        n_jobs=-1,                  # use all CPU cores
        random_state=42,            # reproducibility
        class_weight="balanced"     # handle class imbalance
    )

    model.fit(X_train, y_train)
    print("  Done training.")

    # ── Section 5: Test on each held-out cohort separately ──
    # We don't just compute one AUC across all held-out samples.
    # We compute AUC on each held-out cohort independently to see
    # whether the model generalizes consistently or only to one of them.
    print("\n--- Testing on held-out cohorts ---")

    results = []
    for cohort in held_out_cohorts:
        # Get just the samples from this specific cohort
        cohort_mask = (test_cohorts == cohort)
        X_cohort = X_test[cohort_mask]
        y_cohort = y_test[cohort_mask]

        # Predict probabilities for class 1 (CRC)
        y_pred_proba = model.predict_proba(X_cohort)[:, 1]

        # Compute AUC
        auc = roc_auc_score(y_cohort, y_pred_proba)

        n = len(y_cohort)
        n_crc = int(y_cohort.sum())
        n_ctrl = int((y_cohort == 0).sum())

        print(f"  {cohort}: AUC = {auc:.3f}  (n={n}, CRC={n_crc}, control={n_ctrl})")

        results.append({
            "cohort": cohort,
            "auc": auc,
            "n_samples": n,
            "n_crc": n_crc,
            "n_control": n_ctrl
        })

    # ── Section 6: Compute overall AUC across all held-out samples ──
    # This treats both held-out cohorts as one combined test set.
    y_pred_all = model.predict_proba(X_test)[:, 1]
    overall_auc = roc_auc_score(y_test, y_pred_all)
    print(f"\n  Combined held-out AUC: {overall_auc:.3f}  (n={len(y_test)})")

    # ── Section 7: Save the results ──
    os.makedirs("results", exist_ok=True)
    results_df = pd.DataFrame(results)

    # Add a row for the combined result
    combined_row = pd.DataFrame([{
        "cohort": "COMBINED",
        "auc": overall_auc,
        "n_samples": len(y_test),
        "n_crc": int(y_test.sum()),
        "n_control": int((y_test == 0).sum())
    }])
    results_df = pd.concat([results_df, combined_row], ignore_index=True)

    results_df.to_csv("results/external_validation.csv", index=False)
    print("\n  Saved results/external_validation.csv")

    # ── Section 8: Print a Methods sentence Rachel can paste into the paper ──
    print("\n" + "=" * 60)
    print("Methods sentence to paste into the manuscript:")
    print('  "To assess generalization to unseen cohorts, we held out')
    print('   the two largest cohorts (YuJ_2015 and ZellerG_2014) and')
    print('   trained the model on the remaining five. The model achieved')
    print(f'   AUC {results[0]["auc"]:.3f} on YuJ_2015 and AUC {results[1]["auc"]:.3f} on')
    print(f'   ZellerG_2014, with a combined external AUC of {overall_auc:.3f}.')
    print('   These results indicate the model generalizes to truly unseen')
    print('   patient populations."')
    print("=" * 60)


if __name__ == "__main__":
    main()
