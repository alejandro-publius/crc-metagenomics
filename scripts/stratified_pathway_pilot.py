"""Stratified (species|pathway) pathway pilot — explicit re-evaluation.

The decisions log (results/decisions_addendum.md, "Pathway feature set")
states that stratified taxon|pathway features were "considered but produce
>4000 highly redundant columns that did not improve AUC in pilot testing."
This script makes that pilot reproducible. It re-runs the comparison under
the same country-aware LODO protocol used for the joint model, and writes a
single CSV with the four head-to-head feature sets:

  1. species_only             - species_filtered.csv only
  2. unstrat_joint            - species + unstratified pathways (the current
                                joint model, mirrors train_joint.py)
  3. species_plus_stratified  - species + stratified pathways (the pilot)
  4. species_plus_all         - species + unstratified + stratified

Per-fold prevalence>=10% and mean>=1e-6 filters are applied to ALL pathway
columns (both unstratified and stratified) inside each LODO fold using
ONLY training-fold samples, identical to train_joint.py. Species pass
through (their filter is upstream).

Outputs:
  results/stratified_pathway_pilot.csv  columns: feature_set, cohort, auc,
                                                 n_features, mean_auc

Usage:
    python3 scripts/stratified_pathway_pilot.py \
        [--prevalence-threshold 0.10] [--mean-threshold 1e-6]
"""
import argparse
import gc
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.dirname(__file__))
from lodo_cv import get_lodo_splits  # noqa: E402

DEFAULT_PREVALENCE_THRESHOLD = 0.10
DEFAULT_MEAN_THRESHOLD = 1e-6
RANDOM_STATE = 42


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--prevalence-threshold", type=float,
                   default=DEFAULT_PREVALENCE_THRESHOLD,
                   help=("Minimum fraction of train-fold samples in which a "
                         "pathway must be non-zero to be retained "
                         f"(default {DEFAULT_PREVALENCE_THRESHOLD})."))
    p.add_argument("--mean-threshold", type=float,
                   default=DEFAULT_MEAN_THRESHOLD,
                   help=("Minimum train-fold mean abundance for a pathway "
                         "to be retained "
                         f"(default {DEFAULT_MEAN_THRESHOLD})."))
    return p.parse_args(argv)


def sanitize(c):
    return re.sub(r"[\[\]<>]", "_", c)


def load_data():
    """Load species, raw pathways (both unstrat + stratified), metadata.

    Returns merged X (sample-aligned to metadata) and the four column lists.
    """
    print("Loading metadata + species...")
    md = pd.read_csv("data/processed/metadata_clean.csv")
    sp = pd.read_csv("data/processed/species_filtered.csv")
    sp_cols = [c for c in sp.columns if c != "sample_id"]

    print(f"Loading pathway_abundance.csv (this is ~280MB; ~38k columns)...")
    # Read once. Use float32 for the matrix to save ~50% memory.
    pw_full = pd.read_csv("data/raw/pathway_abundance.csv")
    print(f"  pathway_abundance shape: {pw_full.shape}")

    feature_cols = [c for c in pw_full.columns if c != "sample_id"]
    for c in feature_cols:
        if pw_full[c].dtype == np.float64:
            pw_full[c] = pw_full[c].astype(np.float32)

    pw_unstrat_cols = [c for c in feature_cols if "|" not in c]
    pw_strat_cols = [c for c in feature_cols if "|" in c]
    print(f"  unstratified: {len(pw_unstrat_cols)},  "
          f"stratified: {len(pw_strat_cols)}")

    # Inner-merge to metadata
    mg = md.merge(sp, on="sample_id", how="inner")
    mg = mg.merge(pw_full, on="sample_id", how="inner",
                  suffixes=("_sp", "_pw"))
    print(f"After merge: {mg.shape[0]} samples, {mg.shape[1]} columns")

    mask = mg["label"].isin([0, 1])
    mg = mg.loc[mask].reset_index(drop=True)
    print(f"After label filter (binary): {len(mg)} samples")

    # Build aligned X and the column-name lists (sanitized)
    sp_sane = [sanitize(c) for c in sp_cols]
    unstrat_sane = [sanitize(c) for c in pw_unstrat_cols]
    strat_sane = [sanitize(c) for c in pw_strat_cols]

    rename_map = {}
    for c in sp_cols + pw_unstrat_cols + pw_strat_cols:
        rename_map[c] = sanitize(c)
    mg = mg.rename(columns=rename_map)

    # Disjointness check
    assert len(set(sp_sane) & set(unstrat_sane)) == 0, "sp/unstrat name clash"
    assert len(set(sp_sane) & set(strat_sane)) == 0,   "sp/strat name clash"
    assert len(set(unstrat_sane) & set(strat_sane)) == 0, "unstrat/strat clash"

    meta_cols = ["sample_id", "study_name", "study_condition", "label"]
    if "country" in mg.columns:
        meta_cols.append("country")
    meta = mg[meta_cols].reset_index(drop=True)
    y = mg["label"].astype(int).reset_index(drop=True)

    X = mg[sp_sane + unstrat_sane + strat_sane].reset_index(drop=True)

    # Cast stratified columns to float32 (re-cast may be needed post-merge)
    for c in strat_sane:
        if X[c].dtype == np.float64:
            X[c] = X[c].astype(np.float32)

    del mg, pw_full
    gc.collect()

    print(f"X shape: {X.shape}  (species={len(sp_sane)}, "
          f"unstrat={len(unstrat_sane)}, strat={len(strat_sane)})")
    return X, y, meta, sp_sane, unstrat_sane, strat_sane


def per_fold_pathway_filter(X_train_pw, pw_cols,
                            prevalence_threshold=DEFAULT_PREVALENCE_THRESHOLD,
                            mean_threshold=DEFAULT_MEAN_THRESHOLD):
    """Train-fold prevalence/mean filter, mirrors train_joint.py."""
    Xpw = X_train_pw[pw_cols]
    prev = (Xpw > 0).mean(axis=0)
    ma = Xpw.mean(axis=0)
    keep = [c for c in pw_cols
            if prev[c] >= prevalence_threshold and ma[c] >= mean_threshold]
    return keep


def make_rf():
    return RandomForestClassifier(
        n_estimators=500, max_features="sqrt", min_samples_leaf=5,
        n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced",
    )


def make_xgb():
    # tree_method='hist' keeps wall time per fold sane on wide feature
    # matrices (stratified pathway columns can be 35k+ wide).
    return XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
        eval_metric="logloss", n_jobs=-1, tree_method="hist",
    )


def run_one_feature_set(name, model_name, feature_builder, X, y, meta,
                        country_col):
    """LODO over all cohorts for one feature-set spec.

    feature_builder(X_train_only, X_test_only) -> (X_tr_sub, X_te_sub, n_feat)
    """
    print(f"\n--- {name} | {model_name} ---")
    rows = []
    for cohort, train_idx, test_idx, excluded in get_lodo_splits(
            meta, cohort_col="study_name", country_col=country_col):
        X_tr_full = X.iloc[train_idx]
        X_te_full = X.iloc[test_idx]

        X_tr, X_te, n_feat = feature_builder(X_tr_full, X_te_full)

        model = make_rf() if model_name == "rf" else make_xgb()
        model.fit(X_tr, y.iloc[train_idx])
        y_prob = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y.iloc[test_idx], y_prob)

        excl_str = f"  [excl: {sorted(excluded)}]" if excluded else ""
        print(f"  {cohort:18s} AUC={auc:.3f}  p={n_feat}  "
              f"n_tr={len(train_idx)}  n_te={len(test_idx)}{excl_str}")

        rows.append({
            "feature_set": name,
            "model": model_name,
            "cohort": cohort,
            "auc": auc,
            "n_features": n_feat,
        })
    mean_auc = float(np.mean([r["auc"] for r in rows]))
    for r in rows:
        r["mean_auc"] = mean_auc
    print(f"  mean LODO AUC = {mean_auc:.3f}")
    return rows


def main(argv=None):
    args = parse_args(argv)
    prevalence_threshold = args.prevalence_threshold
    mean_threshold = args.mean_threshold
    print(f"Filter thresholds: prevalence>={prevalence_threshold}, "
          f"mean>={mean_threshold}")

    X, y, meta, sp_cols, unstrat_cols, strat_cols = load_data()
    country_col = "country" if "country" in meta.columns else None
    print(f"country-aware LODO: {country_col is not None}")

    # -------- builders for each feature set --------
    def fb_species(X_tr, X_te):
        return X_tr[sp_cols], X_te[sp_cols], len(sp_cols)

    def fb_unstrat_joint(X_tr, X_te):
        keep = per_fold_pathway_filter(
            X_tr, unstrat_cols,
            prevalence_threshold=prevalence_threshold,
            mean_threshold=mean_threshold,
        )
        cols = sp_cols + keep
        return X_tr[cols], X_te[cols], len(cols)

    def fb_strat_joint(X_tr, X_te):
        keep = per_fold_pathway_filter(
            X_tr, strat_cols,
            prevalence_threshold=prevalence_threshold,
            mean_threshold=mean_threshold,
        )
        cols = sp_cols + keep
        return X_tr[cols], X_te[cols], len(cols)

    def fb_all(X_tr, X_te):
        keep_u = per_fold_pathway_filter(
            X_tr, unstrat_cols,
            prevalence_threshold=prevalence_threshold,
            mean_threshold=mean_threshold,
        )
        keep_s = per_fold_pathway_filter(
            X_tr, strat_cols,
            prevalence_threshold=prevalence_threshold,
            mean_threshold=mean_threshold,
        )
        cols = sp_cols + keep_u + keep_s
        return X_tr[cols], X_te[cols], len(cols)

    feature_sets = [
        ("species_only",            fb_species),
        ("unstrat_joint",           fb_unstrat_joint),
        ("species_plus_stratified", fb_strat_joint),
        ("species_plus_all",        fb_all),
    ]

    all_rows = []
    for name, builder in feature_sets:
        print(f"\n========== feature_set: {name} ==========")
        for model_name in ["rf", "xgb"]:
            all_rows.extend(
                run_one_feature_set(name, model_name, builder, X, y, meta,
                                    country_col)
            )

    df = pd.DataFrame(all_rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/stratified_pathway_pilot.csv", index=False)
    print(f"\nSaved results/stratified_pathway_pilot.csv ({len(df)} rows)")

    # Headline pivot
    pivot = (df.drop_duplicates(["feature_set", "model"])
               .pivot(index="feature_set", columns="model", values="mean_auc"))
    order = ["species_only", "unstrat_joint",
             "species_plus_stratified", "species_plus_all"]
    pivot = pivot.loc[order, ["rf", "xgb"]]
    print("\n=== Headline: mean LODO AUC by feature set ===")
    print(pivot.round(3).to_string())

    # Did stratified pathways help?
    print("\n=== Did stratified pathways rescue the joint model? ===")
    for model_name in ["rf", "xgb"]:
        species = pivot.loc["species_only", model_name]
        unstrat = pivot.loc["unstrat_joint", model_name]
        strat = pivot.loc["species_plus_stratified", model_name]
        both = pivot.loc["species_plus_all", model_name]
        print(f"\n  Model: {model_name.upper()}")
        print(f"    species_only             = {species:.3f}")
        print(f"    unstrat_joint            = {unstrat:.3f}  "
              f"(delta vs species: {unstrat-species:+.3f})")
        print(f"    species + stratified     = {strat:.3f}  "
              f"(delta vs species: {strat-species:+.3f})")
        print(f"    species + unstrat+strat  = {both:.3f}  "
              f"(delta vs species: {both-species:+.3f})")

        if strat > species + 0.005 or both > species + 0.005:
            verdict = "stratified pathways IMPROVED over species baseline"
        elif abs(strat - species) <= 0.005 and abs(both - species) <= 0.005:
            verdict = "stratified pathways NEUTRAL (within +/-0.005 of species)"
        else:
            verdict = "stratified pathways HURT vs species baseline"
        print(f"    -> verdict ({model_name}): {verdict}")

    # Per-fold feature counts (sanity check)
    print("\n=== n_features by feature set (mean across folds) ===")
    nf = (df.groupby(["feature_set", "model"])["n_features"].mean()
            .unstack("model").reindex(order))
    print(nf.round(1).to_string())


if __name__ == "__main__":
    main()
