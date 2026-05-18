"""RebalancedCV-style LODO for the adenoma tasks.

The adenoma LODO has class-balance shifts across the four held-out cohorts
(FengQ_2015=47, ZellerG_2014=42, ThomasAM_2018a=27, YachidaS_2019=67), which
can bias per-fold ROC. This script tests whether per-fold rebalancing
changes the qualitative conclusions.

For each adenoma task (H-vs-A and A-vs-CRC) and each classifier (RF, XGB),
runs leave-one-cohort-out (country-aware, so JPN<->JPN cross-training is
blocked even though only YachidaS_2019 is JPN among adenoma cohorts) under
four rebalancing strategies applied to the TRAINING fold only:

  - baseline           : no per-fold rebalancing (class_weight='balanced' /
                         scale_pos_weight only; matches adenoma_lodo.py)
  - random_under       : random under-sampling of the majority class in the
                         training fold to match the minority count
  - smote              : SMOTE oversampling of the minority class in the
                         training fold (uses imbalanced-learn if available;
                         otherwise a manual kNN-interpolation implementation
                         equivalent to Chawla et al. 2002)
  - rebalanced_weight  : per-sample weight = 1 / (cohort_size *
                         within-cohort class_prevalence). This is the
                         "RebalancedCV" weighting idea: each (cohort, class)
                         cell contributes equal aggregate weight.

The test fold is NEVER touched. AUC is computed on the held-out cohort with
its native class distribution.

Outputs:
  results/adenoma_rebalanced_lodo.csv      per-fold rows
  results/adenoma_rebalanced_summary.csv   mean LODO AUC per (task, strategy)

Usage:
    python3 scripts/rebalanced_adenoma_lodo.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

ADENOMA_COHORTS = ["FengQ_2015", "ZellerG_2014", "ThomasAM_2018a", "YachidaS_2019"]
RANDOM_STATE = 42

# Country map (from data/processed/metadata_clean.csv):
#   FengQ_2015 -> AUT, ZellerG_2014 -> FRA, ThomasAM_2018a -> ITA,
#   YachidaS_2019 -> JPN. All four adenoma cohorts are in distinct countries,
#   so country-aware LODO is a no-op for adenoma but we still pass it through
#   for consistency with the rest of the pipeline.

try:
    from imblearn.over_sampling import SMOTE as _SMOTE  # type: ignore
    _HAVE_IMBLEARN = True
except Exception:
    _HAVE_IMBLEARN = False


def _manual_smote(X, y, k=5, random_state=RANDOM_STATE):
    """Manual SMOTE (Chawla 2002): synthesize minority points by interpolating
    along edges to k nearest minority neighbors. Used only when imbalanced-learn
    is unavailable. Returns (X_res, y_res) as numpy arrays.
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return X, y
    majority = classes[counts.argmax()]
    minority = classes[counts.argmin()]
    n_maj = counts.max()
    n_min = counts.min()
    n_needed = n_maj - n_min
    if n_needed <= 0:
        return X, y
    X_min = X[y == minority]
    if len(X_min) <= 1:
        # Can't interpolate; duplicate
        idx = rng.integers(0, len(X_min), size=n_needed)
        X_new = X_min[idx]
    else:
        k_eff = min(k, len(X_min) - 1)
        nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(X_min)
        _, neigh = nn.kneighbors(X_min)
        # drop self (first column)
        neigh = neigh[:, 1:]
        # Pick random minority seed points then random neighbors
        seed_idx = rng.integers(0, len(X_min), size=n_needed)
        neighbor_choice = rng.integers(0, k_eff, size=n_needed)
        chosen_neighbors = neigh[seed_idx, neighbor_choice]
        gaps = rng.random(size=(n_needed, 1))
        X_new = X_min[seed_idx] + gaps * (X_min[chosen_neighbors] - X_min[seed_idx])
    y_new = np.full(n_needed, minority, dtype=y.dtype)
    X_res = np.vstack([X, X_new])
    y_res = np.concatenate([y, y_new])
    return X_res, y_res


def apply_smote(X_train, y_train, random_state=RANDOM_STATE):
    """Apply SMOTE oversampling, preferring imbalanced-learn when installed."""
    if _HAVE_IMBLEARN:
        # k must be < n_minority
        counts = np.bincount(y_train.astype(int))
        n_min = counts.min()
        k = max(1, min(5, n_min - 1))
        sm = _SMOTE(random_state=random_state, k_neighbors=k)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        return np.asarray(X_res), np.asarray(y_res)
    return _manual_smote(X_train.values if hasattr(X_train, "values") else X_train,
                         y_train.values if hasattr(y_train, "values") else y_train,
                         k=5, random_state=random_state)


def apply_random_under(X_train, y_train, random_state=RANDOM_STATE):
    """Random under-sampling: majority class trimmed to minority count."""
    rng = np.random.default_rng(random_state)
    y_arr = np.asarray(y_train)
    classes, counts = np.unique(y_arr, return_counts=True)
    if len(classes) < 2:
        return X_train, y_train
    majority = classes[counts.argmax()]
    minority = classes[counts.argmin()]
    n_min = counts.min()

    idx_maj = np.where(y_arr == majority)[0]
    idx_min = np.where(y_arr == minority)[0]
    idx_maj_keep = rng.choice(idx_maj, size=n_min, replace=False)
    keep = np.sort(np.concatenate([idx_maj_keep, idx_min]))
    if hasattr(X_train, "iloc"):
        X_res = X_train.iloc[keep].reset_index(drop=True)
    else:
        X_res = X_train[keep]
    if hasattr(y_train, "iloc"):
        y_res = y_train.iloc[keep].reset_index(drop=True)
    else:
        y_res = y_arr[keep]
    return X_res, y_res


def rebalanced_sample_weights(y_train, cohort_train):
    """Per-sample weight = 1 / (cohort_size * within-cohort class_prevalence).

    Each (cohort, class) cell contributes the same aggregate weight, which is
    the RebalancedCV idea (every cohort and every class is equally represented
    in the empirical training loss).
    """
    y_arr = np.asarray(y_train)
    c_arr = np.asarray(cohort_train)
    w = np.zeros(len(y_arr), dtype=float)
    for cohort in np.unique(c_arr):
        mask = c_arr == cohort
        n_c = mask.sum()
        for cls in np.unique(y_arr[mask]):
            cm = mask & (y_arr == cls)
            n_cc = cm.sum()
            if n_cc == 0 or n_c == 0:
                continue
            prevalence = n_cc / n_c
            w[cm] = 1.0 / (n_c * prevalence)
    # Normalize so mean weight is 1 (numerical stability for XGB)
    if w.sum() > 0:
        w = w * (len(w) / w.sum())
    return w


def make_rf():
    return RandomForestClassifier(
        n_estimators=500, max_features="sqrt", min_samples_leaf=5,
        n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced",
    )


def make_xgb(y_train):
    y_arr = np.asarray(y_train)
    n_neg = (y_arr == 0).sum()
    n_pos = (y_arr == 1).sum()
    spw = n_neg / max(n_pos, 1)
    return XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
        eval_metric="logloss", scale_pos_weight=spw, n_jobs=-1,
    )


def fit_and_score(model_name, X_tr, y_tr, X_te, y_te, sample_weight=None):
    if model_name == "rf":
        model = make_rf()
        if sample_weight is not None:
            model.fit(X_tr, y_tr, sample_weight=sample_weight)
        else:
            model.fit(X_tr, y_tr)
    elif model_name == "xgb":
        model = make_xgb(y_tr)
        if sample_weight is not None:
            model.fit(X_tr, y_tr, sample_weight=sample_weight)
        else:
            model.fit(X_tr, y_tr)
    else:
        raise ValueError(model_name)
    y_prob = model.predict_proba(X_te)[:, 1]
    return roc_auc_score(y_te, y_prob)


def get_country_aware_excluded(test_cohort, cohort_country):
    test_country = cohort_country.get(test_cohort)
    return {c for c, ct in cohort_country.items()
            if ct == test_country and c != test_cohort}


def run_task_strategy(task_name, model_name, strategy, label_map, X_all, md_all,
                      cohort_country):
    """Run LODO across adenoma cohorts for one (task, model, strategy)."""
    sub = md_all[md_all["study_condition"].isin(label_map.keys()) &
                 md_all["study_name"].isin(ADENOMA_COHORTS)].copy()
    sub["bin_label"] = sub["study_condition"].map(label_map).astype(int)

    rows = []
    for cohort in sorted(sub["study_name"].unique()):
        excluded = get_country_aware_excluded(cohort, cohort_country)

        test_mask = sub["study_name"] == cohort
        train_mask = (~test_mask) & (~sub["study_name"].isin(excluded))

        if sub.loc[test_mask, "bin_label"].nunique() < 2:
            print(f"    {cohort}: only one class in test, skipped")
            continue
        if sub.loc[train_mask, "bin_label"].nunique() < 2:
            print(f"    {cohort}: only one class in train, skipped")
            continue

        train_idx = sub[train_mask].index.tolist()
        test_idx = sub[test_mask].index.tolist()

        X_tr = X_all.loc[train_idx].reset_index(drop=True)
        X_te = X_all.loc[test_idx]
        y_tr = sub.loc[train_idx, "bin_label"].reset_index(drop=True)
        y_te = sub.loc[test_idx, "bin_label"]
        cohort_tr = sub.loc[train_idx, "study_name"].reset_index(drop=True)

        sample_weight = None
        n_tr_min_before = int(min(np.bincount(y_tr.values)))
        n_tr_maj_before = int(max(np.bincount(y_tr.values)))

        if strategy == "baseline":
            X_use, y_use = X_tr, y_tr
        elif strategy == "random_under":
            X_use, y_use = apply_random_under(X_tr, y_tr,
                                              random_state=RANDOM_STATE)
        elif strategy == "smote":
            X_use_arr, y_use_arr = apply_smote(X_tr, y_tr,
                                               random_state=RANDOM_STATE)
            X_use = pd.DataFrame(X_use_arr, columns=X_tr.columns)
            y_use = pd.Series(y_use_arr.astype(int))
        elif strategy == "rebalanced_weight":
            X_use, y_use = X_tr, y_tr
            sample_weight = rebalanced_sample_weights(y_tr, cohort_tr)
        else:
            raise ValueError(strategy)

        counts_after = np.bincount(np.asarray(y_use).astype(int))
        n_tr_min = int(counts_after.min())
        n_tr_maj = int(counts_after.max())

        auc = fit_and_score(model_name, X_use, y_use, X_te, y_te,
                            sample_weight=sample_weight)

        excl_str = f"  [excl: {sorted(excluded)}]" if excluded else ""
        print(f"    {cohort:18s} AUC={auc:.3f}  "
              f"n_tr={len(y_use)} (min={n_tr_min}, maj={n_tr_maj}; "
              f"raw min={n_tr_min_before}, maj={n_tr_maj_before})  "
              f"n_te={len(y_te)}{excl_str}")

        rows.append({
            "task": task_name,
            "strategy": strategy,
            "cohort": cohort,
            "auc": auc,
            "n_train": int(len(y_use)),
            "n_test": int(len(y_te)),
            "n_train_minority": n_tr_min,
            "n_train_majority": n_tr_maj,
        })

    return rows


def main():
    sp = pd.read_csv("data/processed/species_filtered.csv")
    md = pd.read_csv("data/processed/metadata_clean.csv")
    mg = md.merge(sp, on="sample_id", how="inner")
    feat_cols = [c for c in sp.columns if c != "sample_id"]

    X_all = mg[feat_cols].reset_index(drop=True)
    meta_cols = ["sample_id", "study_name", "study_condition", "country"]
    md_all = mg[meta_cols].reset_index(drop=True)

    cohort_country = (
        md_all.groupby("study_name")["country"]
        .agg(lambda x: x.mode().iloc[0])
        .to_dict()
    )

    tasks = [
        ("h_vs_a_rf",   {"control": 0, "adenoma": 1}, "rf"),
        ("h_vs_a_xgb",  {"control": 0, "adenoma": 1}, "xgb"),
        ("a_vs_crc_rf", {"adenoma": 0, "CRC": 1},     "rf"),
        ("a_vs_crc_xgb",{"adenoma": 0, "CRC": 1},     "xgb"),
    ]
    strategies = ["baseline", "random_under", "smote", "rebalanced_weight"]

    print(f"imbalanced-learn available: {_HAVE_IMBLEARN}")
    if not _HAVE_IMBLEARN:
        print("  -> using manual SMOTE (Chawla 2002 kNN interpolation)")

    print(f"\nAdenoma cohorts: {ADENOMA_COHORTS}")
    print(f"Cohort countries: "
          f"{ {c: cohort_country[c] for c in ADENOMA_COHORTS} }")

    all_rows = []
    for task_name, label_map, model_name in tasks:
        print(f"\n=== {task_name} ===")
        for strategy in strategies:
            print(f"  -- strategy: {strategy} --")
            rows = run_task_strategy(task_name, model_name, strategy,
                                     label_map, X_all, md_all, cohort_country)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    # Add mean_lodo_auc per (task, strategy)
    means = (df.groupby(["task", "strategy"])["auc"].mean()
               .rename("mean_lodo_auc").reset_index())
    df = df.merge(means, on=["task", "strategy"], how="left")

    os.makedirs("results", exist_ok=True)
    df.to_csv("results/adenoma_rebalanced_lodo.csv", index=False)
    print("\nSaved results/adenoma_rebalanced_lodo.csv "
          f"({len(df)} rows)")

    summary = (df.drop_duplicates(["task", "strategy"])
                 [["task", "strategy", "mean_lodo_auc"]]
                 .sort_values(["task", "strategy"]).reset_index(drop=True))
    summary.to_csv("results/adenoma_rebalanced_summary.csv", index=False)
    print(f"Saved results/adenoma_rebalanced_summary.csv ({len(summary)} rows)")

    # Headline table to stdout
    print("\n=== Headline: mean LODO AUC by (task, strategy) ===")
    pivot = summary.pivot(index="task", columns="strategy", values="mean_lodo_auc")
    pivot = pivot[["baseline", "random_under", "smote", "rebalanced_weight"]]
    print(pivot.round(3).to_string())

    print("\n=== Best strategy per task ===")
    for task in pivot.index:
        row = pivot.loc[task]
        best = row.idxmax()
        baseline = row["baseline"]
        delta = row[best] - baseline
        print(f"  {task:14s}  best={best:18s} "
              f"AUC={row[best]:.3f}  vs baseline={baseline:.3f}  "
              f"delta={delta:+.3f}")


if __name__ == "__main__":
    main()
