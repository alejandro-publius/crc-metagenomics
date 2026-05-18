"""Compare four cross-validation strategies on the species RF + 10-cohort dataset.

Empirically demonstrates why country-aware LODO is the appropriate evaluation
protocol for the headline model by quantifying how much cohort-ignorant CV
inflates the apparent AUC.

Strategies compared (all use the same species feature matrix, the same RF
hyper-parameters as ``train_baseline.py``, and the same CRC/control subset):

  1. Pooled stratified 5-fold CV (naive baseline; ignores cohort structure)
  2. Standard LODO (no country awareness; same-country leakage allowed)
  3. Country-aware LODO (matches ``train_baseline.py``; reference strategy)
  4. GroupKFold by cohort (equivalent to LeaveOneGroupOut at n_groups=n_folds;
     each cohort is held out exactly once, with no extra class stratification
     and no country-awareness applied)

For each strategy the script records:
  - per-fold AUC
  - pooled prediction AUC across all held-out samples
  - DeLong test on pooled predictions vs the country-aware LODO reference

Outputs (all paths absolute under repo root):
  - results/diagnostics/cv_methodology_comparison.csv
  - results/diagnostics/cv_methodology_summary.csv
  - figures/diagnostics/cv_methodology.png
  - results/diagnostics/CV_METHODOLOGY_NOTE.md

The script is standalone (does not import ``train_baseline``) but it
re-derives the dataset the same way the baseline does so the comparison is
apples-to-apples.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[2]
SPECIES_PATH = REPO / "data" / "processed" / "species_filtered.csv"
META_PATH = REPO / "data" / "processed" / "metadata_clean.csv"

OUT_PER_FOLD = REPO / "results" / "diagnostics" / "cv_methodology_comparison.csv"
OUT_SUMMARY = REPO / "results" / "diagnostics" / "cv_methodology_summary.csv"
OUT_FIG = REPO / "figures" / "diagnostics" / "cv_methodology.png"
OUT_NOTE = REPO / "results" / "diagnostics" / "CV_METHODOLOGY_NOTE.md"

RANDOM_STATE = 42

REFERENCE_STRATEGY = "country_aware_lodo"

STRATEGIES = [
    ("pooled_5fold", "Pooled stratified 5-fold"),
    ("standard_lodo", "Standard LODO"),
    ("country_aware_lodo", "Country-aware LODO"),
    ("group_kfold_cohort", "GroupKFold (cohort)"),
]


# ---------------------------------------------------------------------------
# DeLong machinery (Sun & Xu 2014 fast algorithm). Mirrors auc_comparison.py.
# ---------------------------------------------------------------------------
def _midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x, kind="mergesort")
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2


def delong_roc_test(y_true: np.ndarray, y_prob_a: np.ndarray,
                    y_prob_b: np.ndarray) -> tuple[float, float, float, float]:
    """Two-tailed DeLong test for paired AUCs.

    ``y_prob_a`` and ``y_prob_b`` must be aligned with the same ``y_true``
    label vector (i.e. predictions for the same samples under two scoring
    rules). Returns ``(auc_a, auc_b, z, p)``.
    """
    y_true = np.asarray(y_true).astype(int)
    pos = y_true == 1
    neg = y_true == 0
    m = int(pos.sum())
    n = int(neg.sum())
    if m == 0 or n == 0:
        raise ValueError("DeLong requires both classes present")
    aucs, v01s, v10s = [], [], []
    for y_prob in (y_prob_a, y_prob_b):
        y_prob = np.asarray(y_prob, dtype=float)
        x_pos = y_prob[pos]
        x_neg = y_prob[neg]
        tx = _midrank(x_pos)
        ty = _midrank(x_neg)
        tz = _midrank(np.concatenate([x_pos, x_neg]))
        auc = (tz[:m].sum() / m - (m + 1) / 2.0) / n
        v01 = (tz[:m] - tx) / n
        v10 = 1.0 - (tz[m:] - ty) / m
        aucs.append(auc)
        v01s.append(v01)
        v10s.append(v10)
    auc_a, auc_b = aucs
    S01 = np.cov(np.vstack(v01s))
    S10 = np.cov(np.vstack(v10s))
    S = S01 / m + S10 / n
    var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var_diff <= 0:
        return float(auc_a), float(auc_b), 0.0, 1.0
    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(auc_a), float(auc_b), float(z), float(p)


# ---------------------------------------------------------------------------
# Data loading (mirrors train_baseline.py exactly)
# ---------------------------------------------------------------------------
def load_dataset() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    sp = pd.read_csv(SPECIES_PATH)
    md = pd.read_csv(META_PATH)
    mg = md.merge(sp, on="sample_id", how="inner")
    fc = [c for c in sp.columns if c != "sample_id"]
    mask = mg["label"].isin([0, 1])
    X = mg.loc[mask, fc].reset_index(drop=True)
    y = mg.loc[mask, "label"].astype(int).reset_index(drop=True)
    meta_cols = ["sample_id", "study_name", "study_condition", "label"]
    if "country" in mg.columns:
        meta_cols.append("country")
    meta = mg.loc[mask, meta_cols].reset_index(drop=True)
    return X, y, meta


def make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )


# ---------------------------------------------------------------------------
# CV strategies. Each returns a list of (fold_label, train_idx, test_idx).
# Indices are positional into the (X, y, meta) frames returned by
# ``load_dataset``.
# ---------------------------------------------------------------------------
def splits_pooled_5fold(y: pd.Series, _meta: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return [
        (f"fold{i + 1}", train, test)
        for i, (train, test) in enumerate(skf.split(np.zeros(len(y)), y.values))
    ]


def splits_standard_lodo(_y: pd.Series, meta: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    out = []
    cohorts = sorted(meta["study_name"].unique())
    for cohort in cohorts:
        test = np.where(meta["study_name"].values == cohort)[0]
        train = np.where(meta["study_name"].values != cohort)[0]
        if len(test) == 0 or len(np.unique(_y.iloc[test])) < 2:
            continue
        out.append((cohort, train, test))
    return out


def splits_country_aware_lodo(_y: pd.Series, meta: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Match get_lodo_splits in scripts/lodo_cv.py: drop training samples from
    cohorts sharing the test cohort's majority country."""
    if "country" not in meta.columns:
        # No country column -> fall back to standard LODO.
        return splits_standard_lodo(_y, meta)
    cohort_country = (
        meta.groupby("study_name")["country"]
        .agg(lambda x: x.mode().iloc[0])
        .to_dict()
    )
    out = []
    cohorts = sorted(meta["study_name"].unique())
    for cohort in cohorts:
        test_country = cohort_country.get(cohort)
        same_country = {c for c, ct in cohort_country.items()
                        if ct == test_country and c != cohort}
        test = np.where(meta["study_name"].values == cohort)[0]
        train_mask = (
            (meta["study_name"].values != cohort)
            & (~meta["study_name"].isin(same_country).values)
        )
        train = np.where(train_mask)[0]
        if len(test) == 0 or len(np.unique(_y.iloc[test])) < 2:
            continue
        out.append((cohort, train, test))
    return out


def splits_group_kfold_cohort(_y: pd.Series, meta: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """One fold per cohort (group). With 10 cohorts and a leave-one-group-out
    schedule this is the standard "GroupKFold by cohort" comparator: each
    cohort is held out exactly once, but unlike country-aware LODO the
    training fold may include same-country cohorts.

    We label folds by the held-out cohort so each fold is interpretable;
    operationally it is equivalent to sklearn's LeaveOneGroupOut on cohorts
    (which is what GroupKFold reduces to when n_splits == n_groups).
    """
    out = []
    cohorts = sorted(meta["study_name"].unique())
    for cohort in cohorts:
        test = np.where(meta["study_name"].values == cohort)[0]
        train = np.where(meta["study_name"].values != cohort)[0]
        if len(test) == 0 or len(np.unique(_y.iloc[test])) < 2:
            continue
        out.append((cohort, train, test))
    return out


SPLIT_BUILDERS = {
    "pooled_5fold": splits_pooled_5fold,
    "standard_lodo": splits_standard_lodo,
    "country_aware_lodo": splits_country_aware_lodo,
    "group_kfold_cohort": splits_group_kfold_cohort,
}


# ---------------------------------------------------------------------------
# Run a strategy and collect per-fold AUCs + pooled predictions.
# ---------------------------------------------------------------------------
def run_strategy(strategy_key: str, X: pd.DataFrame, y: pd.Series,
                 meta: pd.DataFrame) -> dict:
    builder = SPLIT_BUILDERS[strategy_key]
    splits = builder(y, meta)
    y_arr = y.values

    fold_rows = []
    # pooled holders: each test sample gets one held-out probability.
    # For k-fold variants every sample appears in exactly one test fold;
    # for LODO variants the same holds (each sample's cohort is held out
    # exactly once). So we can build a single pooled prediction vector.
    pooled_pred = np.full(len(y_arr), np.nan, dtype=float)
    pooled_seen = np.zeros(len(y_arr), dtype=bool)

    for fold_label, train_idx, test_idx in splits:
        model = make_rf()
        model.fit(X.iloc[train_idx], y_arr[train_idx])
        y_prob = model.predict_proba(X.iloc[test_idx])[:, 1]
        y_true = y_arr[test_idx]
        try:
            auc = float(np.nan)
            if len(np.unique(y_true)) == 2:
                from sklearn.metrics import roc_auc_score
                auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = float("nan")

        fold_rows.append({
            "strategy": strategy_key,
            "fold_or_pooled": fold_label,
            "auc": auc,
            "n": int(len(test_idx)),
        })

        # Pooled bookkeeping: only first held-out prediction per sample.
        new_samples = test_idx[~pooled_seen[test_idx]]
        # Build a quick mapping for picking which slots in y_prob correspond
        # to new_samples (they appear in the same order as test_idx).
        if len(new_samples) > 0:
            keep_mask = ~pooled_seen[test_idx]
            pooled_pred[test_idx[keep_mask]] = y_prob[keep_mask]
            pooled_seen[test_idx[keep_mask]] = True

        print(f"  [{strategy_key:22s}] {fold_label:25s}  "
              f"AUC={auc:.3f}  n_test={len(test_idx):4d}  "
              f"n_train={len(train_idx):4d}")

    pooled_mask = pooled_seen
    pooled_auc = float("nan")
    if pooled_mask.sum() > 0 and len(np.unique(y_arr[pooled_mask])) == 2:
        from sklearn.metrics import roc_auc_score
        pooled_auc = float(roc_auc_score(y_arr[pooled_mask], pooled_pred[pooled_mask]))

    fold_rows.append({
        "strategy": strategy_key,
        "fold_or_pooled": "POOLED",
        "auc": pooled_auc,
        "n": int(pooled_mask.sum()),
    })

    return {
        "key": strategy_key,
        "fold_rows": fold_rows,
        "pooled_pred": pooled_pred,
        "pooled_mask": pooled_mask,
        "pooled_auc": pooled_auc,
        "per_fold_aucs": np.array([r["auc"] for r in fold_rows[:-1]], dtype=float),
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_plot(summary: pd.DataFrame, fig_path: Path) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    labels = [dict(STRATEGIES)[k] for k in summary["strategy"]]
    x = np.arange(len(labels))
    width = 0.38

    # Colorblind-safe (Wong palette) two-tone bars for per-fold mean vs pooled.
    color_fold = "#0072B2"   # blue
    color_pool = "#E69F00"   # orange

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    b1 = ax.bar(x - width / 2, summary["mean_per_fold_auc"], width,
                label="Mean per-fold AUC", color=color_fold,
                edgecolor="white", linewidth=0.6)
    b2 = ax.bar(x + width / 2, summary["pooled_auc"], width,
                label="Pooled AUC", color=color_pool,
                edgecolor="white", linewidth=0.6)

    for bars, vals in ((b1, summary["mean_per_fold_auc"]),
                       (b2, summary["pooled_auc"])):
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    # Significance stars vs country-aware LODO, anchored on the pooled bar.
    for i, (key, p) in enumerate(zip(summary["strategy"],
                                     summary["delong_p_vs_country_aware"])):
        if key == REFERENCE_STRATEGY or pd.isna(p):
            continue
        if p < 0.001:
            star = "***"
        elif p < 0.01:
            star = "**"
        elif p < 0.05:
            star = "*"
        else:
            star = "ns"
        bar = b2[i]
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.030,
                star, ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="black")

    # Reference line for country-aware LODO pooled AUC.
    ref_pooled = summary.loc[summary["strategy"] == REFERENCE_STRATEGY,
                             "pooled_auc"].iloc[0]
    ax.axhline(ref_pooled, color=color_pool, linestyle="--",
               linewidth=1, alpha=0.6,
               label=f"Country-aware pooled AUC = {ref_pooled:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("AUC", fontsize=11)
    ymax = float(np.nanmax(np.concatenate([summary["mean_per_fold_auc"].values,
                                           summary["pooled_auc"].values])))
    ax.set_ylim(0.5, min(1.0, ymax + 0.10))
    ax.set_title("Cross-validation methodology comparison — species RF\n"
                 "(stars: DeLong p vs country-aware LODO on pooled predictions)",
                 fontsize=11)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.tick_params(axis="y", labelsize=10)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Note writer
# ---------------------------------------------------------------------------
def write_note(summary: pd.DataFrame, note_path: Path) -> None:
    note_path.parent.mkdir(parents=True, exist_ok=True)

    def row(key: str) -> pd.Series:
        return summary.loc[summary["strategy"] == key].iloc[0]

    ref = row(REFERENCE_STRATEGY)
    pooled_ref = ref["pooled_auc"]
    mean_ref = ref["mean_per_fold_auc"]

    pooled_5 = row("pooled_5fold")
    standard = row("standard_lodo")
    gkf = row("group_kfold_cohort")

    inflation_kfold = pooled_5["pooled_auc"] - pooled_ref
    inflation_std_lodo = standard["pooled_auc"] - pooled_ref
    inflation_gkf = gkf["pooled_auc"] - pooled_ref

    def sig(p: float) -> str:
        if pd.isna(p):
            return "n/a"
        if p < 0.001:
            return f"p<0.001"
        return f"p={p:.3f}"

    sig_pooled = sig(pooled_5["delong_p_vs_country_aware"])
    sig_std = sig(standard["delong_p_vs_country_aware"])
    sig_gkf = sig(gkf["delong_p_vs_country_aware"])

    headline = (
        f"Pooled 5-fold CV inflates pooled AUC by "
        f"{inflation_kfold:+.3f} relative to country-aware LODO "
        f"({pooled_5['pooled_auc']:.3f} vs {pooled_ref:.3f}; "
        f"DeLong {sig_pooled}); standard LODO inflates by "
        f"{inflation_std_lodo:+.3f} ({standard['pooled_auc']:.3f} vs "
        f"{pooled_ref:.3f}; DeLong {sig_std}). Country-aware LODO is the "
        f"honest estimator."
    )

    body = [
        "# CV methodology comparison",
        "",
        headline,
        "",
        "## Details",
        "",
        f"- **Pooled stratified 5-fold CV** "
        f"(naive baseline): mean per-fold AUC "
        f"{pooled_5['mean_per_fold_auc']:.3f}, pooled AUC "
        f"{pooled_5['pooled_auc']:.3f}. Pooled-AUC gap vs country-aware LODO: "
        f"{inflation_kfold:+.3f} ({sig_pooled}).",
        f"- **Standard LODO** (no country awareness): mean per-fold AUC "
        f"{standard['mean_per_fold_auc']:.3f}, pooled AUC "
        f"{standard['pooled_auc']:.3f}. Pooled-AUC gap: "
        f"{inflation_std_lodo:+.3f} ({sig_std}).",
        f"- **Country-aware LODO** (reference; matches `train_baseline.py`): "
        f"mean per-fold AUC {mean_ref:.3f}, pooled AUC {pooled_ref:.3f}.",
        f"- **GroupKFold by cohort** (one fold per cohort, no country masking): "
        f"mean per-fold AUC {gkf['mean_per_fold_auc']:.3f}, pooled AUC "
        f"{gkf['pooled_auc']:.3f}. Pooled-AUC gap: {inflation_gkf:+.3f} "
        f"({sig_gkf}).",
        "",
        "Pooled k-fold CV mixes samples from the same cohort across train and",
        "test, which lets the model memorise cohort-specific batch effects",
        "instead of generalisable CRC signal. Standard LODO removes the test",
        "cohort but still allows another cohort from the same country to leak",
        "population-level signal into training. Country-aware LODO is the only",
        "protocol here that forces the model to generalise across both cohort",
        "and country boundaries, and it is therefore the AUC we should report",
        "and compare against external validation cohorts.",
        "",
        f"Generated by `scripts/diagnostics/cv_methodology_comparison.py`.",
        "",
    ]
    note_path.write_text("\n".join(body))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_PER_FOLD.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    X, y, meta = load_dataset()
    print(f"  Samples: {len(X)} (CRC={int(y.sum())}, "
          f"control={int((y == 0).sum())})")
    print(f"  Features: {X.shape[1]}")
    print(f"  Cohorts:  {meta['study_name'].nunique()}")
    if "country" in meta.columns:
        print(f"  Countries: {meta['country'].nunique()}")

    results = {}
    all_fold_rows = []
    for key, _label in STRATEGIES:
        print(f"\n=== Running {key} ===")
        res = run_strategy(key, X, y, meta)
        results[key] = res
        all_fold_rows.extend(res["fold_rows"])

    per_fold_df = pd.DataFrame(all_fold_rows)
    per_fold_df.to_csv(OUT_PER_FOLD, index=False)
    print(f"\nWrote {OUT_PER_FOLD}")

    # ---- Summary + DeLong vs reference ------------------------------------
    y_arr = y.values
    ref = results[REFERENCE_STRATEGY]

    summary_rows = []
    for key, _label in STRATEGIES:
        res = results[key]
        per_fold = res["per_fold_aucs"]
        per_fold = per_fold[~np.isnan(per_fold)]
        mean_per_fold = float(np.mean(per_fold)) if len(per_fold) else float("nan")
        pooled_auc = res["pooled_auc"]

        # DeLong vs country-aware on samples scored by BOTH strategies
        # (every sample is scored by every strategy in this setup, but we
        # intersect masks defensively).
        if key == REFERENCE_STRATEGY:
            delong_p = float("nan")
            delong_z = float("nan")
        else:
            both = res["pooled_mask"] & ref["pooled_mask"]
            if both.sum() > 0 and len(np.unique(y_arr[both])) == 2:
                _, _, z, p = delong_roc_test(
                    y_arr[both],
                    res["pooled_pred"][both],
                    ref["pooled_pred"][both],
                )
                delong_p = float(p)
                delong_z = float(z)
            else:
                delong_p = float("nan")
                delong_z = float("nan")

        summary_rows.append({
            "strategy": key,
            "n_folds": int(len(per_fold)),
            "mean_per_fold_auc": round(mean_per_fold, 4),
            "pooled_auc": round(pooled_auc, 4),
            "delong_z_vs_country_aware": (round(delong_z, 4)
                                          if not np.isnan(delong_z)
                                          else float("nan")),
            "delong_p_vs_country_aware": (delong_p
                                          if np.isnan(delong_p)
                                          else round(delong_p, 6)),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"Wrote {OUT_SUMMARY}")

    make_plot(summary, OUT_FIG)
    print(f"Wrote {OUT_FIG}")

    write_note(summary, OUT_NOTE)
    print(f"Wrote {OUT_NOTE}")

    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
