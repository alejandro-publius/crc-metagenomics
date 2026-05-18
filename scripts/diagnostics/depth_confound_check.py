"""Cohort-sequencing-depth confounding check for species SHAP rankings.

Reviewers commonly worry that the SHAP-based feature importance for a
metagenomic classifier could be driven by the per-cohort sequencing depth
(e.g., the F. nucleatum signal might just track cohorts with deeper
sequencing where the species is more detectable). This script tests that
directly:

  1. For each cohort, compute median read depth from metadata.
  2. For each cohort, refit the species RF on the other 9 cohorts and
     compute per-sample TreeSHAP on the held-out cohort. Aggregate to a
     per-cohort mean |SHAP| per species and a per-cohort rank.
  3. For each of the top 20 species by global SHAP (from
     results/shap_crc_features.csv), compute Spearman rho between the
     per-cohort SHAP rank and per-cohort median depth.

Outputs:
  - results/diagnostics/depth_confound_shap.csv
        species, rho_depth_vs_rank, p_value
  - figures/diagnostics/depth_vs_fnucleatum_shap.png
        per-cohort F. nucleatum mean |SHAP| vs cohort median depth.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier

REPO = Path(__file__).resolve().parents[2]
SPECIES_CSV = REPO / "data" / "processed" / "species_filtered.csv"
META_CSV = REPO / "data" / "processed" / "metadata_clean.csv"
SHAP_GLOBAL_CSV = REPO / "results" / "shap_crc_features.csv"

OUT_CSV = REPO / "results" / "diagnostics" / "depth_confound_shap.csv"
OUT_FIG = REPO / "figures" / "diagnostics" / "depth_vs_fnucleatum_shap.png"

RANDOM_STATE = 42
TOP_N = 20
FNUC_TOKEN = "Fusobacterium_nucleatum"


def short_name(feature: str) -> str:
    if "|" not in feature:
        return feature
    leaf = feature.split("|")[-1]
    if leaf.startswith("s__"):
        leaf = leaf[3:]
    return leaf


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    sp = pd.read_csv(SPECIES_CSV)
    md = pd.read_csv(META_CSV)
    mg = md.merge(sp, on="sample_id", how="inner")

    fc = [c for c in sp.columns if c != "sample_id"]
    mask = mg["study_condition"].isin(["CRC", "control"])
    work = mg.loc[mask].reset_index(drop=True)
    X = work[fc].copy()
    y = (work["study_condition"] == "CRC").astype(int)
    cohorts = work["study_name"].values
    depth = work["number_reads"].astype(float).values

    print(f"Samples: {len(work)} (CRC={int(y.sum())}, ctrl={int((1-y).sum())})")
    cohort_list = sorted(np.unique(cohorts).tolist())
    print(f"Cohorts: {len(cohort_list)}")

    median_depth = {
        c: float(np.median(depth[cohorts == c])) for c in cohort_list
    }

    shap_global = pd.read_csv(SHAP_GLOBAL_CSV).sort_values(
        "mean_abs_shap", ascending=False
    )
    top_features = shap_global["feature"].head(TOP_N).tolist()

    # Per-cohort, per-species mean |SHAP|. Build a (cohort x feature) matrix.
    per_cohort_shap = pd.DataFrame(index=cohort_list, columns=fc, dtype=float)
    fnuc_feature = None
    for c in fc:
        if FNUC_TOKEN in c:
            fnuc_feature = c
            break

    for test_cohort in cohort_list:
        tr = cohorts != test_cohort
        te = cohorts == test_cohort
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        print(f"  fold: hold out {test_cohort:25s} "
              f"(n_train={int(tr.sum())}, n_test={int(te.sum())})")
        rf = RandomForestClassifier(
            n_estimators=500, max_features="sqrt",
            min_samples_leaf=5, n_jobs=-1,
            random_state=RANDOM_STATE, class_weight="balanced",
        )
        rf.fit(X.loc[tr], y.loc[tr])
        sv = shap.TreeExplainer(rf).shap_values(X.loc[te])
        if isinstance(sv, np.ndarray) and sv.ndim == 3:
            sv = sv[:, :, 1]
        elif isinstance(sv, list):
            sv = np.array(sv[1])
        per_cohort_shap.loc[test_cohort, :] = np.abs(sv).mean(axis=0)

    per_cohort_shap = per_cohort_shap.dropna(how="all")

    # Per-cohort rank (higher mean|SHAP| -> rank 1)
    per_cohort_rank = per_cohort_shap.rank(axis=1, ascending=False, method="min")

    depth_vec = np.array([median_depth[c] for c in per_cohort_shap.index])

    rows = []
    for feat in top_features:
        if feat not in per_cohort_rank.columns:
            continue
        ranks = per_cohort_rank[feat].astype(float).values
        rho, p = spearmanr(depth_vec, ranks)
        rows.append({
            "species": short_name(feat),
            "rho_depth_vs_rank": round(float(rho), 4),
            "p_value": round(float(p), 4),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")

    fnuc_row = out[out["species"] == FNUC_TOKEN]
    if len(fnuc_row):
        print(f"F. nucleatum: rho(depth vs rank) = {fnuc_row['rho_depth_vs_rank'].iloc[0]:+.3f}, "
              f"p = {fnuc_row['p_value'].iloc[0]:.3f}")

    # F. nucleatum scatter: per-cohort mean |SHAP| vs median depth (in millions)
    if fnuc_feature is None or fnuc_feature not in per_cohort_shap.columns:
        print("F. nucleatum feature not found; skipping figure.")
        return

    fn_shap = per_cohort_shap[fnuc_feature].astype(float).values
    depth_m = depth_vec / 1e6

    # Report the same statistic that goes into the CSV: per-cohort SHAP
    # RANK vs per-cohort median depth. The scatter still plots raw
    # mean|SHAP| so readers can see the per-cohort spread.
    fn_ranks = per_cohort_rank[fnuc_feature].astype(float).values
    rho_val, p_val = spearmanr(depth_vec, fn_ranks)

    plt.style.use("seaborn-v0_8-whitegrid")
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    cohorts_idx = list(per_cohort_shap.index)
    for i, c in enumerate(cohorts_idx):
        ax.scatter(
            depth_m[i], fn_shap[i],
            s=140, color=cmap(i % 10), edgecolor="black", linewidth=0.8,
            label=c, zorder=3,
        )

    ax.set_xlabel("Cohort median sequencing depth (millions of reads)", fontsize=11)
    ax.set_ylabel("F. nucleatum per-cohort mean |SHAP|", fontsize=11)
    ax.set_title(
        f"F. nucleatum SHAP vs cohort sequencing depth\n"
        f"Spearman rho = {rho_val:+.3f}, p = {p_val:.3f}",
        fontsize=12,
    )
    ax.tick_params(labelsize=10)
    ax.legend(loc="best", fontsize=8, frameon=True, ncol=2)
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
