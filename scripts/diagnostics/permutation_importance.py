"""Permutation importance for the species RF model (CRC vs control).

This script addresses the well-known critique that TreeSHAP can be biased
toward features with higher cardinality / more split opportunities.
Permutation importance is model-agnostic and is computed on the same fitted
estimator, so it offers an independent cross-check of the TreeSHAP species
ranking that drives the biological interpretation in the manuscript.

Outputs:
  - results/diagnostics/permutation_importance_species_rf.csv
        feature, perm_imp_mean, perm_imp_std, perm_imp_rank
  - results/diagnostics/permutation_vs_shap_correlation.csv
        feature, perm_rank, shap_rank, both_top20
  - figures/diagnostics/permutation_vs_shap.png
        scatter of SHAP rank vs permutation rank for the top 50 species,
        with the four oral pathobionts highlighted and labeled.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

REPO = Path(__file__).resolve().parents[2]
SPECIES_CSV = REPO / "data" / "processed" / "species_filtered.csv"
META_CSV = REPO / "data" / "processed" / "metadata_clean.csv"

OUT_PERM_CSV = REPO / "results" / "diagnostics" / "permutation_importance_species_rf.csv"
OUT_CMP_CSV = REPO / "results" / "diagnostics" / "permutation_vs_shap_correlation.csv"
OUT_FIG = REPO / "figures" / "diagnostics" / "permutation_vs_shap.png"

RANDOM_STATE = 42
N_REPEATS = 30

ORAL_PATHOBIONTS = {
    "Fusobacterium_nucleatum": "F. nucleatum",
    "Peptostreptococcus_stomatis": "P. stomatis",
    "Parvimonas_micra": "P. micra",
    "Gemella_morbillorum": "G. morbillorum",
}


def short_name(feature: str) -> str:
    """Last clade in a MetaPhlAn lineage string, with s__ stripped."""
    if "|" not in feature:
        return feature
    leaf = feature.split("|")[-1]
    if leaf.startswith("s__"):
        leaf = leaf[3:]
    return leaf


def main() -> None:
    OUT_PERM_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    sp = pd.read_csv(SPECIES_CSV)
    md = pd.read_csv(META_CSV)
    mg = md.merge(sp, on="sample_id", how="inner")
    fc = [c for c in sp.columns if c != "sample_id"]
    mask = mg["study_condition"].isin(["CRC", "control"])
    X = mg.loc[mask, fc].reset_index(drop=True)
    y = (mg.loc[mask, "study_condition"] == "CRC").astype(int).reset_index(drop=True)
    print(f"Fitting species RF on {len(X)} samples / {X.shape[1]} features")

    rf = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    rf.fit(X, y)

    print(f"Permutation importance (n_repeats={N_REPEATS}, scoring=roc_auc)...")
    pi = permutation_importance(
        rf, X, y,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=-1,
    )

    perm_df = pd.DataFrame({
        "feature": X.columns.tolist(),
        "perm_imp_mean": pi.importances_mean,
        "perm_imp_std": pi.importances_std,
    })
    perm_df["perm_imp_rank"] = (
        perm_df["perm_imp_mean"].rank(ascending=False, method="min").astype(int)
    )
    perm_df = perm_df.sort_values("perm_imp_rank").reset_index(drop=True)
    perm_df_out = perm_df.copy()
    perm_df_out["perm_imp_mean"] = perm_df_out["perm_imp_mean"].round(4)
    perm_df_out["perm_imp_std"] = perm_df_out["perm_imp_std"].round(4)
    perm_df_out.to_csv(OUT_PERM_CSV, index=False)
    print(f"Wrote {OUT_PERM_CSV}")

    print("Computing TreeSHAP on the same fit...")
    sv = shap.TreeExplainer(rf).shap_values(X)
    if isinstance(sv, np.ndarray) and sv.ndim == 3:
        sv = sv[:, :, 1]
    elif isinstance(sv, list):
        sv = np.array(sv[1])
    shap_mean = np.abs(sv).mean(axis=0).flatten()
    shap_df = pd.DataFrame({"feature": X.columns.tolist(), "mean_abs_shap": shap_mean})
    shap_df["shap_rank"] = (
        shap_df["mean_abs_shap"].rank(ascending=False, method="min").astype(int)
    )

    cmp = perm_df[["feature", "perm_imp_rank"]].merge(
        shap_df[["feature", "shap_rank"]], on="feature"
    )
    cmp = cmp.rename(columns={"perm_imp_rank": "perm_rank"})
    cmp["both_top20"] = (cmp["perm_rank"] <= 20) & (cmp["shap_rank"] <= 20)
    cmp = cmp.sort_values("shap_rank").reset_index(drop=True)
    cmp.to_csv(OUT_CMP_CSV, index=False)
    print(f"Wrote {OUT_CMP_CSV}")

    n_overlap_top20 = int(cmp["both_top20"].sum())
    print(f"Top-20 overlap between SHAP and permutation: {n_overlap_top20}/20")

    plot_df = cmp[(cmp["shap_rank"] <= 50) | (cmp["perm_rank"] <= 50)].copy()
    plot_df["short"] = plot_df["feature"].apply(short_name)

    plt.style.use("seaborn-v0_8-whitegrid")
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7.5, 7.0))

    is_oral = plot_df["short"].isin(ORAL_PATHOBIONTS.keys())
    ax.scatter(
        plot_df.loc[~is_oral, "shap_rank"],
        plot_df.loc[~is_oral, "perm_rank"],
        s=45,
        color=cmap(0),
        alpha=0.7,
        edgecolor="white",
        linewidth=0.6,
        label="Other species (top 50)",
    )
    ax.scatter(
        plot_df.loc[is_oral, "shap_rank"],
        plot_df.loc[is_oral, "perm_rank"],
        s=130,
        color=cmap(3),
        edgecolor="black",
        linewidth=1.0,
        label="Oral pathobionts",
        zorder=5,
    )

    # Manual offsets per pathobiont so labels do not collide in the
    # top-right corner of the scatter (the axes are inverted, so positive
    # x-offset moves a label toward higher ranks).
    label_offsets = {
        "Gemella_morbillorum": (12, 14),
        "Fusobacterium_nucleatum": (14, -10),
        "Peptostreptococcus_stomatis": (14, 10),
        "Parvimonas_micra": (12, -14),
    }
    for _, r in plot_df.loc[is_oral].iterrows():
        label = ORAL_PATHOBIONTS[r["short"]]
        dx, dy = label_offsets.get(r["short"], (8, 6))
        ax.annotate(
            label,
            (r["shap_rank"], r["perm_rank"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            color="black",
            arrowprops=dict(arrowstyle="-", color="black", lw=0.6),
        )

    lo, hi = 0.5, max(plot_df[["shap_rank", "perm_rank"]].max()) + 2
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="grey", linewidth=1,
            label="Equal rank")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel("TreeSHAP rank (1 = highest mean |SHAP|)", fontsize=11)
    ax.set_ylabel("Permutation-importance rank (1 = largest AUC drop)", fontsize=11)
    ax.set_title(
        f"SHAP vs permutation importance — species RF, CRC vs control\n"
        f"Top-20 overlap: {n_overlap_top20}/20",
        fontsize=12,
    )
    ax.tick_params(labelsize=10)
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
