#!/usr/bin/env python3
"""Plot held-out portability across molecular representations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript" / "generalization_risk" / "figures"


def auc_by_cohort(path: str) -> pd.DataFrame:
    frame = pd.read_csv(ROOT / path)
    return (
        frame.groupby("cohort")
        .apply(lambda group: roc_auc_score(group.y_true, group.y_prob), include_groups=False)
        .rename("auc")
        .reset_index()
    )


def internal_rows() -> pd.DataFrame:
    tables = []
    sources = [
        ("Species abundance", auc_by_cohort("results/preds_species_rf.csv")),
        ("Community pathways", pd.read_csv(ROOT / "results/bio_pathway_results.csv").rename(columns={"bio_pw_auc": "auc"})[["cohort", "auc"]]),
        ("Species-resolved pathways\n(source-only correction)", pd.read_csv(ROOT / "results/species_aware_correction/lodo_results.csv").query("model == 'stratified_source_only'")[["cohort", "auc"]]),
        ("Gene families", pd.read_csv(ROOT / "results/gene_family_lodo_results.csv")[["cohort", "auc"]]),
        ("Frozen mechanism genes", pd.read_csv(ROOT / "results/mechanism_panel/lodo_results.csv").query("model == 'mechanism_only'")[["cohort", "auc"]]),
    ]
    for representation, table in sources:
        table = table.copy()
        table["representation"] = representation
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = internal_rows()
    external = pd.read_csv(ROOT / "results/external_cohort/uncertainty_metrics.csv").query(
        "scope == 'overall'"
    ).iloc[0]
    order = [
        "Frozen mechanism genes",
        "Gene families",
        "Species-resolved pathways\n(source-only correction)",
        "Community pathways",
        "Species abundance",
    ]
    rng = np.random.default_rng(20260808)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    colors = ["#9CA3AF", "#7C3AED", "#D97706", "#0F766E", "#2563EB"]
    summary_rows = []
    for y, (label, color) in enumerate(zip(order, colors)):
        values = data.loc[data.representation == label, "auc"].to_numpy()
        jitter = rng.uniform(-0.11, 0.11, len(values))
        ax.scatter(values, y + jitter, s=28, color=color, alpha=0.58, edgecolor="none")
        mean = values.mean()
        ax.plot([values.min(), values.max()], [y, y], color=color, lw=1.5, alpha=0.8)
        ax.scatter([mean], [y], s=82, color=color, edgecolor="white", linewidth=1.1, zorder=3)
        summary_rows.append({
            "representation": label.replace("\n", " "), "setting": "internal LODO",
            "mean_auc": mean, "min_auc": values.min(), "max_auc": values.max(),
            "n_cohorts": len(values),
        })

    external_y = len(order)
    ax.errorbar(
        external.auc, external_y,
        xerr=[[external.auc - external.auc_ci_low], [external.auc_ci_high - external.auc]],
        fmt="*", markersize=14, color="#DC2626", ecolor="#DC2626", capsize=4,
        label="Untouched external cohort (95% bootstrap CI)",
    )
    summary_rows.append({
        "representation": "Species abundance", "setting": "external PRJNA763023",
        "mean_auc": external.auc, "min_auc": external.auc_ci_low,
        "max_auc": external.auc_ci_high, "n_cohorts": 1,
    })

    ax.axvline(0.5, color="#6B7280", linestyle="--", lw=1)
    ax.set_yticks(range(len(order) + 1), order + ["Species abundance\n(external, n=200)"])
    ax.set_xlim(0.48, 0.98)
    ax.set_xlabel("Area under the ROC curve")
    ax.set_title("Added biological detail does not guarantee cross-study portability", loc="left", weight="bold")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    fig.text(
        0.01, 0.01,
        "Circles show ten held-out development cohorts; large circles are means and lines span cohort ranges.",
        fontsize=8.3, color="#4B5563",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT / "PortabilityLandscape.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "PortabilityLandscape.pdf", bbox_inches="tight")
    pd.DataFrame(summary_rows).to_csv(
        ROOT / "results" / "portability_summary.csv", index=False
    )


if __name__ == "__main__":
    main()
