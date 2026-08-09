"""Create the main intervention-readiness attrition figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "intervention_readiness"
OUT = ROOT / "manuscript" / "intervention_readiness" / "figures"


def main() -> None:
    funnel = pd.read_csv(RESULTS / "candidate_attrition_funnel.csv")
    taxon = pd.read_csv(RESULTS / "candidate_taxon_summary.csv").sort_values(
        "dominant_taxon_fraction", ascending=True
    )
    parent = pd.read_csv(RESULTS / "parent_adjustment_summary.csv")
    parent = parent[parent["parent_adjustment_gate"].eq("pass")]
    combined = taxon.merge(parent, on="gene_id", validate="one_to_one")

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    figure = plt.figure(figsize=(11.5, 7.2), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=[0.82, 1.2])
    funnel_ax = figure.add_subplot(grid[0, :])
    source_ax = figure.add_subplot(grid[1, 0])
    delta_ax = figure.add_subplot(grid[1, 1])

    funnel_ax.set_xlim(0, 1)
    funnel_ax.set_ylim(0, 1)
    funnel_ax.axis("off")
    x_positions = np.linspace(0.12, 0.88, len(funnel))
    widths = [0.20, 0.18, 0.18, 0.18]
    colors = ["#264653", "#2A9D8F", "#E9C46A", "#D1495B"]
    for index, row in funnel.iterrows():
        x = x_positions[index]
        funnel_ax.add_patch(
            plt.Rectangle(
                (x - widths[index] / 2, 0.35),
                widths[index],
                0.38,
                facecolor=colors[index],
                edgecolor="none",
                alpha=0.96,
            )
        )
        funnel_ax.text(
            x,
            0.57,
            f"{int(row.n_candidates):,}",
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
            color="white",
        )
        funnel_ax.text(
            x,
            0.28,
            str(row.label).replace(" ", "\n", 1),
            ha="center",
            va="top",
            fontsize=9,
        )
        if index < len(funnel) - 1:
            funnel_ax.annotate(
                "",
                xy=(x_positions[index + 1] - widths[index + 1] / 2 - 0.01, 0.54),
                xytext=(x + widths[index] / 2 + 0.01, 0.54),
                arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.6},
            )
    funnel_ax.set_title(
        "A  Candidate attrition through prespecified intervention-readiness gates",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )

    labels = [value.replace("UniRef90_", "") for value in combined["gene_id"]]
    y = np.arange(len(combined))
    source_ax.barh(
        y - 0.16,
        combined["dominant_taxon_fraction"],
        height=0.3,
        label="largest observed carrier",
        color="#2A9D8F",
    )
    source_ax.barh(
        y + 0.16,
        combined["parent_taxon_fraction"],
        height=0.3,
        label="archived parent-species proxy",
        color="#F4A261",
    )
    source_ax.axvline(0.80, color="#D1495B", linestyle="--", linewidth=1.5)
    source_ax.text(
        0.80,
        0.96,
        "  frozen 80% gate",
        color="#A62A3A",
        va="top",
        transform=source_ax.get_xaxis_transform(),
    )
    source_ax.set_yticks(y, labels)
    source_ax.set_xlim(0, 1)
    source_ax.set_xlabel("Fraction of taxon-stratified abundance")
    source_ax.set_title("B  No candidate has a dominant carrier", loc="left", fontweight="bold")
    source_ax.legend(frameon=False, loc="lower right", fontsize=8)

    ordered = combined.sort_values("median_delta_auc", ascending=True)
    y2 = np.arange(len(ordered))
    delta_ax.barh(y2, ordered["median_delta_auc"], color="#457B9D")
    delta_ax.axvline(0.02, color="#D1495B", linestyle="--", linewidth=1.5)
    delta_ax.set_yticks(
        y2, [value.replace("UniRef90_", "") for value in ordered["gene_id"]]
    )
    delta_ax.set_xlabel("Median held-out AUC gain beyond parent species")
    delta_ax.set_title(
        "C  Four signals pass species adjustment",
        loc="left",
        fontweight="bold",
    )
    delta_ax.text(
        0.02,
        0.96,
        "  frozen 0.02 gate",
        color="#A62A3A",
        va="top",
        transform=delta_ax.get_xaxis_transform(),
    )

    figure.suptitle(
        "Cross-population CRC gene-family signals fail taxonomic address resolution",
        fontsize=14,
        fontweight="bold",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT / "Figure1_target_attrition.png", dpi=300, bbox_inches="tight")
    figure.savefig(OUT / "Figure1_target_attrition.pdf", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
