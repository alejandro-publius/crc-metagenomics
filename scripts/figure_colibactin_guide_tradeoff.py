"""Plot the opposing conservation and specificity rankings of two guides."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "intervention_readiness"
OUT = ROOT / "manuscript" / "intervention_readiness" / "figures"


def main() -> None:
    conservation = pd.read_csv(
        RESULTS / "colibactin_human_isolate_conservation_summary.csv"
    ).sort_values("benchmark_role")
    specificity = pd.read_csv(
        RESULTS / "colibactin_specificity_pilot_summary.csv"
    ).sort_values("benchmark_role")
    guides = conservation["guide_id"].tolist()
    if guides != specificity["guide_id"].tolist():
        raise ValueError("conservation and specificity guide ordering differs")

    labels = ["Primary clbB", "Secondary clbC"]
    colors = ["#2A9D8F", "#D1495B"]
    y = np.arange(len(guides))
    height = 0.32

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    figure, (conservation_ax, specificity_ax) = plt.subplots(
        1, 2, figsize=(10.5, 4.6), constrained_layout=True
    )

    coverage = conservation["coverage_fraction"].to_numpy() * 100
    unique = conservation["unique_site_fraction"].to_numpy() * 100
    conservation_ax.barh(
        y - height / 2,
        coverage,
        height=height,
        color="#457B9D",
        label="exact-site coverage",
    )
    conservation_ax.barh(
        y + height / 2,
        unique,
        height=height,
        color="#A8DADC",
        label="unique-site coverage",
    )
    conservation_ax.axvline(90, color="#555555", linestyle="--", linewidth=1.2)
    conservation_ax.set_xlim(89, 101.5)
    conservation_ax.set_yticks(y, labels)
    conservation_ax.invert_yaxis()
    conservation_ax.set_xlabel("Human-isolate genomes covered (%)")
    conservation_ax.set_title(
        "A  Both guides pass conservation",
        loc="left",
        fontweight="bold",
    )
    conservation_ax.legend(frameon=False, loc="center right", fontsize=8)
    for index, (coverage_value, unique_value) in enumerate(zip(coverage, unique)):
        conservation_ax.text(
            coverage_value + 0.15,
            index - height / 2,
            f"{coverage_value:.1f}%",
            va="center",
            fontsize=8,
        )
        conservation_ax.text(
            unique_value + 0.15,
            index + height / 2,
            f"{unique_value:.1f}%",
            va="center",
            fontsize=8,
        )

    flagged = specificity["n_flagged_sites"].to_numpy()
    specificity_ax.barh(y, flagged, color=colors, height=0.5)
    specificity_ax.set_yticks(y, labels)
    specificity_ax.invert_yaxis()
    specificity_ax.set_xlim(0, max(flagged) + 1.8)
    specificity_ax.set_xlabel("Flagged sites in 11 protected references")
    specificity_ax.set_title(
        "B  Specificity favors the primary guide",
        loc="left",
        fontweight="bold",
    )
    for index, value in enumerate(flagged):
        status = "pilot pass" if value == 0 else "not passed"
        specificity_ax.text(
            value + 0.12,
            index,
            f"{int(value)}  ({status})",
            va="center",
            fontweight="bold",
            color=colors[index],
        )

    figure.suptitle(
        "Published colibactin guides show a conservation–specificity tradeoff",
        fontsize=13,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.01,
        "Sequence flags are screening motifs, not evidence of exposure or editing in human cells.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUT / "Figure2_colibactin_guide_tradeoff.png",
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(OUT / "Figure2_colibactin_guide_tradeoff.pdf", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
