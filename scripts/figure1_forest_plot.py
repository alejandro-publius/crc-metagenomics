"""Generate Figure 1 (forest plot) for the manuscript.

Per-cohort and pooled LODO AUCs with 95% bootstrap confidence interval
whiskers, for species RF, joint RF, and joint XGBoost. Reads
results/bootstrap_ci.csv and writes manuscript/figures/Figure1_Forest_Plot
(.png and .pdf).

Usage:
    python3 scripts/figure1_forest_plot.py
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COHORT_ORDER = [
    "FengQ_2015",
    "ThomasAM_2018a",
    "ThomasAM_2018b",
    "ThomasAM_2019_c",
    "VogtmannE_2016",
    "YuJ_2015",
    "ZellerG_2014",
]

COHORT_LABEL = {
    "FengQ_2015":      "FengQ 2015 (AUT)",
    "ThomasAM_2018a":  "ThomasAM 2018a (ITA)",
    "ThomasAM_2018b":  "ThomasAM 2018b (ITA)",
    "ThomasAM_2019_c": "ThomasAM 2019c (JPN)",
    "VogtmannE_2016":  "VogtmannE 2016 (USA)",
    "YuJ_2015":        "YuJ 2015 (CHN)",
    "ZellerG_2014":    "ZellerG 2014 (FRA)",
}

MODELS = [
    # (csv_label, display_label, color, marker, vertical_offset)
    ("species_rf", "Species RF",    "#1f77b4", "o", +0.22),
    ("joint_rf",   "Joint RF",      "#b3242c", "s",  0.00),
    ("joint_xgb",  "Joint XGBoost", "#e07a4d", "D", -0.22),
]


def main():
    bc = pd.read_csv("results/bootstrap_ci.csv")

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#f7f5f1")
    ax.set_facecolor("#f7f5f1")

    # y positions: top is high y, so we count down. Reserve y=0 for "Pooled".
    # Cohort i goes at y = (n_cohorts - i) + 1
    n = len(COHORT_ORDER)
    cohort_y = {coh: (n - i) + 1 for i, coh in enumerate(COHORT_ORDER)}
    pooled_y = 0.0

    yticks = []
    yticklabels = []

    # Plot each model series
    for csv_label, disp, color, marker, dy in MODELS:
        sub = bc[bc["model"] == csv_label]
        ys, xs, los, his = [], [], [], []
        for coh in COHORT_ORDER:
            row = sub[sub["cohort"] == coh].iloc[0]
            ys.append(cohort_y[coh] + dy)
            xs.append(row["auc"])
            los.append(row["auc"] - row["ci_lo"])
            his.append(row["ci_hi"] - row["auc"])
        # Pooled
        prow = sub[sub["cohort"] == "pooled"].iloc[0]
        ys.append(pooled_y + dy)
        xs.append(prow["auc"])
        los.append(prow["auc"] - prow["ci_lo"])
        his.append(prow["ci_hi"] - prow["auc"])
        ax.errorbar(
            xs, ys, xerr=[los, his],
            fmt=marker, color=color, ecolor=color, capsize=3,
            elinewidth=1.2, markersize=6.5, label=disp,
        )

    # Y ticks: cohort labels and pooled label
    yticks = [cohort_y[c] for c in COHORT_ORDER] + [pooled_y]
    yticklabels = [COHORT_LABEL[c] for c in COHORT_ORDER] + ["Pooled (n = 646)"]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    # Dashed separator between cohorts and pooled
    ax.axhline((pooled_y + cohort_y[COHORT_ORDER[-1]]) / 2,
               color="gray", linestyle="--", linewidth=0.8)

    # X axis
    ax.set_xlabel("AUC", fontsize=12, fontweight="bold")
    ax.set_xlim(0.55, 0.95)
    ax.set_xticks([0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])

    # Light vertical gridlines
    ax.grid(axis="x", color="gray", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    # Spines: hide top + right
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax.set_title(
        "Figure 1. Per-cohort and pooled AUC with 95% bootstrap CI\n"
        "under LODO cross-validation",
        fontsize=12, fontweight="bold",
    )

    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="lightgray")

    plt.tight_layout()
    out_dir = "manuscript/figures"
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "Figure1_Forest_Plot.png"), dpi=300,
                facecolor=fig.get_facecolor())
    fig.savefig(os.path.join(out_dir, "Figure1_Forest_Plot.pdf"),
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Saved manuscript/figures/Figure1_Forest_Plot.png")
    print("Saved manuscript/figures/Figure1_Forest_Plot.pdf")


if __name__ == "__main__":
    main()
