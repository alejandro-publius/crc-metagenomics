#!/usr/bin/env python3
"""Plot observed target AUC against two pre-label performance estimates."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


ROOT = Path(__file__).resolve().parents[1]
data = pd.read_csv(ROOT / "results/generalization_risk/outer_cohort_predictions.csv")
out = ROOT / "manuscript/generalization_risk/figures"
out.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharex=True, sharey=True)
panels = [
    ("historical_mean_estimate", "A  Historical model mean"),
    ("unlabeled_risk_estimate", "B  Unlabeled risk model"),
]
colors = dict(zip(sorted(data.cohort.unique()), plt.cm.tab10.colors))
for ax, (column, title) in zip(axes, panels):
    for cohort, group in data.groupby("cohort"):
        ax.scatter(group[column], group.observed_auc, s=24, alpha=0.78,
                   color=colors[cohort], edgecolor="none")
    ax.plot([0.45, 0.95], [0.45, 0.95], color="#4B5563", lw=1,
            linestyle="--")
    mae = mean_absolute_error(data.observed_auc, data[column])
    ax.text(0.03, 0.95, f"MAE = {mae:.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=10)
    ax.set_title(title, loc="left", fontsize=11, weight="bold")
    ax.set_xlabel("Estimated target AUC before labels")
    ax.grid(color="#E5E7EB", linewidth=0.7)
axes[0].set_ylabel("Observed held-out target AUC")
axes[0].set_xlim(0.45, 0.95)
axes[0].set_ylim(0.45, 0.95)
fig.suptitle("Unlabeled shift signals do not improve target-performance estimation",
             x=0.08, ha="left", fontsize=13, weight="bold")
fig.text(0.08, 0.01,
         "Each point is one frozen model × target cohort; outer evaluation leaves the entire cohort out.",
         fontsize=8.5, color="#4B5563")
fig.tight_layout(rect=[0, 0.05, 1, 0.94])
fig.savefig(out / "GeneralizationRisk.png", dpi=300, bbox_inches="tight")
fig.savefig(out / "GeneralizationRisk.pdf", bbox_inches="tight")
