"""Generate Figure 2 (pooled ROC curves) for the manuscript.

Pooled LODO ROC curves for the three headline models (Species RF, Joint
RF, Joint XGBoost). AUC values displayed in the legend are taken from
the pooled rows of ``results/bootstrap_ci.csv`` so the figure stays
consistent with the bootstrap-CI table reported elsewhere.

Reads:
    results/preds_species_rf.csv
    results/preds_joint_rf.csv
    results/preds_joint_xgb.csv
    results/bootstrap_ci.csv

Writes:
    manuscript/figures/Figure2_ROC_Curves.png  (300 DPI raster)
    manuscript/figures/Figure2_ROC_Curves.pdf  (vector)

Usage:
    python3 scripts/figure2_roc_curves.py
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT_DIR = REPO / "manuscript" / "figures"

MODELS = [
    # (csv_label, display_label, preds_csv_path, color)
    ("species_rf", "Species RF",    RESULTS / "preds_species_rf.csv", "#1f77b4"),
    ("joint_rf",   "Joint RF",      RESULTS / "preds_joint_rf.csv",   "#b3242c"),
    ("joint_xgb",  "Joint XGBoost", RESULTS / "preds_joint_xgb.csv",  "#e07a4d"),
]


def pooled_auc_lookup() -> dict[str, float]:
    """Return {model_name: pooled AUC} pulled from bootstrap_ci.csv."""
    bc = pd.read_csv(RESULTS / "bootstrap_ci.csv")
    pooled = bc[bc["cohort"] == "pooled"]
    return {row["model"]: float(row["auc"]) for _, row in pooled.iterrows()}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    auc_pooled = pooled_auc_lookup()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.0, 6.5))
    fig.patch.set_facecolor("#f7f5f1")
    ax.set_facecolor("#f7f5f1")

    for csv_label, disp, preds_path, color in MODELS:
        df = pd.read_csv(preds_path)
        y_true = df["y_true"].to_numpy().astype(int)
        y_prob = df["y_prob"].to_numpy().astype(float)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_disp = auc_pooled.get(csv_label, float("nan"))
        ax.plot(
            fpr, tpr,
            color=color, linewidth=2.0,
            label=f"{disp} (AUC = {auc_disp:.3f})",
        )

    # Chance line
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1.0,
            label="Chance")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.005)
    ax.set_xlabel("False positive rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True positive rate", fontsize=12, fontweight="bold")
    ax.set_title(
        "Figure 2. Pooled LODO ROC curves\n"
        "(10-cohort hold-out predictions; AUC from bootstrap_ci.csv)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="lower right", frameon=True, facecolor="white",
              edgecolor="lightgray", fontsize=10)
    ax.tick_params(labelsize=10)

    # Hide top + right spines for a cleaner look (matches Figure 1).
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    png_path = OUT_DIR / "Figure2_ROC_Curves.png"
    pdf_path = OUT_DIR / "Figure2_ROC_Curves.pdf"
    fig.savefig(png_path, dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
