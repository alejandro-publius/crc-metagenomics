"""Pooled ROC and Precision-Recall curves for the three models.

Produces a 2-panel figure (ROC, PR) with one curve per model annotated by
its AUC / AUPRC, plus a CSV of pooled AUROC and AUPRC values.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUT_CSV = REPO / "results" / "diagnostics" / "auprc_pooled.csv"
OUT_FIG = REPO / "figures" / "diagnostics" / "roc_pr_pooled.png"

MODELS = [
    ("species_rf", "Species RF", RESULTS / "preds_species_rf.csv"),
    ("joint_rf", "Joint RF", RESULTS / "preds_joint_rf.csv"),
    ("joint_xgb", "Joint XGB", RESULTS / "preds_joint_xgb.csv"),
]


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    cmap = plt.get_cmap("tab10")

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12.0, 5.5))

    rows = []
    for i, (key, label, path) in enumerate(MODELS):
        df = pd.read_csv(path)
        y_true = df["y_true"].to_numpy().astype(int)
        y_prob = df["y_prob"].to_numpy().astype(float)
        n = int(len(df))

        auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))
        rows.append(
            {
                "model": key,
                "auroc": round(auroc, 4),
                "auprc": round(auprc, 4),
                "n": n,
            }
        )

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax_roc.plot(
            fpr,
            tpr,
            color=cmap(i),
            linewidth=2,
            label=f"{label} (AUC = {auroc:.3f})",
        )

        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ax_pr.plot(
            rec,
            prec,
            color=cmap(i),
            linewidth=2,
            label=f"{label} (AUPRC = {auprc:.3f})",
        )

    # ROC chance line
    ax_roc.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="Chance")
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1.005)
    ax_roc.set_xlabel("False positive rate", fontsize=11)
    ax_roc.set_ylabel("True positive rate", fontsize=11)
    ax_roc.set_title("ROC — pooled LODO predictions", fontsize=11)
    ax_roc.legend(loc="lower right", fontsize=10, frameon=True)
    ax_roc.tick_params(labelsize=10)

    # PR baseline = positive prevalence (use first model; all share pool)
    prevalence = float(np.mean(pd.read_csv(MODELS[0][2])["y_true"]))
    ax_pr.axhline(
        prevalence,
        color="grey",
        linestyle="--",
        linewidth=1,
        label=f"Prevalence = {prevalence:.3f}",
    )
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1.005)
    ax_pr.set_xlabel("Recall", fontsize=11)
    ax_pr.set_ylabel("Precision", fontsize=11)
    ax_pr.set_title("Precision-Recall — pooled LODO predictions", fontsize=11)
    ax_pr.legend(loc="lower left", fontsize=10, frameon=True)
    ax_pr.tick_params(labelsize=10)

    out_df = pd.DataFrame(rows, columns=["model", "auroc", "auprc", "n"])
    out_df.to_csv(OUT_CSV, index=False)

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_FIG}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
