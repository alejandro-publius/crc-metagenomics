"""Reliability diagrams, Brier scores, and Expected Calibration Error.

Computes calibration diagnostics for pooled LODO predictions from three
models (species RF, joint RF, joint XGB). Writes a metrics CSV and a
3-panel reliability figure with per-panel probability histograms.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUT_CSV = REPO / "results" / "diagnostics" / "calibration_metrics.csv"
OUT_FIG = REPO / "figures" / "diagnostics" / "calibration_reliability.png"

MODELS = [
    ("species_rf", "Species RF", RESULTS / "preds_species_rf.csv"),
    ("joint_rf", "Joint RF", RESULTS / "preds_joint_rf.csv"),
    ("joint_xgb", "Joint XGB", RESULTS / "preds_joint_xgb.csv"),
]

N_BINS = 10


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Equal-width binning ECE."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right-closed except first bin includes left edge
    bin_idx = np.clip(np.digitize(y_prob, edges, right=True) - 1, 0, n_bins - 1)
    ece = 0.0
    n = len(y_prob)
    for b in range(n_bins):
        mask = bin_idx == b
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(y_prob[mask]))
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def reliability_points(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    """Return (mean_pred, frac_pos, counts) per bin (NaN where empty)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, edges, right=True) - 1, 0, n_bins - 1)
    mean_pred = np.full(n_bins, np.nan)
    frac_pos = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = bin_idx == b
        counts[b] = int(mask.sum())
        if counts[b] > 0:
            mean_pred[b] = float(np.mean(y_prob[mask]))
            frac_pos[b] = float(np.mean(y_true[mask]))
    return mean_pred, frac_pos, counts


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    cmap = plt.get_cmap("tab10")

    fig = plt.figure(figsize=(13.5, 6.5))
    # 2 rows x 3 cols; top = reliability, bottom = histogram
    gs = fig.add_gridspec(
        nrows=2, ncols=3, height_ratios=[3.0, 1.0], hspace=0.08, wspace=0.28
    )

    metrics_rows = []
    for i, (key, label, path) in enumerate(MODELS):
        df = pd.read_csv(path)
        y_true = df["y_true"].to_numpy().astype(int)
        y_prob = df["y_prob"].to_numpy().astype(float)
        n = int(len(df))

        brier = float(brier_score_loss(y_true, y_prob))
        ece = expected_calibration_error(y_true, y_prob, N_BINS)
        metrics_rows.append(
            {"model": key, "brier": round(brier, 5), "ece": round(ece, 5), "n": n}
        )

        mean_pred, frac_pos, counts = reliability_points(y_true, y_prob, N_BINS)

        ax = fig.add_subplot(gs[0, i])
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="Perfect")
        valid = ~np.isnan(mean_pred)
        ax.plot(
            mean_pred[valid],
            frac_pos[valid],
            marker="o",
            color=cmap(i),
            linewidth=2,
            markersize=7,
            label=label,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability", fontsize=10)
        if i == 0:
            ax.set_ylabel("Fraction of positives", fontsize=10)
        ax.set_title(
            f"{label}\nBrier = {brier:.4f}    ECE = {ece:.4f}",
            fontsize=11,
        )
        ax.tick_params(labelsize=10)
        ax.legend(loc="upper left", fontsize=9, frameon=True)

        ax_h = fig.add_subplot(gs[1, i], sharex=ax)
        edges = np.linspace(0.0, 1.0, N_BINS + 1)
        widths = np.diff(edges)
        ax_h.bar(
            edges[:-1],
            counts,
            width=widths,
            align="edge",
            color=cmap(i),
            alpha=0.55,
            edgecolor="white",
            linewidth=0.5,
        )
        ax_h.set_xlim(0, 1)
        ax_h.set_xlabel("Predicted probability", fontsize=10)
        if i == 0:
            ax_h.set_ylabel("Count", fontsize=10)
        ax_h.tick_params(labelsize=9)

    out_df = pd.DataFrame(metrics_rows, columns=["model", "brier", "ece", "n"])
    out_df.to_csv(OUT_CSV, index=False)

    fig.suptitle(
        "Reliability diagrams (10-bin equal-width) — pooled LODO predictions",
        fontsize=12,
        y=0.995,
    )
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_FIG}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
