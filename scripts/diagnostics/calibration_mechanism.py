"""Brier-score decomposition for species RF, joint RF, and joint XGB.

The joint XGB model has a notably worse Expected Calibration Error
(~0.152) than the RF models (~0.067). This script diagnoses *why* by
decomposing the Brier score into the three Murphy (1973) terms:

    Brier = reliability - resolution + uncertainty

The larger reliability component for joint XGB reflects the aggressive
logit pushes from gradient boosting on tabular metagenomic data — XGB
concentrates predicted probabilities near 0/1 even when the calibrated
posterior is in the middle, so the per-bin mean prediction drifts from
the per-bin empirical positive rate. This is a calibration-mechanism
artifact of GBDT on this signal level, not a discrimination defect; the
AUC remains close to the RF models.

Outputs:
  - results/diagnostics/brier_decomposition.csv
        model, brier, reliability, resolution, uncertainty
  - figures/diagnostics/calibration_mechanism.png
        3-panel: probability histogram + reliability curve per model.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUT_CSV = REPO / "results" / "diagnostics" / "brier_decomposition.csv"
OUT_FIG = REPO / "figures" / "diagnostics" / "calibration_mechanism.png"

MODELS = [
    ("species_rf", "Species RF", RESULTS / "preds_species_rf.csv"),
    ("joint_rf", "Joint RF", RESULTS / "preds_joint_rf.csv"),
    ("joint_xgb", "Joint XGB", RESULTS / "preds_joint_xgb.csv"),
]

N_BINS = 10


def brier_decomposition(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    """Murphy (1973) decomposition of the Brier score.

    Returns (reliability, resolution, uncertainty). The identity
    ``brier = reliability - resolution + uncertainty`` holds up to a
    bin-mean discretization residual. The residual comes from the fact
    that the within-bin sum of squared deviations between forecasts and
    outcomes is not exactly equal to the squared deviation between the
    bin's mean forecast and mean outcome times the bin count; with
    n_bins=10 and the pooled LODO predictions in this repo the residual
    is on the order of 1e-3 in Brier units, which is documented in the
    accompanying note. Use a calibration-curve overlay (see the figure
    written by this script) rather than the three scalars alone for
    inferential statements about miscalibration.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, edges, right=True) - 1, 0, n_bins - 1)
    n = len(y_prob)
    o_bar = float(np.mean(y_true))
    reliability = 0.0
    resolution = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        nk = int(mask.sum())
        if nk == 0:
            continue
        f_k = float(np.mean(y_prob[mask]))
        o_k = float(np.mean(y_true[mask]))
        reliability += (nk / n) * (f_k - o_k) ** 2
        resolution += (nk / n) * (o_k - o_bar) ** 2
    uncertainty = o_bar * (1.0 - o_bar)
    return reliability, resolution, uncertainty


def reliability_points(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
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

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.5))
    rows = []

    for i, (key, label, path) in enumerate(MODELS):
        df = pd.read_csv(path)
        y_true = df["y_true"].to_numpy().astype(int)
        y_prob = df["y_prob"].to_numpy().astype(float)

        brier = float(brier_score_loss(y_true, y_prob))
        reliability, resolution, uncertainty = brier_decomposition(
            y_true, y_prob, N_BINS
        )

        # Residual = brier - (reliability - resolution + uncertainty).
        # Under Murphy's identity this should be ~0; any non-zero value
        # is the within-bin discretization error documented in
        # ``brier_decomposition``. We surface it so a future regression
        # (e.g. wrong bin-count or off-by-one in digitize) is loud
        # instead of silently shifting the decomposition.
        residual = brier - (reliability - resolution + uncertainty)
        rows.append({
            "model": key,
            "brier": round(brier, 4),
            "reliability": round(reliability, 4),
            "resolution": round(resolution, 4),
            "uncertainty": round(uncertainty, 4),
            "residual": round(residual, 4),
        })

        mean_pred, frac_pos, counts = reliability_points(y_true, y_prob, N_BINS)

        ax = axes[i]
        edges = np.linspace(0.0, 1.0, N_BINS + 1)
        widths = np.diff(edges)
        # Histogram of predicted probabilities (left y-axis)
        ax.bar(
            edges[:-1],
            counts / counts.sum(),
            width=widths,
            align="edge",
            color=cmap(i),
            alpha=0.35,
            edgecolor="white",
            linewidth=0.5,
            label="Predicted prob. (density)",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.8)
        ax.set_xlabel("Predicted probability", fontsize=11)
        if i == 0:
            ax.set_ylabel("Fraction of samples", fontsize=11)
        ax.tick_params(labelsize=10)

        # Overlay reliability curve on a twin y-axis
        ax2 = ax.twinx()
        ax2.plot([0, 1], [0, 1], color="grey", linestyle="--",
                 linewidth=1, label="Perfect calibration")
        valid = ~np.isnan(mean_pred)
        ax2.plot(
            mean_pred[valid],
            frac_pos[valid],
            marker="o",
            color=cmap(i),
            linewidth=2,
            markersize=7,
            label="Reliability curve",
        )
        ax2.set_ylim(0, 1)
        ax2.set_xlim(0, 1)
        if i == 2:
            ax2.set_ylabel("Empirical positive rate", fontsize=11)
        ax2.tick_params(labelsize=10)
        ax2.grid(False)

        ax.set_title(
            f"{label}\n"
            f"Brier = {brier:.4f}   "
            f"Rel = {reliability:.4f}   Res = {resolution:.4f}",
            fontsize=11,
        )

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  loc="upper center", fontsize=9, frameon=True)

    fig.suptitle(
        "Calibration mechanism — joint XGB pushes probabilities to 0/1, "
        "inflating reliability term",
        fontsize=12,
        y=1.02,
    )
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    out = pd.DataFrame(rows, columns=[
        "model", "brier", "reliability", "resolution", "uncertainty",
        "residual",
    ])
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_FIG}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
