"""Per-cohort operating characteristics for species RF at the pooled Youden-J threshold.

Uses the pooled LODO Youden-J threshold (held constant across cohorts) and
reports sensitivity, specificity, PPV, NPV per cohort, plus a paired bar
chart of sensitivity vs specificity.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

REPO = Path(__file__).resolve().parents[2]
PREDS = REPO / "results" / "preds_species_rf.csv"
OUT_CSV = REPO / "results" / "diagnostics" / "per_cohort_operating_chars.csv"
OUT_FIG = REPO / "figures" / "diagnostics" / "per_cohort_sens_spec.png"


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def _ops(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> dict:
    pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    return {
        "n": int(len(y_true)),
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
    }


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PREDS)
    y_true_all = df["y_true"].to_numpy().astype(int)
    y_prob_all = df["y_prob"].to_numpy().astype(float)

    thr = youden_threshold(y_true_all, y_prob_all)

    rows = []
    pooled = _ops(y_true_all, y_prob_all, thr)
    pooled_row = {"cohort": "POOLED", "threshold": round(thr, 4), **pooled}
    rows.append(pooled_row)

    for cohort, sub in df.groupby("cohort"):
        yt = sub["y_true"].to_numpy().astype(int)
        yp = sub["y_prob"].to_numpy().astype(float)
        ops = _ops(yt, yp, thr)
        rows.append({"cohort": cohort, "threshold": round(thr, 4), **ops})

    out_df = pd.DataFrame(rows)
    for c in ("sensitivity", "specificity", "ppv", "npv"):
        out_df[c] = out_df[c].astype(float).round(4)
    cols = [
        "cohort",
        "threshold",
        "n",
        "n_pos",
        "n_neg",
        "tp",
        "fp",
        "tn",
        "fn",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
    ]
    out_df = out_df[cols]
    out_df.to_csv(OUT_CSV, index=False)

    # Plot: paired bars per cohort (exclude pooled summary row from bars)
    plt.style.use("seaborn-v0_8-whitegrid")
    cmap = plt.get_cmap("tab10")

    per_cohort = out_df[out_df["cohort"] != "POOLED"].copy()
    per_cohort = per_cohort.sort_values("cohort").reset_index(drop=True)

    cohorts = per_cohort["cohort"].tolist()
    sens = per_cohort["sensitivity"].to_numpy()
    spec = per_cohort["specificity"].to_numpy()
    ns = per_cohort["n"].to_numpy()

    x = np.arange(len(cohorts))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(9.0, 0.95 * len(cohorts) + 3), 5.5))

    b1 = ax.bar(
        x - width / 2,
        sens,
        width,
        label="Sensitivity",
        color=cmap(0),
        edgecolor="white",
        linewidth=0.6,
    )
    b2 = ax.bar(
        x + width / 2,
        spec,
        width,
        label="Specificity",
        color=cmap(1),
        edgecolor="white",
        linewidth=0.6,
    )

    for bars, vals in ((b1, sens), (b2, spec)):
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.012,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.axhline(
        pooled["sensitivity"],
        color=cmap(0),
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label=f"Pooled sens = {pooled['sensitivity']:.3f}",
    )
    ax.axhline(
        pooled["specificity"],
        color=cmap(1),
        linestyle=":",
        linewidth=1,
        alpha=0.7,
        label=f"Pooled spec = {pooled['specificity']:.3f}",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(cohorts, ns)], rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Value", fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_title(
        f"Species RF — per-cohort operating characteristics at pooled Youden-J (t={thr:.3f})",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.tick_params(axis="y", labelsize=10)

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_FIG}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
