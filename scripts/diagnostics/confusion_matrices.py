"""Confusion matrices at Youden-J and 0.5 thresholds.

For each of three pooled LODO models, build confusion matrices at the
Youden-J optimal threshold (argmax of TPR-FPR on the ROC curve) and at a
fixed 0.5 threshold. Reports sensitivity, specificity, PPV, NPV, F1, MCC.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUT_CSV = REPO / "results" / "diagnostics" / "confusion_matrices.csv"
OUT_FIG = REPO / "figures" / "diagnostics" / "confusion_matrices.png"

MODELS = [
    ("species_rf", "Species RF", RESULTS / "preds_species_rf.csv"),
    ("joint_rf", "Joint RF", RESULTS / "preds_joint_rf.csv"),
    ("joint_xgb", "Joint XGB", RESULTS / "preds_joint_xgb.csv"),
]


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def metrics_at(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> dict:
    pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    denom = float(np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else float("nan")
    return {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
        "mcc": mcc,
    }


def _round_metrics(d: dict) -> dict:
    out = dict(d)
    for k in ("sensitivity", "specificity", "ppv", "npv", "f1", "mcc"):
        out[k] = round(float(d[k]), 4)
    return out


def _draw_cm(ax, cm: np.ndarray, title: str) -> None:
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=10)
    ax.set_yticklabels(["True 0", "True 1"], fontsize=10)
    ax.set_title(title, fontsize=10)
    total = cm.sum()
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(2):
        for j in range(2):
            val = int(cm[i, j])
            pct = (val / total * 100.0) if total else 0.0
            ax.text(
                j,
                i,
                f"{val}\n({pct:.1f}%)",
                ha="center",
                va="center",
                fontsize=11,
                color="white" if val > thresh else "black",
            )
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")

    rows = []
    fig, axes = plt.subplots(3, 2, figsize=(8.0, 11.0))

    for r, (key, label, path) in enumerate(MODELS):
        df = pd.read_csv(path)
        y_true = df["y_true"].to_numpy().astype(int)
        y_prob = df["y_prob"].to_numpy().astype(float)
        n = int(len(df))

        thr_y = youden_threshold(y_true, y_prob)
        for c, (ttype, thr) in enumerate(
            [("youden_j", thr_y), ("fixed_0.5", 0.5)]
        ):
            m = metrics_at(y_true, y_prob, thr)
            rows.append(
                {
                    "model": key,
                    "threshold_type": ttype,
                    "threshold": round(float(thr), 4),
                    **_round_metrics(m),
                    "n": n,
                }
            )
            cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], dtype=int)
            ax = axes[r, c]
            ttitle = "Youden-J" if ttype == "youden_j" else "Threshold = 0.5"
            title = (
                f"{label} — {ttitle} (t={thr:.3f})\n"
                f"Sens={m['sensitivity']:.3f}  Spec={m['specificity']:.3f}  "
                f"MCC={m['mcc']:.3f}"
            )
            _draw_cm(ax, cm, title)

    cols = [
        "model",
        "threshold_type",
        "threshold",
        "tp",
        "fp",
        "tn",
        "fn",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "f1",
        "mcc",
        "n",
    ]
    out_df = pd.DataFrame(rows)[cols]
    out_df.to_csv(OUT_CSV, index=False)

    fig.suptitle(
        "Confusion matrices — pooled LODO predictions",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_FIG}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
