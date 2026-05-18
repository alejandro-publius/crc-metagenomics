"""Base-rate-adjusted PPV / NPV for the species RF screening model.

The pooled LODO dataset is balanced (~50/50 case/control), but the US
population prevalence of CRC is far lower (~5% lifetime, ~0.5% annual
incidence in screening-age adults). Sensitivity and specificity are
prevalence-invariant, but PPV and NPV are not. This script holds the
species RF operating point (Youden-J on the pooled predictions) fixed
and reweights PPV / NPV across a prevalence sweep using Bayes' rule:

    PPV = (sens * prev) / (sens * prev + (1 - spec) * (1 - prev))
    NPV = (spec * (1 - prev)) / (spec * (1 - prev) + (1 - sens) * prev)

At 5% population prevalence with the Youden operating point, PPV is
~10-12%, meaning the model alone is insufficient as a primary screen
but could serve as a triage layer (e.g., stratifying who should be
prioritized for a confirmatory test such as colonoscopy).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

REPO = Path(__file__).resolve().parents[2]
PREDS = REPO / "results" / "preds_species_rf.csv"
OUT_CSV = REPO / "results" / "diagnostics" / "base_rate_ppv.csv"
OUT_FIG = REPO / "figures" / "diagnostics" / "base_rate_ppv.png"


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def sens_spec_at(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> tuple[float, float]:
    pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return float(sens), float(spec)


def ppv_npv(sens: float, spec: float, prev: float) -> tuple[float, float]:
    denom_ppv = sens * prev + (1.0 - spec) * (1.0 - prev)
    denom_npv = spec * (1.0 - prev) + (1.0 - sens) * prev
    ppv = (sens * prev) / denom_ppv if denom_ppv > 0 else float("nan")
    npv = (spec * (1.0 - prev)) / denom_npv if denom_npv > 0 else float("nan")
    return float(ppv), float(npv)


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PREDS)
    y_true = df["y_true"].to_numpy().astype(int)
    y_prob = df["y_prob"].to_numpy().astype(float)

    thr = youden_threshold(y_true, y_prob)
    sens, spec = sens_spec_at(y_true, y_prob, thr)

    prevalences = np.linspace(0.005, 0.50, 12)

    rows = []
    for p in prevalences:
        ppv, npv = ppv_npv(sens, spec, float(p))
        rows.append(
            {
                "prevalence": round(float(p), 4),
                "ppv": round(ppv, 4),
                "npv": round(npv, 4),
                "sens_fixed": round(sens, 4),
                "spec_fixed": round(spec, 4),
            }
        )

    out_df = pd.DataFrame(rows, columns=["prevalence", "ppv", "npv", "sens_fixed", "spec_fixed"])
    out_df.to_csv(OUT_CSV, index=False)

    # Figure: 2-panel PPV vs prevalence and NPV vs prevalence
    plt.style.use("seaborn-v0_8-whitegrid")
    # Colorblind-safe palette (Wong / Okabe-Ito)
    c_ppv = "#0072B2"  # blue
    c_npv = "#D55E00"  # vermillion
    c_ref = "#999999"

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))

    # Dense sweep for smooth curves (overlays the 12 sampled points)
    dense = np.linspace(0.001, 0.60, 600)
    ppv_dense = np.array([ppv_npv(sens, spec, float(p))[0] for p in dense])
    npv_dense = np.array([ppv_npv(sens, spec, float(p))[1] for p in dense])

    axes[0].plot(dense, ppv_dense, color=c_ppv, linewidth=2, label="PPV")
    axes[0].plot(
        out_df["prevalence"],
        out_df["ppv"],
        marker="o",
        linestyle="none",
        color=c_ppv,
        markersize=6,
        markeredgecolor="white",
        markeredgewidth=0.8,
        label="Sampled prevalences",
    )
    axes[0].axvline(0.05, color=c_ref, linestyle="--", linewidth=1.2,
                    label="US lifetime CRC (~5%)")
    axes[0].axvline(0.005, color=c_ref, linestyle=":", linewidth=1.2,
                    label="Annual screening (~0.5%)")
    axes[0].set_xlabel("Population CRC prevalence", fontsize=11)
    axes[0].set_ylabel("Positive predictive value", fontsize=11)
    axes[0].set_xlim(0, 0.55)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title(
        f"PPV vs prevalence  (sens={sens:.3f}, spec={spec:.3f})",
        fontsize=11,
    )
    axes[0].legend(loc="lower right", fontsize=9, frameon=True)
    axes[0].tick_params(labelsize=10)

    axes[1].plot(dense, npv_dense, color=c_npv, linewidth=2, label="NPV")
    axes[1].plot(
        out_df["prevalence"],
        out_df["npv"],
        marker="s",
        linestyle="none",
        color=c_npv,
        markersize=6,
        markeredgecolor="white",
        markeredgewidth=0.8,
        label="Sampled prevalences",
    )
    axes[1].axvline(0.05, color=c_ref, linestyle="--", linewidth=1.2,
                    label="US lifetime CRC (~5%)")
    axes[1].axvline(0.005, color=c_ref, linestyle=":", linewidth=1.2,
                    label="Annual screening (~0.5%)")
    axes[1].set_xlabel("Population CRC prevalence", fontsize=11)
    axes[1].set_ylabel("Negative predictive value", fontsize=11)
    axes[1].set_xlim(0, 0.55)
    axes[1].set_ylim(0.5, 1.005)
    axes[1].set_title(
        f"NPV vs prevalence  (sens={sens:.3f}, spec={spec:.3f})",
        fontsize=11,
    )
    axes[1].legend(loc="lower left", fontsize=9, frameon=True)
    axes[1].tick_params(labelsize=10)

    fig.suptitle(
        "Species RF (Youden-J fixed): base-rate-adjusted PPV and NPV",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Headline at 5%
    ppv5, npv5 = ppv_npv(sens, spec, 0.05)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_FIG}")
    print(f"Threshold (Youden-J) = {thr:.4f}")
    print(f"Observed pooled sens = {sens:.4f}, spec = {spec:.4f}")
    print(f"At prevalence 5%: PPV = {ppv5:.4f}, NPV = {npv5:.4f}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
