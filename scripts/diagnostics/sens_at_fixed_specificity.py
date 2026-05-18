"""Sensitivity at clinically-meaningful specificity operating points.

For each of the three pooled LODO models (species RF, joint RF, joint
XGB), find the score threshold that achieves a target specificity of
0.90 and 0.95 — the two specificity floors most commonly cited for
screening tools. At each operating point, compute:

    - achieved specificity (the score threshold sweep is discrete, so
      we pick the threshold giving the largest specificity <= target,
      breaking ties toward higher sensitivity)
    - sensitivity at that threshold
    - PPV and NPV at US population CRC prevalence (5%) via Bayes' rule
    - Number-needed-to-test = 1 / PPV (expected screens per true
      positive at population prevalence)

Outputs:
    results/diagnostics/sens_at_fixed_spec.csv
    figures/diagnostics/sens_at_fixed_spec.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUT_CSV = REPO / "results" / "diagnostics" / "sens_at_fixed_spec.csv"
OUT_FIG = REPO / "figures" / "diagnostics" / "sens_at_fixed_spec.png"

MODELS = [
    ("species_rf", "Species RF", RESULTS / "preds_species_rf.csv"),
    ("joint_rf", "Joint RF", RESULTS / "preds_joint_rf.csv"),
    ("joint_xgb", "Joint XGB", RESULTS / "preds_joint_xgb.csv"),
]

TARGET_SPECS = [0.90, 0.95]
PREVALENCE = 0.05  # US lifetime CRC prevalence (for PPV/NPV anchoring)


def ppv_npv(sens: float, spec: float, prev: float) -> tuple[float, float]:
    denom_ppv = sens * prev + (1.0 - spec) * (1.0 - prev)
    denom_npv = spec * (1.0 - prev) + (1.0 - sens) * prev
    ppv = (sens * prev) / denom_ppv if denom_ppv > 0 else float("nan")
    npv = (spec * (1.0 - prev)) / denom_npv if denom_npv > 0 else float("nan")
    return float(ppv), float(npv)


def threshold_for_specificity(
    y_true: np.ndarray, y_prob: np.ndarray, target_spec: float
) -> tuple[float, float, float]:
    """Pick the threshold whose achieved specificity is closest to the
    target while not falling below it. If no threshold reaches the
    target (rare), pick the one with the highest achieved specificity.
    Returns (threshold, achieved_sens, achieved_spec).
    """
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    spec_arr = 1.0 - fpr  # one per threshold

    valid = spec_arr >= target_spec
    if not valid.any():
        # Fallback: take the maximum-specificity threshold available
        idx = int(np.argmax(spec_arr))
    else:
        # Among thresholds that meet target_spec, take the one with the
        # highest sensitivity (i.e., the smallest threshold satisfying
        # the specificity floor).
        idx_candidates = np.where(valid)[0]
        idx = int(idx_candidates[np.argmax(tpr[idx_candidates])])

    return float(thr[idx]), float(tpr[idx]), float(spec_arr[idx])


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for key, label, path in MODELS:
        df = pd.read_csv(path)
        y_true = df["y_true"].to_numpy().astype(int)
        y_prob = df["y_prob"].to_numpy().astype(float)
        for target in TARGET_SPECS:
            thr, sens, spec = threshold_for_specificity(y_true, y_prob, target)
            ppv, npv = ppv_npv(sens, spec, PREVALENCE)
            nntt = (1.0 / ppv) if ppv > 0 else float("nan")
            rows.append(
                {
                    "model": key,
                    "target_spec": round(float(target), 4),
                    "achieved_spec": round(spec, 4),
                    "threshold": round(thr, 4),
                    "sensitivity": round(sens, 4),
                    "ppv_5pct": round(ppv, 4),
                    "npv_5pct": round(npv, 4),
                    "number_needed_to_test_5pct": round(nntt, 2),
                }
            )

    out_df = pd.DataFrame(
        rows,
        columns=[
            "model",
            "target_spec",
            "achieved_spec",
            "threshold",
            "sensitivity",
            "ppv_5pct",
            "npv_5pct",
            "number_needed_to_test_5pct",
        ],
    )
    out_df.to_csv(OUT_CSV, index=False)

    # Plot: grouped bars, one group per model, two bars per group
    plt.style.use("seaborn-v0_8-whitegrid")
    c90 = "#0072B2"  # blue (colorblind-safe)
    c95 = "#E69F00"  # orange (colorblind-safe)

    model_keys = [m[0] for m in MODELS]
    model_labels = [m[1] for m in MODELS]
    sens90 = [out_df[(out_df.model == m) & (out_df.target_spec == 0.90)]
              ["sensitivity"].iloc[0] for m in model_keys]
    sens95 = [out_df[(out_df.model == m) & (out_df.target_spec == 0.95)]
              ["sensitivity"].iloc[0] for m in model_keys]
    ach90 = [out_df[(out_df.model == m) & (out_df.target_spec == 0.90)]
             ["achieved_spec"].iloc[0] for m in model_keys]
    ach95 = [out_df[(out_df.model == m) & (out_df.target_spec == 0.95)]
             ["achieved_spec"].iloc[0] for m in model_keys]

    x = np.arange(len(model_keys))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    b1 = ax.bar(
        x - width / 2,
        sens90,
        width,
        label="Sensitivity at spec >= 0.90",
        color=c90,
        edgecolor="white",
        linewidth=0.7,
    )
    b2 = ax.bar(
        x + width / 2,
        sens95,
        width,
        label="Sensitivity at spec >= 0.95",
        color=c95,
        edgecolor="white",
        linewidth=0.7,
    )

    for bar, sens, spec in zip(b1, sens90, ach90):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            sens + 0.012,
            f"{sens:.2f}\n(spec {spec:.2f})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar, sens, spec in zip(b2, sens95, ach95):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            sens + 0.012,
            f"{sens:.2f}\n(spec {spec:.2f})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Reference: published FIT sensitivity (~0.79) at spec ~0.94 (Imperiale 2014)
    ax.axhline(
        0.79,
        color="#009E73",  # bluish green (colorblind-safe)
        linestyle="--",
        linewidth=1.2,
        label="FIT sensitivity ~0.79 (Imperiale 2014, spec ~0.94)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=11)
    ax.set_ylabel("Sensitivity at fixed specificity", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_title(
        "Sensitivity at clinically-meaningful specificity floors\n(pooled LODO predictions, n = 1339)",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    ax.tick_params(axis="y", labelsize=10)

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_FIG}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
