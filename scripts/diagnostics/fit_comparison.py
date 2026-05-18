"""Head-to-head comparison: species RF microbiome model vs published FIT.

For the Discussion paragraph on "Where does microbiome screening fit
relative to FIT?". The fecal immunochemical test (FIT) is the current
non-invasive gold standard for CRC screening. Imperiale et al. (2014,
NEJM 370:1287) report FIT sensitivity 0.79 / specificity 0.94 for CRC
in an average-risk asymptomatic screening cohort. Specificity for the
advanced-adenoma sub-endpoint is higher (~0.96) but at substantially
lower sensitivity (~0.24); the latter is included here as a separate
row for transparency.

This script anchors a like-for-like comparison: at the same 94%
specificity operating point as FIT, what sensitivity does the species
RF achieve on our pooled LODO predictions? PPV / NPV are computed at
the US population CRC lifetime prevalence of 5% so the rows in the
output table are directly comparable.

Honest reading: at FIT's specificity floor, the current microbiome
model lags FIT on sensitivity. Microbiome screening is best framed as
complementary (e.g., a stratifier of FIT-negative patients) rather
than a replacement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

REPO = Path(__file__).resolve().parents[2]
PREDS = REPO / "results" / "preds_species_rf.csv"
OUT_CSV = REPO / "results" / "diagnostics" / "fit_vs_microbiome.csv"

# Published FIT performance (Imperiale 2014, NEJM 370:1287)
FIT_CRC_SENS = 0.79
FIT_CRC_SPEC = 0.94
FIT_AA_SENS = 0.24   # advanced-adenoma sub-endpoint
FIT_AA_SPEC = 0.96

PREVALENCE = 0.05  # US lifetime CRC prevalence


def ppv_npv(sens: float, spec: float, prev: float) -> tuple[float, float]:
    denom_ppv = sens * prev + (1.0 - spec) * (1.0 - prev)
    denom_npv = spec * (1.0 - prev) + (1.0 - sens) * prev
    ppv = (sens * prev) / denom_ppv if denom_ppv > 0 else float("nan")
    npv = (spec * (1.0 - prev)) / denom_npv if denom_npv > 0 else float("nan")
    return float(ppv), float(npv)


def sens_at_specificity(
    y_true: np.ndarray, y_prob: np.ndarray, target_spec: float
) -> tuple[float, float, float]:
    """Pick the threshold giving the highest sensitivity among those
    that meet target_spec. Returns (threshold, sens, achieved_spec).
    """
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    spec_arr = 1.0 - fpr
    valid = spec_arr >= target_spec
    if not valid.any():
        idx = int(np.argmax(spec_arr))
    else:
        candidates = np.where(valid)[0]
        idx = int(candidates[np.argmax(tpr[candidates])])
    return float(thr[idx]), float(tpr[idx]), float(spec_arr[idx])


def youden(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float, float]:
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thr[idx]), float(tpr[idx]), float(1.0 - fpr[idx])


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PREDS)
    y_true = df["y_true"].to_numpy().astype(int)
    y_prob = df["y_prob"].to_numpy().astype(float)
    n_total = int(len(df))

    # Species RF at FIT's 94% specificity operating point
    thr_94, sens_94, spec_94 = sens_at_specificity(y_true, y_prob, FIT_CRC_SPEC)
    ppv_94, npv_94 = ppv_npv(sens_94, spec_94, PREVALENCE)

    # Species RF at FIT's 96% specificity operating point
    thr_96, sens_96, spec_96 = sens_at_specificity(y_true, y_prob, FIT_AA_SPEC)
    ppv_96, npv_96 = ppv_npv(sens_96, spec_96, PREVALENCE)

    # Species RF at its own Youden-J operating point (for reference)
    thr_yj, sens_yj, spec_yj = youden(y_true, y_prob)
    ppv_yj, npv_yj = ppv_npv(sens_yj, spec_yj, PREVALENCE)

    # FIT (Imperiale 2014) PPV / NPV at 5% prevalence
    fit_crc_ppv, fit_crc_npv = ppv_npv(FIT_CRC_SENS, FIT_CRC_SPEC, PREVALENCE)
    fit_aa_ppv, fit_aa_npv = ppv_npv(FIT_AA_SENS, FIT_AA_SPEC, PREVALENCE)

    rows = [
        {
            "assay": "FIT (CRC endpoint)",
            "sensitivity": round(FIT_CRC_SENS, 4),
            "specificity": round(FIT_CRC_SPEC, 4),
            "ppv_5pct": round(fit_crc_ppv, 4),
            "npv_5pct": round(fit_crc_npv, 4),
            "source": "Imperiale 2014, NEJM 370:1287",
            "notes": "Single FIT vs colonoscopy; average-risk asymptomatic screening cohort (n=9989).",
        },
        {
            "assay": "FIT (advanced adenoma endpoint)",
            "sensitivity": round(FIT_AA_SENS, 4),
            "specificity": round(FIT_AA_SPEC, 4),
            "ppv_5pct": round(fit_aa_ppv, 4),
            "npv_5pct": round(fit_aa_npv, 4),
            "source": "Imperiale 2014, NEJM 370:1287",
            "notes": "FIT detects advanced adenomas poorly; shown for completeness, not directly comparable to CRC row.",
        },
        {
            "assay": "Species RF (at FIT-matched 94% spec)",
            "sensitivity": round(sens_94, 4),
            "specificity": round(spec_94, 4),
            "ppv_5pct": round(ppv_94, 4),
            "npv_5pct": round(npv_94, 4),
            "source": f"This work; pooled LODO n={n_total}; threshold={thr_94:.4f}",
            "notes": "Like-for-like comparison vs FIT CRC row at matched specificity.",
        },
        {
            "assay": "Species RF (at FIT-AA-matched 96% spec)",
            "sensitivity": round(sens_96, 4),
            "specificity": round(spec_96, 4),
            "ppv_5pct": round(ppv_96, 4),
            "npv_5pct": round(npv_96, 4),
            "source": f"This work; pooled LODO n={n_total}; threshold={thr_96:.4f}",
            "notes": "Comparator for the FIT-AA row at matched specificity.",
        },
        {
            "assay": "Species RF (Youden-J)",
            "sensitivity": round(sens_yj, 4),
            "specificity": round(spec_yj, 4),
            "ppv_5pct": round(ppv_yj, 4),
            "npv_5pct": round(npv_yj, 4),
            "source": f"This work; pooled LODO n={n_total}; threshold={thr_yj:.4f}",
            "notes": "Balanced operating point on pooled LODO; not specificity-matched to FIT.",
        },
    ]

    out_df = pd.DataFrame(
        rows,
        columns=[
            "assay",
            "sensitivity",
            "specificity",
            "ppv_5pct",
            "npv_5pct",
            "source",
            "notes",
        ],
    )
    out_df.to_csv(OUT_CSV, index=False)

    print(f"Wrote {OUT_CSV}")
    print(out_df.to_string(index=False))
    print(
        f"\nFIT gap at 94% spec: FIT sens={FIT_CRC_SENS:.2f}, "
        f"species RF sens={sens_94:.4f}  (gap = {FIT_CRC_SENS - sens_94:+.4f})"
    )


if __name__ == "__main__":
    main()
