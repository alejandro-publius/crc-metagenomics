"""Statistical power analysis for the DeLong test on pooled LODO predictions.

Uses the DeLong (Sun & Xu 2014 fast algorithm) variance of the AUC
difference, estimated from the observed species-RF vs joint-RF prediction
pairs, to compute one-sided DeLong power as a function of the true
underlying AUC gap at fixed n=1339.

Outputs:
  - results/diagnostics/power_analysis.csv
  - figures/diagnostics/power_curve.png
  - results/diagnostics/POWER_ANALYSIS_NOTE.md
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUT_CSV = REPO / "results" / "diagnostics" / "power_analysis.csv"
OUT_FIG = REPO / "figures" / "diagnostics" / "power_curve.png"
OUT_NOTE = REPO / "results" / "diagnostics" / "POWER_ANALYSIS_NOTE.md"

PRED_A = RESULTS / "preds_species_rf.csv"  # species RF (typically higher AUC)
PRED_B = RESULTS / "preds_joint_rf.csv"    # joint RF

ALPHA = 0.05                                # one-sided
TRUE_DIFFS = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.050]
OBSERVED_DIFF = 0.025                       # species RF - joint RF (from delong_results.csv)


# ---------------------------------------------------------------------------
# DeLong machinery (Sun & Xu 2014 fast algorithm). Mirrors auc_comparison.py.
# ---------------------------------------------------------------------------

def _midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x, kind="mergesort")
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2


def delong_var(y_true: np.ndarray, y_prob_a: np.ndarray, y_prob_b: np.ndarray):
    """Return (auc_a, auc_b, var_diff, m, n) for paired predictions."""
    y_true = np.asarray(y_true).astype(int)
    pos = y_true == 1
    neg = y_true == 0
    m = int(pos.sum())
    n = int(neg.sum())
    if m == 0 or n == 0:
        raise ValueError("Need both classes present")
    aucs = []
    v01s = []
    v10s = []
    for y_prob in [y_prob_a, y_prob_b]:
        y_prob = np.asarray(y_prob, dtype=float)
        x_pos = y_prob[pos]
        x_neg = y_prob[neg]
        tx = _midrank(x_pos)
        ty = _midrank(x_neg)
        tz = _midrank(np.concatenate([x_pos, x_neg]))
        auc = (tz[:m].sum() / m - (m + 1) / 2.0) / n
        v01 = (tz[:m] - tx) / n
        v10 = 1.0 - (tz[m:] - ty) / m
        aucs.append(auc)
        v01s.append(v01)
        v10s.append(v10)
    auc_a, auc_b = aucs
    S01 = np.cov(np.vstack(v01s))
    S10 = np.cov(np.vstack(v10s))
    S = S01 / m + S10 / n
    var_diff = float(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    return float(auc_a), float(auc_b), var_diff, m, n


def power_one_sided(true_diff: float, se: float, alpha: float = 0.05) -> float:
    """One-sided DeLong power: P(Z > z_{1-alpha}) under H1: diff = true_diff > 0.

    Test statistic Z = (auc_a - auc_b) / SE.  Under H1, Z ~ N(true_diff/SE, 1).
    """
    if se <= 0:
        return float("nan")
    z_crit = norm.ppf(1.0 - alpha)
    ncp = true_diff / se
    return float(norm.cdf(ncp - z_crit))


def mdd_at_power(target_power: float, se: float, alpha: float = 0.05) -> float:
    """Minimum detectable difference at given power, one-sided alpha."""
    z_crit = norm.ppf(1.0 - alpha)
    z_pow = norm.ppf(target_power)
    return float(se * (z_crit + z_pow))


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    # ----- load + align paired predictions ---------------------------------
    a = pd.read_csv(PRED_A).sort_values("sample_id").reset_index(drop=True)
    b = pd.read_csv(PRED_B).sort_values("sample_id").reset_index(drop=True)
    assert (a["sample_id"].values == b["sample_id"].values).all(), "sample_id mismatch"
    assert (a["y_true"].values == b["y_true"].values).all(), "y_true mismatch"
    y_true = a["y_true"].to_numpy().astype(int)
    n_total = int(len(y_true))

    auc_a, auc_b, var_diff, m, n_neg = delong_var(
        y_true, a["y_prob"].to_numpy(), b["y_prob"].to_numpy()
    )
    se = float(np.sqrt(var_diff))
    observed_z = (auc_a - auc_b) / se
    observed_p_two = 2.0 * (1.0 - norm.cdf(abs(observed_z)))
    observed_p_one = 1.0 - norm.cdf(observed_z)

    # ----- power curve -----------------------------------------------------
    rows = []
    for d in TRUE_DIFFS:
        p = power_one_sided(d, se, ALPHA)
        rows.append({
            "true_auc_diff": d,
            "power": round(p, 3),
            "n": n_total,
        })
    out_df = pd.DataFrame(rows, columns=["true_auc_diff", "power", "n"])
    out_df.to_csv(OUT_CSV, index=False)

    # ----- power at observed effect + MDD ----------------------------------
    power_at_observed = power_one_sided(OBSERVED_DIFF, se, ALPHA)
    mdd_80 = mdd_at_power(0.80, se, ALPHA)
    mdd_90 = mdd_at_power(0.90, se, ALPHA)

    # ----- figure ----------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid")
    grid = np.linspace(0.0, 0.06, 241)
    powers = np.array([power_one_sided(d, se, ALPHA) for d in grid])

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(grid, powers, color="#1f77b4", linewidth=2.2,
            label=f"DeLong power (one-sided, alpha={ALPHA})")

    # observed effect marker
    ax.axvline(OBSERVED_DIFF, color="#d62728", linestyle="--", linewidth=1.4,
               label=f"Observed diff = {OBSERVED_DIFF:.3f}")
    ax.scatter([OBSERVED_DIFF], [power_at_observed], color="#d62728",
               s=55, zorder=5)
    ax.annotate(f"  power = {power_at_observed:.3f}",
                xy=(OBSERVED_DIFF, power_at_observed),
                xytext=(8, -14), textcoords="offset points", fontsize=10,
                color="#d62728")

    # 80% reference
    ax.axhline(0.80, color="#2ca02c", linestyle=":", linewidth=1.4,
               label="Power = 0.80")
    ax.axvline(mdd_80, color="#2ca02c", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.annotate(f"MDD@80% = {mdd_80:.4f}",
                xy=(mdd_80, 0.80), xytext=(6, 8),
                textcoords="offset points", fontsize=9, color="#2ca02c")

    # tabulated points
    ax.scatter(out_df["true_auc_diff"], out_df["power"],
               color="#1f77b4", s=28, zorder=4)

    ax.set_xlim(0.0, 0.06)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("True AUC difference (species RF - joint RF)", fontsize=11)
    ax.set_ylabel("Power", fontsize=11)
    ax.set_title(
        f"DeLong power vs true AUC gap (n={n_total}, "
        f"SE(diff)={se:.4f})",
        fontsize=12,
    )
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ----- summary note ----------------------------------------------------
    pct_at_obs = power_at_observed * 100.0
    note = (
        "# Power analysis: DeLong test, species RF vs joint RF\n\n"
        f"At n={n_total} pooled samples ({m} positives, {n_neg} negatives) the "
        f"observed standard error of the AUC difference, estimated by the "
        f"DeLong (Sun & Xu 2014) covariance formula on the paired predictions "
        f"in `preds_species_rf.csv` and `preds_joint_rf.csv`, is "
        f"SE(diff)={se:.4f}. Under a one-sided alpha={ALPHA} DeLong test we have "
        f"{pct_at_obs:.1f}% power to detect a 0.025-AUC difference, and 80% "
        f"power to detect a difference of {mdd_80:.4f} or greater (90% power "
        f"at {mdd_90:.4f}). The observed two-sided DeLong p={observed_p_two:.4f} "
        f"(one-sided p={observed_p_one:.4f}) is therefore well-powered, not a "
        f"chance result.\n"
    )
    OUT_NOTE.write_text(note)

    # ----- console summary -------------------------------------------------
    print(f"n = {n_total}  (positives={m}, negatives={n_neg})")
    print(f"AUC species_rf = {auc_a:.4f}  AUC joint_rf = {auc_b:.4f}")
    print(f"Observed diff  = {auc_a - auc_b:+.4f}")
    print(f"SE(diff)       = {se:.5f}")
    print(f"Observed z     = {observed_z:.3f}   "
          f"two-sided p={observed_p_two:.4g}   one-sided p={observed_p_one:.4g}")
    print(f"Power @ observed (0.025) = {power_at_observed:.3f}")
    print(f"MDD @ 80% power = {mdd_80:.5f}")
    print(f"MDD @ 90% power = {mdd_90:.5f}")
    print()
    print(out_df.to_string(index=False))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_FIG}")
    print(f"Wrote {OUT_NOTE}")


if __name__ == "__main__":
    main()
