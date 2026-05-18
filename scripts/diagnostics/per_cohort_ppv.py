"""Per-cohort base-rate-adjusted PPV / NPV for the species RF screening model.

Whereas ``base_rate_ppv.py`` reweights the pooled Youden operating point
across a prevalence sweep, this script does the reweighting *per cohort*.
For each of the 10 cohorts we:

  1. Compute the Youden-J optimal threshold on that cohort's held-out
     predictions (so each cohort is calibrated to its own ROC).
  2. Measure the cohort's observed sensitivity / specificity at that
     local threshold.
  3. Apply Bayes' rule at 4 representative prevalences (0.005, 0.02,
     0.05, 0.10) covering annual-screening through high-risk-clinic
     population rates.

The deliverable is a per-cohort PPV table and a paired-bar figure that
makes it obvious which sites the species RF would be clinically useful
in as a triage layer (PPV >> background prevalence) versus where the
local operating characteristics are too weak to add screening value.

Bayes' rule:

    PPV = (sens * prev) / (sens * prev + (1 - spec) * (1 - prev))
    NPV = (spec * (1 - prev)) / (spec * (1 - prev) + (1 - sens) * prev)

NNT (number needed to test to find one true positive) = 1 / PPV.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

REPO = Path(__file__).resolve().parents[2]
PREDS = REPO / "results" / "preds_species_rf.csv"
POOLED_PPV = REPO / "results" / "diagnostics" / "base_rate_ppv.csv"
OUT_CSV = REPO / "results" / "diagnostics" / "per_cohort_ppv.csv"
OUT_FIG = REPO / "figures" / "diagnostics" / "per_cohort_ppv.png"
OUT_MD = REPO / "results" / "diagnostics" / "PER_COHORT_CLINICAL_TRANSLATION.md"

PREVALENCES = (0.005, 0.02, 0.05, 0.10)


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


def pooled_5pct_ppv() -> float:
    """Look up the pooled PPV at 5% prevalence from the existing CSV."""
    df = pd.read_csv(POOLED_PPV)
    row = df.loc[(df["prevalence"] - 0.05).abs().idxmin()]
    return float(row["ppv"])


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PREDS)

    rows = []
    for cohort, sub in df.groupby("cohort"):
        yt = sub["y_true"].to_numpy().astype(int)
        yp = sub["y_prob"].to_numpy().astype(float)
        # Need at least one positive and one negative to compute Youden.
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        thr = youden_threshold(yt, yp)
        sens, spec = sens_spec_at(yt, yp, thr)

        row: dict[str, float | str | int] = {
            "cohort": cohort,
            "n": int(len(yt)),
            "threshold": round(thr, 4),
            "sens_observed": round(sens, 4),
            "spec_observed": round(spec, 4),
        }
        for p in PREVALENCES:
            ppv, npv = ppv_npv(sens, spec, p)
            tag = f"{p * 100:g}pct".replace(".", "_")
            row[f"ppv_at_{tag}"] = round(ppv, 4)
            if p == 0.05:
                row["npv_at_5pct"] = round(npv, 4)
                row["nnt_at_5pct"] = round(1.0 / ppv, 4) if ppv > 0 else float("nan")
        rows.append(row)

    out_df = pd.DataFrame(rows)
    # Enforce required column order / names matching the spec.
    rename = {
        "ppv_at_0_5pct": "ppv_at_0.5pct",
        "ppv_at_2pct": "ppv_at_2pct",
        "ppv_at_5pct": "ppv_at_5pct",
        "ppv_at_10pct": "ppv_at_10pct",
    }
    out_df = out_df.rename(columns=rename)
    cols = [
        "cohort",
        "n",
        "sens_observed",
        "spec_observed",
        "ppv_at_0.5pct",
        "ppv_at_2pct",
        "ppv_at_5pct",
        "ppv_at_10pct",
        "npv_at_5pct",
        "nnt_at_5pct",
    ]
    out_df = out_df[cols].sort_values("cohort").reset_index(drop=True)
    out_df.to_csv(OUT_CSV, index=False)

    # ------------------------------------------------------------------
    # Figure: paired bar chart, 4 prevalence bars per cohort.
    # ------------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid")
    # Okabe-Ito colorblind-safe palette (4 distinct hues).
    bar_colors = ["#56B4E9", "#0072B2", "#E69F00", "#D55E00"]
    ref_color = "#666666"

    cohorts = out_df["cohort"].tolist()
    ns = out_df["n"].to_numpy()
    ppv_cols = ["ppv_at_0.5pct", "ppv_at_2pct", "ppv_at_5pct", "ppv_at_10pct"]
    ppv_labels = ["0.5%", "2%", "5%", "10%"]

    x = np.arange(len(cohorts))
    n_bars = len(ppv_cols)
    width = 0.78 / n_bars

    fig, ax = plt.subplots(figsize=(max(11.0, 1.05 * len(cohorts) + 3), 5.8))

    bar_groups = []
    for i, (col, lbl, c) in enumerate(zip(ppv_cols, ppv_labels, bar_colors)):
        offset = (i - (n_bars - 1) / 2) * width
        vals = out_df[col].to_numpy()
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=f"Prevalence {lbl}",
            color=c,
            edgecolor="white",
            linewidth=0.5,
        )
        bar_groups.append((bars, vals))

    # Value labels above each bar.
    for bars, vals in bar_groups:
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.006,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    pooled_5 = pooled_5pct_ppv()
    ax.axhline(
        pooled_5,
        color=ref_color,
        linestyle="--",
        linewidth=1.2,
        label=f"Pooled PPV @ 5% = {pooled_5:.3f}",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{c}\n(n={n})" for c, n in zip(cohorts, ns)],
        rotation=30,
        ha="right",
        fontsize=9.5,
    )
    ax.set_ylabel("Positive predictive value", fontsize=11)
    y_max = max(0.85, float(out_df[ppv_cols].to_numpy().max()) + 0.08)
    ax.set_ylim(0, y_max)
    ax.set_title(
        "Species RF — per-cohort base-rate-adjusted PPV at local Youden-J threshold",
        fontsize=11.5,
    )
    ax.legend(loc="upper right", fontsize=9, frameon=True, ncol=2)
    ax.tick_params(axis="y", labelsize=10)

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # One-paragraph clinical-translation summary.
    # ------------------------------------------------------------------
    ranked = out_df.sort_values("ppv_at_5pct", ascending=False).reset_index(drop=True)
    flagged = ranked[ranked["ppv_at_5pct"] > 0.20]
    weakest = ranked.tail(3).iloc[::-1]

    def _fmt(row: pd.Series) -> str:
        return (
            f"{row['cohort']} (n={int(row['n'])}, PPV={row['ppv_at_5pct']:.3f}, "
            f"NNT={row['nnt_at_5pct']:.1f})"
        )

    if len(flagged):
        flagged_txt = "; ".join(_fmt(r) for _, r in flagged.iterrows())
        flagged_sentence = (
            f"At an assumed 5% target-population prevalence, the species RF "
            f"clears the PPV > 20% clinically-meaningful triage bar in "
            f"{len(flagged)} of {len(ranked)} cohorts: {flagged_txt}. These "
            f"are the sites where the model could plausibly be deployed as a "
            f"pre-colonoscopy enrichment layer."
        )
    else:
        flagged_sentence = (
            f"At an assumed 5% target-population prevalence, no cohort clears "
            f"the PPV > 20% triage bar, indicating that the species RF alone "
            f"is insufficient as a standalone enrichment layer at population "
            f"base rates regardless of site-local calibration."
        )

    weakest_txt = "; ".join(_fmt(r) for _, r in weakest.iterrows())
    pooled_sentence = (
        f"By contrast the weakest cohorts ({weakest_txt}) sit at or below the "
        f"pooled 5%-prevalence reference of PPV = {pooled_5:.3f} (NNT "
        f"≈ {1.0 / pooled_5:.1f}), reflecting the same sensitivity/"
        f"specificity ceiling seen in the pooled LODO analysis."
    )

    md = (
        "# Per-cohort clinical translation (species RF)\n\n"
        f"Each cohort's Youden-J optimal threshold was estimated locally on "
        f"its held-out predictions, observed sensitivity/specificity were "
        f"measured at that threshold, and PPV / NPV / NNT were reweighted "
        f"via Bayes' rule across four representative population prevalences "
        f"(0.5%, 2%, 5%, 10%). Sensitivity and specificity are prevalence-"
        f"invariant, so any shift in PPV across cohorts at a fixed prevalence "
        f"reflects genuine site-level differences in the discrimination ceiling "
        f"of the species RF.\n\n"
        f"{flagged_sentence} {pooled_sentence}\n"
    )

    OUT_MD.write_text(md)

    # ------------------------------------------------------------------
    # Console summary.
    # ------------------------------------------------------------------
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_FIG}")
    print(f"Wrote {OUT_MD}")
    print(f"Pooled PPV @ 5% reference = {pooled_5:.4f}")
    print(out_df.to_string(index=False))
    if len(flagged):
        print(
            "\nCohorts with PPV @ 5% > 20%: "
            + ", ".join(
                f"{r['cohort']} ({r['ppv_at_5pct']:.4f})"
                for _, r in flagged.iterrows()
            )
        )
    else:
        print("\nNo cohorts clear PPV @ 5% > 20%.")


if __name__ == "__main__":
    main()
