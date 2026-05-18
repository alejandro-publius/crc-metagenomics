"""Species RF AUC stratified by age band, sex, and BMI category.

For each subgroup, compute AUC on the pooled LODO predictions and a
1000-iteration bootstrap 95% CI. Drops samples with missing values for
the relevant stratifier. Produces a forest plot.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
PREDS = REPO / "results" / "preds_species_rf.csv"
META = REPO / "data" / "processed" / "metadata_clean.csv"
OUT_CSV = REPO / "results" / "diagnostics" / "subgroup_auc.csv"
OUT_FIG = REPO / "figures" / "diagnostics" / "subgroup_auc.png"

N_BOOT = 1000
SEED = 20240518


def age_band(a: float) -> str:
    if pd.isna(a):
        return "NA"
    if a < 50:
        return "<50"
    if a <= 65:
        return "50-65"
    return ">65"


def bmi_band(b: float) -> str:
    if pd.isna(b):
        return "NA"
    if b < 25:
        return "<25"
    if b <= 30:
        return "25-30"
    return ">30"


def bootstrap_auc_ci(
    y_true: np.ndarray, y_prob: np.ndarray, n_boot: int, rng: np.random.Generator
):
    n = len(y_true)
    if n == 0 or len(np.unique(y_true)) < 2:
        return float("nan"), float("nan"), float("nan")
    point = float(roc_auc_score(y_true, y_prob))
    boots = np.empty(n_boot)
    boots[:] = np.nan
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        yp = y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        boots[i] = roc_auc_score(yt, yp)
    valid = boots[~np.isnan(boots)]
    if valid.size == 0:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(valid, [2.5, 97.5])
    return point, float(lo), float(hi)


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    preds = pd.read_csv(PREDS)
    meta = pd.read_csv(META)[["sample_id", "age", "gender", "BMI"]]
    df = preds.merge(meta, on="sample_id", how="left")

    rng = np.random.default_rng(SEED)

    # Build stratification frames per variable, dropping NaN for that variable only
    age_df = df.dropna(subset=["age"]).copy()
    age_df["level"] = age_df["age"].apply(age_band)
    age_levels = ["<50", "50-65", ">65"]

    sex_df = df.dropna(subset=["gender"]).copy()
    sex_df["level"] = sex_df["gender"].astype(str).str.lower()
    sex_levels = ["female", "male"]

    bmi_df = df.dropna(subset=["BMI"]).copy()
    bmi_df["level"] = bmi_df["BMI"].apply(bmi_band)
    bmi_levels = ["<25", "25-30", ">30"]

    plan = [
        ("age_band", age_df, age_levels),
        ("sex", sex_df, sex_levels),
        ("bmi_category", bmi_df, bmi_levels),
    ]

    rows = []
    for var, sub_df, levels in plan:
        for level in levels:
            mask = sub_df["level"] == level
            ss = sub_df[mask]
            n = int(len(ss))
            n_pos = int((ss["y_true"] == 1).sum())
            n_neg = int((ss["y_true"] == 0).sum())
            if n == 0 or n_pos == 0 or n_neg == 0:
                rows.append(
                    {
                        "variable": var,
                        "level": level,
                        "n": n,
                        "n_pos": n_pos,
                        "n_neg": n_neg,
                        "auc": float("nan"),
                        "ci_lo": float("nan"),
                        "ci_hi": float("nan"),
                    }
                )
                continue
            yt = ss["y_true"].to_numpy().astype(int)
            yp = ss["y_prob"].to_numpy().astype(float)
            auc, lo, hi = bootstrap_auc_ci(yt, yp, N_BOOT, rng)
            rows.append(
                {
                    "variable": var,
                    "level": level,
                    "n": n,
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "auc": auc,
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )

    # Overall reference
    yt_all = df["y_true"].to_numpy().astype(int)
    yp_all = df["y_prob"].to_numpy().astype(float)
    overall_auc, overall_lo, overall_hi = bootstrap_auc_ci(yt_all, yp_all, N_BOOT, rng)
    rows.insert(
        0,
        {
            "variable": "overall",
            "level": "all",
            "n": int(len(df)),
            "n_pos": int((df["y_true"] == 1).sum()),
            "n_neg": int((df["y_true"] == 0).sum()),
            "auc": overall_auc,
            "ci_lo": overall_lo,
            "ci_hi": overall_hi,
        },
    )

    out_df = pd.DataFrame(rows)
    for c in ("auc", "ci_lo", "ci_hi"):
        out_df[c] = out_df[c].astype(float).round(4)
    out_df = out_df[["variable", "level", "n", "n_pos", "n_neg", "auc", "ci_lo", "ci_hi"]]
    out_df.to_csv(OUT_CSV, index=False)

    # Forest plot
    plt.style.use("seaborn-v0_8-whitegrid")
    cmap = plt.get_cmap("tab10")
    var_colors = {
        "overall": "#333333",
        "age_band": cmap(0),
        "sex": cmap(2),
        "bmi_category": cmap(3),
    }
    var_labels = {
        "overall": "Overall",
        "age_band": "Age band",
        "sex": "Sex",
        "bmi_category": "BMI category",
    }

    plot_df = out_df.iloc[::-1].reset_index(drop=True)  # top-to-bottom = file order
    y_pos = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(9.5, 0.55 * len(plot_df) + 2.0))
    for i, row in plot_df.iterrows():
        color = var_colors.get(row["variable"], "#555555")
        if not np.isnan(row["auc"]):
            ax.plot(
                [row["ci_lo"], row["ci_hi"]],
                [y_pos[i], y_pos[i]],
                color=color,
                linewidth=2.0,
            )
            ax.plot(row["auc"], y_pos[i], "o", color=color, markersize=7)

    ax.axvline(
        overall_auc,
        color="#333333",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label=f"Overall AUC = {overall_auc:.3f}",
    )
    ax.axvline(0.5, color="grey", linestyle=":", linewidth=1, alpha=0.6, label="Chance = 0.50")

    labels = []
    for _, row in plot_df.iterrows():
        vlab = var_labels.get(row["variable"], row["variable"])
        labels.append(f"{vlab}: {row['level']}  (n={row['n']})")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)

    # Right-side annotations with AUC [CI]
    xmax = max(0.05 + float(plot_df["ci_hi"].max(skipna=True) or 1.0), 1.0)
    for i, row in plot_df.iterrows():
        if np.isnan(row["auc"]):
            txt = "n/a"
        else:
            txt = f"{row['auc']:.3f} [{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]"
        ax.text(xmax + 0.005, y_pos[i], txt, va="center", fontsize=9)

    ax.set_xlim(0.3, xmax + 0.18)
    ax.set_xlabel("AUC (95% bootstrap CI)", fontsize=11)
    ax.set_title(
        "Species RF — subgroup AUC (pooled LODO, 1000-iter bootstrap)",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.tick_params(axis="x", labelsize=10)

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_FIG}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
