"""Cumulative gains and lift curves for the pooled LODO models.

Lift / cumulative-gains analysis answers a different clinical question than
ROC or decision-curve analysis: if a screening program can only afford to
follow up the top K% of patients ranked by predicted risk, what fraction of
true cases does that capture, and how much better is that than random
selection?

For each pooled LODO model we rank samples by predicted probability in
descending order, then walk the deciles (10%, 20%, ..., 100%):

    cumulative_gains(K%) = cumulative_TP / total_positives
    lift(K%) = (cumulative_TP / cumulative_n) / overall_prevalence

A lift of, say, 1.8 at the top 20% means a clinic referring only those
top-quintile patients would catch CRC ~1.8x as often per patient screened
as referring uniformly at random.

Outputs:
    results/diagnostics/lift_curves.csv
    figures/diagnostics/lift_curves.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUT_CSV = REPO / "results" / "diagnostics" / "lift_curves.csv"
OUT_FIG = REPO / "figures" / "diagnostics" / "lift_curves.png"

MODELS = [
    ("species_rf", "Species RF", RESULTS / "preds_species_rf.csv"),
    ("joint_rf", "Joint RF", RESULTS / "preds_joint_rf.csv"),
    ("joint_xgb", "Joint XGB", RESULTS / "preds_joint_xgb.csv"),
]

# Decile cuts: 10%, 20%, ..., 100%
DECILES = np.round(np.arange(0.10, 1.00 + 1e-9, 0.10), 2)

COLOR_MODEL = {
    "species_rf": "#0072B2",  # blue
    "joint_rf": "#D55E00",    # vermillion
    "joint_xgb": "#009E73",   # bluish green
}
COLOR_RANDOM = "#999999"


def lift_for_model(
    y_true: np.ndarray, y_prob: np.ndarray, deciles: np.ndarray
) -> list[dict]:
    """Sort by predicted probability descending; compute gains and lift at
    each decile cut. Ties in y_prob are broken by the stable sort, which is
    sufficient for monotone cumulative summaries.
    """
    n = y_true.size
    n_pos = int(y_true.sum())
    prevalence = n_pos / n if n else float("nan")

    # Descending sort by predicted probability.
    order = np.argsort(-y_prob, kind="stable")
    y_sorted = y_true[order]
    cum_tp = np.cumsum(y_sorted)

    rows: list[dict] = []
    for frac in deciles:
        cum_n = max(int(round(float(frac) * n)), 1)
        # Clamp to n in case of rounding.
        cum_n = min(cum_n, n)
        tp = int(cum_tp[cum_n - 1])
        gains = tp / n_pos if n_pos else float("nan")
        capture_rate = tp / cum_n
        lift = capture_rate / prevalence if prevalence > 0 else float("nan")
        rows.append(
            {
                "frac_screened": round(float(frac), 4),
                "cumulative_n": cum_n,
                "cumulative_tp": tp,
                "total_positives": n_pos,
                "cumulative_gains": round(gains, 6),
                "lift": round(lift, 6),
                "prevalence": round(prevalence, 6),
            }
        )
    return rows


def compute_lift_table() -> pd.DataFrame:
    rows: list[dict] = []
    for key, _label, path in MODELS:
        df = pd.read_csv(path)
        y_true = df["y_true"].to_numpy().astype(int)
        y_prob = df["y_prob"].to_numpy().astype(float)
        model_rows = lift_for_model(y_true, y_prob, DECILES)
        for r in model_rows:
            r2 = {"model": key, **r}
            rows.append(r2)
    return pd.DataFrame(
        rows,
        columns=[
            "model",
            "frac_screened",
            "cumulative_n",
            "cumulative_tp",
            "total_positives",
            "cumulative_gains",
            "lift",
            "prevalence",
        ],
    )


def plot_curves(out_df: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))

    # --- Left panel: lift vs % population screened --------------------------
    for key, label, _path in MODELS:
        sub = (
            out_df[out_df.model == key].sort_values("frac_screened")
        )
        axes[0].plot(
            sub["frac_screened"] * 100.0,
            sub["lift"],
            marker="o",
            markersize=5,
            color=COLOR_MODEL[key],
            linewidth=2.0,
            label=label,
            markeredgecolor="white",
            markeredgewidth=0.7,
        )
    axes[0].axhline(
        1.0,
        color=COLOR_RANDOM,
        linestyle="--",
        linewidth=1.2,
        label="Random (lift = 1)",
    )
    axes[0].set_xlabel("% population screened (ranked by predicted risk)", fontsize=11)
    axes[0].set_ylabel("Lift", fontsize=11)
    axes[0].set_xlim(0, 100)
    ymin_l = min(0.9, float(out_df["lift"].min()) * 0.95)
    ymax_l = float(out_df["lift"].max()) * 1.05
    axes[0].set_ylim(ymin_l, ymax_l)
    axes[0].set_title("Lift curve", fontsize=11)
    axes[0].legend(loc="upper right", fontsize=9, frameon=True)
    axes[0].tick_params(labelsize=10)

    # --- Right panel: cumulative gains --------------------------------------
    # Reference line: random selection (cum_gain == frac_screened).
    axes[1].plot(
        [0, 100],
        [0, 1],
        color=COLOR_RANDOM,
        linestyle="--",
        linewidth=1.2,
        label="Random (gain = % screened)",
    )
    for key, label, _path in MODELS:
        sub = (
            out_df[out_df.model == key].sort_values("frac_screened")
        )
        axes[1].plot(
            sub["frac_screened"] * 100.0,
            sub["cumulative_gains"],
            marker="o",
            markersize=5,
            color=COLOR_MODEL[key],
            linewidth=2.0,
            label=label,
            markeredgecolor="white",
            markeredgewidth=0.7,
        )
    axes[1].set_xlabel("% population screened (ranked by predicted risk)", fontsize=11)
    axes[1].set_ylabel("Cumulative gains (fraction of cases captured)", fontsize=11)
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Cumulative gains", fontsize=11)
    axes[1].legend(loc="lower right", fontsize=9, frameon=True)
    axes[1].tick_params(labelsize=10)

    fig.suptitle(
        "Lift and cumulative gains (pooled LODO predictions, n = 1339)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    out_df = compute_lift_table()
    out_df.to_csv(OUT_CSV, index=False)
    plot_curves(out_df)

    # Headline: top-20% lift per model.
    top20 = (
        out_df[out_df.frac_screened == 0.20]
        .set_index("model")[["lift", "cumulative_gains"]]
    )
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_FIG}")
    print("Lift / gains at top-20%:")
    print(top20.to_string())


if __name__ == "__main__":
    main()
