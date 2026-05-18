"""Decision-curve analysis (Vickers & Elkin 2006) for the pooled LODO models.

Standard discrimination metrics (AUC, PPV, sensitivity at fixed specificity)
quantify how well a model separates cases from controls but do not weigh the
clinical consequence of acting on a prediction. Decision-curve analysis (DCA)
fills that gap by computing *net benefit* at each candidate threshold
probability p_t:

    net_benefit(p_t) = (TP / n) - (FP / n) * (p_t / (1 - p_t))

The (p_t / (1 - p_t)) term is the exchange rate at which the decision-maker
considers one false positive equivalent to (1 - p_t)/p_t true positives.
A model with higher net benefit than the two reference strategies
("treat all" and "treat none") over a clinically relevant threshold range
is the one a screening reviewer would actually prefer.

We sweep p_t from 0.01 to 0.50 in 0.01 steps for all three pooled LODO models
(species RF, joint RF, joint XGB) and compare against the two references.

Outputs:
    results/diagnostics/decision_curves.csv
    figures/diagnostics/decision_curves.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUT_CSV = REPO / "results" / "diagnostics" / "decision_curves.csv"
OUT_FIG = REPO / "figures" / "diagnostics" / "decision_curves.png"

MODELS = [
    ("species_rf", "Species RF", RESULTS / "preds_species_rf.csv"),
    ("joint_rf", "Joint RF", RESULTS / "preds_joint_rf.csv"),
    ("joint_xgb", "Joint XGB", RESULTS / "preds_joint_xgb.csv"),
]

# Threshold probability sweep: 0.01, 0.02, ..., 0.50
THRESHOLDS = np.round(np.arange(0.01, 0.50 + 1e-9, 0.01), 4)

# Colorblind-safe palette (Wong / Okabe-Ito)
COLOR_MODEL = {
    "species_rf": "#0072B2",  # blue
    "joint_rf": "#D55E00",    # vermillion
    "joint_xgb": "#009E73",   # bluish green
}
COLOR_ALL = "#999999"   # grey - treat all
COLOR_NONE = "#000000"  # black - treat none (always 0)


def net_benefit_model(
    y_true: np.ndarray, y_prob: np.ndarray, pt: float
) -> tuple[float, int, int]:
    """Vickers & Elkin net benefit for a probabilistic model at threshold pt.

    Returns (net_benefit, n_tp, n_fp).
    """
    n = y_true.size
    pred = (y_prob >= pt).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    # Guard against pt == 1 (would divide by zero); we only sweep up to 0.50.
    odds = pt / (1.0 - pt)
    nb = (tp / n) - (fp / n) * odds
    return float(nb), tp, fp


def net_benefit_treat_all(
    y_true: np.ndarray, pt: float
) -> tuple[float, int, int]:
    """Net benefit for the 'treat everyone' strategy."""
    n = y_true.size
    tp = int((y_true == 1).sum())
    fp = int((y_true == 0).sum())
    odds = pt / (1.0 - pt)
    nb = (tp / n) - (fp / n) * odds
    return float(nb), tp, fp


def compute_curves() -> pd.DataFrame:
    rows: list[dict] = []
    for key, _label, path in MODELS:
        df = pd.read_csv(path)
        y_true = df["y_true"].to_numpy().astype(int)
        y_prob = df["y_prob"].to_numpy().astype(float)
        for pt in THRESHOLDS:
            nb, tp, fp = net_benefit_model(y_true, y_prob, float(pt))
            rows.append(
                {
                    "model": key,
                    "threshold_pt": round(float(pt), 4),
                    "net_benefit": round(nb, 6),
                    "n_tp": tp,
                    "n_fp": fp,
                    "strategy": "model",
                }
            )

    # Reference strategies are functions of the labels only; use the first
    # model's labels (all three CSVs share the same n=1339 cohort).
    ref_df = pd.read_csv(MODELS[0][2])
    y_true_ref = ref_df["y_true"].to_numpy().astype(int)

    for pt in THRESHOLDS:
        nb_all, tp_all, fp_all = net_benefit_treat_all(y_true_ref, float(pt))
        rows.append(
            {
                "model": "treat_all",
                "threshold_pt": round(float(pt), 4),
                "net_benefit": round(nb_all, 6),
                "n_tp": tp_all,
                "n_fp": fp_all,
                "strategy": "all",
            }
        )
        rows.append(
            {
                "model": "treat_none",
                "threshold_pt": round(float(pt), 4),
                "net_benefit": 0.0,
                "n_tp": 0,
                "n_fp": 0,
                "strategy": "none",
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "model",
            "threshold_pt",
            "net_benefit",
            "n_tp",
            "n_fp",
            "strategy",
        ],
    )


def first_threshold_above_treat_none(
    out_df: pd.DataFrame, model_key: str
) -> float | None:
    """Smallest pt at which the model's net benefit strictly exceeds 0
    (i.e., beats treat-none)."""
    sub = out_df[(out_df.model == model_key) & (out_df.strategy == "model")]
    sub = sub.sort_values("threshold_pt")
    positives = sub[sub.net_benefit > 0]
    if positives.empty:
        return None
    return float(positives.iloc[0]["threshold_pt"])


def plot_curves(out_df: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(9.0, 5.5))

    # Reference: treat-all
    sub_all = (
        out_df[out_df.strategy == "all"].sort_values("threshold_pt")
    )
    ax.plot(
        sub_all["threshold_pt"],
        sub_all["net_benefit"],
        color=COLOR_ALL,
        linestyle="--",
        linewidth=1.6,
        label="Treat all",
    )

    # Reference: treat-none (always zero)
    ax.axhline(
        0.0,
        color=COLOR_NONE,
        linestyle=":",
        linewidth=1.2,
        label="Treat none",
    )

    # Model curves
    for key, label, _path in MODELS:
        sub = (
            out_df[(out_df.model == key) & (out_df.strategy == "model")]
            .sort_values("threshold_pt")
        )
        ax.plot(
            sub["threshold_pt"],
            sub["net_benefit"],
            color=COLOR_MODEL[key],
            linewidth=2.0,
            label=label,
        )

    # Cosmetics
    ax.set_xlabel("Threshold probability p$_t$", fontsize=11)
    ax.set_ylabel("Net benefit", fontsize=11)
    ax.set_xlim(0.0, 0.50)
    # Generous lower bound so treat-all dive into negatives is visible.
    ymin = float(min(out_df.net_benefit.min(), -0.05))
    ymax = float(max(out_df.net_benefit.max(), 0.05)) * 1.05
    ax.set_ylim(ymin * 1.05, ymax)
    ax.set_title(
        "Decision-curve analysis (Vickers & Elkin)\n"
        "Pooled LODO predictions, n = 1339",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    out_df = compute_curves()
    out_df.to_csv(OUT_CSV, index=False)
    plot_curves(out_df)

    # Headline diagnostic: species RF's first threshold beating treat-none.
    pt_species = first_threshold_above_treat_none(out_df, "species_rf")
    pt_joint_rf = first_threshold_above_treat_none(out_df, "joint_rf")
    pt_joint_xgb = first_threshold_above_treat_none(out_df, "joint_xgb")

    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_FIG}")
    print(
        "First pt with net benefit > treat-none: "
        f"species_rf={pt_species}, joint_rf={pt_joint_rf}, joint_xgb={pt_joint_xgb}"
    )


if __name__ == "__main__":
    main()
