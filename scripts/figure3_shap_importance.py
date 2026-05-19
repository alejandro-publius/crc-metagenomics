"""Generate Figure 3 (top-15 SHAP feature importance) for the manuscript.

Side-by-side bar charts of the top 15 features (by mean |SHAP|) for the
joint Random Forest and the joint XGBoost CRC-vs-control models.

Reads:
    results/shap_crc_features.csv  (joint RF mean |SHAP| per feature)
    results/shap_crc_xgb.csv       (joint XGB mean |SHAP| per feature)

Writes:
    manuscript/figures/Figure3_SHAP_Importance.png  (300 DPI raster)
    manuscript/figures/Figure3_SHAP_Importance.pdf  (vector)

Usage:
    python3 scripts/figure3_shap_importance.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT_DIR = REPO / "manuscript" / "figures"

TOP_N = 15

PANELS = [
    # (csv_path, panel title, bar color)
    (RESULTS / "shap_crc_features.csv",
     "Joint Random Forest", "#1f77b4"),
    (RESULTS / "shap_crc_xgb.csv",
     "Joint XGBoost",       "#e07a4d"),
]


def short_label(feature: str) -> str:
    """Compress a long MetaPhlAn-style lineage into a readable label.

    Returns the species-level token (s__...) where present, falling back
    to the deepest non-empty segment. Underscores are converted to
    spaces so the bar labels render cleanly.
    """
    if not isinstance(feature, str):
        return str(feature)
    parts = feature.split("|")
    species = next((p for p in reversed(parts) if p.startswith("s__")), None)
    if species:
        return species.replace("s__", "").replace("_", " ")
    # Fallback: deepest token, strip the rank prefix if present.
    deepest = parts[-1] if parts else feature
    for prefix in ("g__", "f__", "o__", "c__", "p__", "k__"):
        if deepest.startswith(prefix):
            deepest = deepest[len(prefix):]
            break
    return deepest.replace("_", " ")


def load_top(path: Path, top_n: int = TOP_N) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "mean_abs_shap" not in df.columns or "feature" not in df.columns:
        raise ValueError(
            f"{path} must contain 'feature' and 'mean_abs_shap' columns; "
            f"got {list(df.columns)}"
        )
    df = df.sort_values("mean_abs_shap", ascending=False).head(top_n).copy()
    df["short"] = df["feature"].map(short_label)
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 7.5))
    fig.patch.set_facecolor("#f7f5f1")

    for ax, (csv_path, title, color) in zip(axes, PANELS):
        df = load_top(csv_path, TOP_N)
        # Reverse so the largest bar appears at the top of the chart.
        df = df.iloc[::-1].reset_index(drop=True)
        ax.set_facecolor("#f7f5f1")
        ax.barh(df["short"], df["mean_abs_shap"], color=color,
                edgecolor="black", linewidth=0.4, alpha=0.85)
        ax.set_xlabel("Mean |SHAP|", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.tick_params(axis="y", labelsize=9)
        ax.tick_params(axis="x", labelsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="x", color="gray", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)

    fig.suptitle(
        f"Figure 3. Top {TOP_N} features by mean |SHAP| for the CRC vs. "
        "control task",
        fontsize=13, fontweight="bold", y=1.00,
    )
    fig.tight_layout()

    png_path = OUT_DIR / "Figure3_SHAP_Importance.png"
    pdf_path = OUT_DIR / "Figure3_SHAP_Importance.pdf"
    fig.savefig(png_path, dpi=300, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    fig.savefig(pdf_path, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
