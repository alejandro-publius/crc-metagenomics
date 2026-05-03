"""
Generate Figure 5: multi-classifier SHAP comparison for the two-phase model.

Three-panel horizontal bar chart showing top 10 features (by mean absolute SHAP
value) for each of the three classifiers in the adenoma-carcinoma sequence:

    Panel A: Healthy vs Adenoma (commensal depletion phase)
    Panel B: CRC vs Healthy        (oral pathobiont colonization)
    Panel C: Adenoma vs CRC        (oral pathobiont colonization)

USAGE
-----
1. Place this script in the project's `scripts/` directory.
2. Confirm the SHAP result file paths below match your repo. Edit if needed.
3. Run from the repo root:
       python3 scripts/figure5_shap_three_panel.py
4. Output saved to `figures/figure5_three_panel_shap.png` (300 dpi).

INPUT FILE FORMAT
-----------------
Each CSV has columns:
    feature        - either a MetaPhlAn taxonomy string (k__...|s__Genus_species)
                     or a pathway string (e.g. "PANTOSYN-PWY: ...")
    mean_abs_shap  - float
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# -----------------------------------------------------------------------------
# CONFIG: edit these paths to match your repo
# -----------------------------------------------------------------------------
RESULTS_DIR = Path("results")
# Manuscript figure (Figure 4): three-panel SHAP across the three classifiers
# in the adenoma-carcinoma sequence.
MANUSCRIPT_PNG = Path("manuscript/figures/Figure4_Three_Panel_SHAP.png")
MANUSCRIPT_PDF = Path("manuscript/figures/Figure4_Three_Panel_SHAP.pdf")
# Legacy script-output location (kept for backward compatibility with
# REPRODUCING.md). Will be written if the legacy figures/ dir exists.
LEGACY_OUTPUT = Path("figures/figure5_three_panel_shap.png")

SHAP_FILES = {
    "Healthy vs Adenoma": RESULTS_DIR / "shap_healthy_vs_adenoma.csv",
    "CRC vs Healthy":     RESULTS_DIR / "shap_crc_features.csv",
    "Adenoma vs CRC":     RESULTS_DIR / "shap_adenoma_vs_crc.csv",
}

PANEL_SUBTITLES = {
    "Healthy vs Adenoma": "commensal depletion phase",
    "CRC vs Healthy":     "oral pathobiont colonization",
    "Adenoma vs CRC":     "oral pathobiont colonization",
}

TOP_N = 10
FEATURE_COL = "feature"
SHAP_COL = "mean_abs_shap"

# -----------------------------------------------------------------------------
# Feature label parsing
# -----------------------------------------------------------------------------
def clean_feature_name(raw: str) -> tuple[str, bool]:
    """Convert raw feature string to a clean display label.
    Returns (label, is_species_flag) where is_species_flag is True for species
    (which should be italicized in the figure)."""
    s = str(raw).strip()

    # MetaPhlAn taxonomy: take the last segment after s__
    m = re.search(r"s__([A-Za-z0-9_]+)", s)
    if m:
        species_name = m.group(1).replace("_", " ")
        return species_name, True

    # Pathway with prefix like "PANTOSYN-PWY: pantothenate and coenzyme A..."
    m = re.match(r"^([A-Z0-9\-]+-PWY|PWY-?\d+|PWY\d+|[A-Z]+-PWY):\s*(.+)$", s)
    if m:
        # Use a compact label: "PWY-621 (sucrose degradation III)"
        prefix, desc = m.group(1), m.group(2)
        # Trim parenthetical at end if very long
        desc_short = desc.split(":")[0].strip()
        if len(desc_short) > 38:
            desc_short = desc_short[:35] + "..."
        return f"{prefix} ({desc_short})", False

    # Fallback: return as-is, truncate if very long
    if len(s) > 50:
        return s[:47] + "...", False
    return s, False


def load_shap(path: Path, top_n: int = TOP_N) -> pd.DataFrame:
    """Load CSV and return top-N rows sorted descending by SHAP."""
    if not path.exists():
        raise FileNotFoundError(
            f"SHAP file not found: {path}\n"
            f"Edit SHAP_FILES at the top of this script."
        )
    df = pd.read_csv(path)
    if FEATURE_COL not in df.columns or SHAP_COL not in df.columns:
        raise ValueError(
            f"{path} must have columns '{FEATURE_COL}' and '{SHAP_COL}'. "
            f"Found: {list(df.columns)}"
        )
    df = df.sort_values(SHAP_COL, ascending=False).head(top_n).reset_index(drop=True)
    # Parse labels
    parsed = df[FEATURE_COL].apply(clean_feature_name)
    df["label"] = [p[0] for p in parsed]
    df["is_species"] = [p[1] for p in parsed]
    # Reverse so highest-ranked feature plots at the top of horizontal bar chart
    return df.iloc[::-1].reset_index(drop=True)


# -----------------------------------------------------------------------------
# Build figure
# -----------------------------------------------------------------------------
def build_figure():
    cmap = LinearSegmentedColormap.from_list(
        "yellow_red", ["#fde047", "#f59e0b", "#dc2626", "#7f1d1d"]
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 7),
                             gridspec_kw={"wspace": 0.65})
    panel_letters = ["A", "B", "C"]

    for ax, (title, path), letter in zip(axes, SHAP_FILES.items(), panel_letters):
        df = load_shap(path)
        n = len(df)
        # Color: top-ranked feature (last row, top of bar chart) = lightest yellow
        # bottom-ranked (first row, bottom of bar chart) = darkest red
        colors = [cmap(1 - i / max(n - 1, 1)) for i in range(n)]

        y_positions = list(range(n))
        ax.barh(y_positions, df[SHAP_COL].values, color=colors,
                edgecolor="black", linewidth=0.5)
        ax.set_yticks(y_positions)
        # Italicize species labels using mathtext
        labels = []
        for label_text, is_sp in zip(df["label"], df["is_species"]):
            if is_sp:
                # Replace spaces with mathtext-friendly spacing
                labels.append(r"$\it{" + label_text.replace(" ", r"\ ") + r"}$")
            else:
                labels.append(label_text)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Mean |SHAP value|", fontsize=11)
        ax.set_title(f"{title}\n({PANEL_SUBTITLES[title]})",
                     fontsize=12, fontweight="bold", pad=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(-0.02, 1.10, letter, transform=ax.transAxes,
                fontsize=18, fontweight="bold", va="top", ha="right")

    plt.tight_layout()
    MANUSCRIPT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(MANUSCRIPT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(MANUSCRIPT_PDF, bbox_inches="tight")
    print(f"Saved: {MANUSCRIPT_PNG}")
    print(f"Saved: {MANUSCRIPT_PDF}")
    if LEGACY_OUTPUT.parent.exists():
        plt.savefig(LEGACY_OUTPUT, dpi=300, bbox_inches="tight")
        print(f"Saved: {LEGACY_OUTPUT}")


if __name__ == "__main__":
    build_figure()
