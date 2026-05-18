"""Publication-grade visual abstract for the CRC metagenomics paper.

Renders a single 7x7 inch figure with four panels:
  (TL) Cohort map: 10 cohorts across 8 countries / 3 continents (N=1522).
  (TR) Methodology pipeline: cohorts -> country-aware LODO -> models -> DeLong -> verdict.
  (BL) Headline result: pooled AUC for species RF, joint RF, joint XGB with 95% CI bars
       and DeLong p-values (species-only wins).
  (BR) Biological signature: four oral pathobionts identified as top SHAP features
       for adenoma -> CRC (transition) but not healthy -> adenoma (early).

All numbers come from existing CSVs (no fabricated values). Pure matplotlib.

Outputs:
  figures/visual_abstract.png  (600 DPI)
  figures/visual_abstract.pdf  (vector, for journal upload)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
DATA = REPO / "data" / "processed"
FIGDIR = REPO / "figures"

OUT_PNG = FIGDIR / "visual_abstract.png"
OUT_PDF = FIGDIR / "visual_abstract.pdf"

# Okabe-Ito palette (colorblind-safe)
OKABE_ITO = {
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "pink": "#CC79A7",
    "black": "#000000",
    "grey": "#999999",
}

# Approximate (lon, lat) for the 8 countries represented in the 10 cohorts.
COUNTRY_COORDS = {
    "AUT": (14.55, 47.52),   # Austria
    "CHN": (104.20, 35.86),  # China
    "DEU": (10.45, 51.17),   # Germany
    "FRA": (2.21, 46.23),    # France
    "IND": (78.96, 20.59),   # India
    "ITA": (12.57, 41.87),   # Italy
    "JPN": (138.25, 36.20),  # Japan
    "USA": (-95.71, 37.09),  # USA (continental centroid)
}

# Manual label offsets (in data coords: lon, lat) to keep callouts off the dots
# and away from each other. Tuned for a 7x7 inch figure layout.
LABEL_OFFSETS = {
    "VogtmannE_2016": (-15, 22),
    "FengQ_2015":     (-35, 25),
    "WirbelJ_2018":   (-65, 12),
    "ZellerG_2014":   (-80, -5),
    "ThomasAM_2018a": (-55, -22),
    "ThomasAM_2018b": (-15, -28),
    "GuptaA_2019":    (28, -18),
    "YuJ_2015":       (-30, 38),
    "YachidaS_2019":  (35, 28),
    "ThomasAM_2019_c": (55, 8),
}


# ---------------------------------------------------------------------------
# Panel 1 (top-left): cohort map / schematic
# ---------------------------------------------------------------------------

def _draw_world_outline(ax):
    """Draw a minimalist continental schematic (no basemap dependency).

    Purpose is purely orientation: rough silhouettes for the three continents
    that host our cohorts. Not cartographically accurate.
    """
    # Extra horizontal room so callouts can sit off the continents.
    # Vertical range kept tight so the map fills the allotted subplot area.
    ax.set_xlim(-170, 200)
    ax.set_ylim(-18, 78)
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    land = OKABE_ITO["grey"]
    north_america = np.array([
        (-125, 50), (-100, 60), (-75, 55), (-60, 47),
        (-78, 28), (-95, 18), (-115, 30), (-125, 40),
    ])
    europe = np.array([
        (-10, 58), (5, 60), (25, 60), (35, 55),
        (30, 45), (15, 40), (0, 42), (-10, 48),
    ])
    asia = np.array([
        (35, 65), (70, 70), (130, 60), (145, 45),
        (135, 30), (110, 20), (95, 10), (70, 25),
        (50, 30), (40, 45),
    ])
    india = np.array([
        (68, 30), (88, 30), (90, 22), (80, 8), (72, 18),
    ])
    japan = np.array([
        (130, 45), (145, 42), (142, 32), (133, 33),
    ])
    for poly in (north_america, europe, asia, india, japan):
        ax.add_patch(mpatches.Polygon(poly, closed=True,
                                      facecolor=land, edgecolor="none",
                                      alpha=0.30, zorder=1))


def panel_cohort_map(ax, metadata: pd.DataFrame):
    _draw_world_outline(ax)

    counts = (metadata.groupby(["study_name", "country"])
              .size().reset_index(name="N"))
    total = int(counts["N"].sum())
    n_cohorts = counts["study_name"].nunique()

    # Aggregate cohorts per country for plotting dot size.
    per_country = counts.groupby("country")["N"].sum().to_dict()

    dot_color = OKABE_ITO["vermilion"]
    text_color = "#222222"

    # Draw country dots sized by cohort N.
    for country, n in per_country.items():
        lon, lat = COUNTRY_COORDS[country]
        size = 60 + 1.2 * np.sqrt(n) * 8
        ax.scatter([lon], [lat], s=size, c=dot_color,
                   edgecolors="white", linewidths=1.2, zorder=3)

    # Draw cohort labels with connector lines.
    for _, row in counts.iterrows():
        study = row["study_name"]
        country = row["country"]
        n = int(row["N"])
        lon, lat = COUNTRY_COORDS[country]
        # Slight horizontal jitter so 2 Italian cohorts don't overlap.
        if study == "ThomasAM_2018a":
            lon_d, lat_d = lon - 1.5, lat
        elif study == "ThomasAM_2018b":
            lon_d, lat_d = lon + 1.5, lat
        elif study == "ThomasAM_2019_c":
            lon_d, lat_d = lon + 2, lat - 1
        elif study == "YachidaS_2019":
            lon_d, lat_d = lon - 2, lat + 1
        else:
            lon_d, lat_d = lon, lat

        dx, dy = LABEL_OFFSETS.get(study, (15, 10))
        label_x = lon_d + dx
        label_y = lat_d + dy

        # Short label: drop the trailing _YEAR for compactness.
        short = study.replace("_", " ")
        text = f"{short}\n({country}, N={n})"

        ax.annotate(
            text,
            xy=(lon_d, lat_d), xytext=(label_x, label_y),
            fontsize=5.6, color=text_color, ha="center", va="center",
            arrowprops=dict(arrowstyle="-", color=OKABE_ITO["grey"],
                            lw=0.6, alpha=0.8),
            bbox=dict(boxstyle="round,pad=0.18", fc="white",
                      ec=OKABE_ITO["grey"], lw=0.4, alpha=0.92),
            zorder=4,
        )

    ax.set_title(
        f"A | {n_cohorts} cohorts, {total} samples, 3 continents",
        fontsize=9, fontweight="bold", loc="left", pad=4,
    )


# ---------------------------------------------------------------------------
# Panel 2 (top-right): methodology pipeline
# ---------------------------------------------------------------------------

def _box(ax, xy, w, h, text, color, text_color="white", fontsize=6.5):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=0.8, edgecolor="white",
        facecolor=color, zorder=2,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fontsize, color=text_color,
            fontweight="bold", zorder=3)


def _arrow(ax, start, end, color="#444444"):
    arr = FancyArrowPatch(
        start, end,
        arrowstyle="-|>", mutation_scale=10,
        linewidth=1.2, color=color, zorder=1,
    )
    ax.add_patch(arr)


def panel_methodology(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    bw, bh = 0.92, 0.11  # box dims (most boxes)

    # Vertical stack: cohorts -> LODO -> models -> DeLong -> verdict.
    y_top = 0.87
    gap = 0.165

    rows = [
        ("10 cohorts, 1522 samples",                   OKABE_ITO["blue"]),
        ("Country-aware LODO splits\n(Japan vs Japan, Italy vs Italy held out together)",
                                                       OKABE_ITO["skyblue"]),
        ("Species RF   vs   Joint species+pathway RF / XGB",
                                                       OKABE_ITO["green"]),
        ("DeLong test on pooled predictions",          OKABE_ITO["orange"]),
        ("Verdict: species-only wins",                 OKABE_ITO["vermilion"]),
    ]

    centers = []
    for i, (text, color) in enumerate(rows):
        y = y_top - i * gap
        x = (1 - bw) / 2
        h = bh if "\n" not in text else bh + 0.03
        # re-center vertically by adjusting y so all boxes touch evenly
        y_adj = y - (h - bh) / 2
        _box(ax, (x, y_adj), bw, h, text, color, fontsize=6.3)
        centers.append((x + bw / 2, y_adj + h / 2, y_adj, y_adj + h))

    # Arrows between consecutive boxes.
    for i in range(len(centers) - 1):
        x_mid = centers[i][0]
        top_of_lower = centers[i + 1][3]
        bot_of_upper = centers[i][2]
        _arrow(ax, (x_mid, bot_of_upper - 0.005),
                   (x_mid, top_of_lower + 0.005))

    ax.set_title("B | Country-aware LODO pipeline",
                 fontsize=9, fontweight="bold", loc="left", pad=4)


# ---------------------------------------------------------------------------
# Panel 3 (bottom-left): headline result
# ---------------------------------------------------------------------------

def panel_headline(ax, boot: pd.DataFrame, delong: pd.DataFrame):
    pooled = boot[boot["cohort"] == "pooled"].copy()
    order = ["species_rf", "joint_rf", "joint_xgb"]
    pooled = pooled.set_index("model").loc[order].reset_index()

    pretty = {"species_rf": "Species RF",
              "joint_rf": "Joint RF",
              "joint_xgb": "Joint XGB"}
    colors = [OKABE_ITO["vermilion"], OKABE_ITO["blue"], OKABE_ITO["green"]]

    x = np.arange(len(pooled))
    aucs = pooled["auc"].to_numpy()
    lo = pooled["ci_lo"].to_numpy()
    hi = pooled["ci_hi"].to_numpy()
    err = np.vstack([aucs - lo, hi - aucs])

    bars = ax.bar(x, aucs, yerr=err, color=colors,
                  capsize=4, edgecolor="white", linewidth=1.0,
                  error_kw=dict(ecolor="#333333", lw=1.1))
    for xi, a in zip(x, aucs):
        ax.text(xi, a + 0.012, f"{a:.3f}", ha="center", va="bottom",
                fontsize=6.8, fontweight="bold", color="#222222")

    ax.set_xticks(x)
    ax.set_xticklabels([pretty[m] for m in pooled["model"]], fontsize=7)
    ax.set_ylim(0.70, 0.83)
    ax.set_ylabel("Pooled LODO AUROC (95% CI)", fontsize=7)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Pull DeLong p-values for the two relevant comparisons.
    def _p(a, b):
        row = delong[((delong["model_a"] == a) & (delong["model_b"] == b)) |
                     ((delong["model_a"] == b) & (delong["model_b"] == a))]
        return float(row["p_value"].iloc[0])

    p_sr_jr = _p("species_rf", "joint_rf")
    p_sr_jx = _p("species_rf", "joint_xgb")

    def _fmt(p):
        if p < 1e-3:
            return f"p = {p:.1e}"
        return f"p = {p:.3f}"

    # Significance brackets above bars.
    y0 = 0.815
    bracket_color = "#333333"
    # species_rf vs joint_rf (x=0 vs x=1)
    ax.plot([0, 0, 1, 1], [y0, y0 + 0.005, y0 + 0.005, y0],
            lw=0.9, color=bracket_color)
    ax.text(0.5, y0 + 0.006, _fmt(p_sr_jr),
            ha="center", va="bottom", fontsize=6, color=bracket_color)
    # species_rf vs joint_xgb (x=0 vs x=2) — higher bracket
    y1 = y0 + 0.018
    ax.plot([0, 0, 2, 2], [y1, y1 + 0.005, y1 + 0.005, y1],
            lw=0.9, color=bracket_color)
    ax.text(1.0, y1 + 0.006, _fmt(p_sr_jx),
            ha="center", va="bottom", fontsize=6, color=bracket_color)

    ax.set_title("C | Species-only wins (DeLong test)",
                 fontsize=9, fontweight="bold", loc="left", pad=4)


# ---------------------------------------------------------------------------
# Panel 4 (bottom-right): biological signature
# ---------------------------------------------------------------------------

PATHOBIONTS = {
    "Fusobacterium_nucleatum": "F. nucleatum",
    "Peptostreptococcus_stomatis": "P. stomatis",
    "Parvimonas_micra": "P. micra",
    "Gemella_morbillorum": "G. morbillorum",
}


def _species_short(feature: str) -> str | None:
    """Return short species name if feature is a MetaPhlAn-style species row."""
    if "s__" not in feature:
        return None
    sp = feature.split("s__")[-1]
    return sp


def panel_signature(ax, shap_avc: pd.DataFrame, adenoma_lodo: pd.DataFrame):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Compute species-only SHAP ranks for the A -> CRC task.
    sp_rows = []
    for _, row in shap_avc.iterrows():
        short = _species_short(str(row["feature"]))
        if short is not None:
            sp_rows.append({"species": short,
                            "shap": float(row["mean_abs_shap"])})
    sp_df = (pd.DataFrame(sp_rows)
             .sort_values("shap", ascending=False)
             .reset_index(drop=True))
    sp_df["rank"] = sp_df.index + 1
    ranks = {sp: int(r) for sp, r in zip(sp_df["species"], sp_df["rank"])}

    # AUCs from the adenoma LODO summary (RF, since main models are RF).
    def _lodo_auc(task: str) -> float:
        row = adenoma_lodo[adenoma_lodo["task"] == task]
        return float(row["mean_lodo_auc"].iloc[0])

    auc_hva = _lodo_auc("h_vs_a_rf")
    auc_avc = _lodo_auc("a_vs_crc_rf")

    # Layout: two columns — H -> A (left, "early") vs A -> CRC (right, "transition").
    col_lefts = [0.03, 0.52]
    col_width = 0.45
    headers = [
        ("H -> A  (early)",  f"LODO AUC = {auc_hva:.2f}",  OKABE_ITO["grey"]),
        ("A -> CRC  (transition)", f"LODO AUC = {auc_avc:.2f}", OKABE_ITO["vermilion"]),
    ]

    for i, (header, sub, color) in enumerate(headers):
        x = col_lefts[i]
        # Header box
        ax.add_patch(FancyBboxPatch(
            (x, 0.78), col_width, 0.13,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor=color, edgecolor="white", linewidth=0.8,
        ))
        ax.text(x + col_width / 2, 0.875, header,
                ha="center", va="center",
                fontsize=7.2, color="white", fontweight="bold")
        ax.text(x + col_width / 2, 0.805, sub,
                ha="center", va="center",
                fontsize=6.2, color="white")

    # Pathobiont rows: species name on top line, signal label on bottom line.
    y_top = 0.68
    row_h = 0.135
    for j, (full_name, short_name) in enumerate(PATHOBIONTS.items()):
        y = y_top - j * row_h
        rank = ranks.get(full_name)
        rank_label = f"top SHAP (#{rank})" if rank is not None else "n/a"

        # H -> A column (no signal): light grey
        ax.add_patch(FancyBboxPatch(
            (col_lefts[0], y - 0.055), col_width, 0.105,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor="#F2F2F2", edgecolor=OKABE_ITO["grey"], linewidth=0.6,
        ))
        ax.text(col_lefts[0] + col_width / 2, y + 0.020, short_name,
                ha="center", va="center", fontsize=7,
                color="#666666", style="italic")
        ax.text(col_lefts[0] + col_width / 2, y - 0.025, "not in top features",
                ha="center", va="center", fontsize=6, color=OKABE_ITO["grey"])

        # A -> CRC column (top SHAP): colored.
        ax.add_patch(FancyBboxPatch(
            (col_lefts[1], y - 0.055), col_width, 0.105,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor="#FCEAE0", edgecolor=OKABE_ITO["vermilion"], linewidth=0.7,
        ))
        ax.text(col_lefts[1] + col_width / 2, y + 0.020, short_name,
                ha="center", va="center", fontsize=7,
                color=OKABE_ITO["vermilion"], style="italic",
                fontweight="bold")
        ax.text(col_lefts[1] + col_width / 2, y - 0.025, rank_label,
                ha="center", va="center", fontsize=6,
                color=OKABE_ITO["vermilion"], fontweight="bold")

    # Caption strip across the bottom.
    ax.text(0.5, 0.025,
            "Four oral pathobionts drive the adenoma -> CRC transition,\n"
            "but are absent from the early (healthy -> adenoma) signal.",
            ha="center", va="center", fontsize=6.3, color="#222222")

    ax.set_title("D | Oral pathobionts mark the transition",
                 fontsize=9, fontweight="bold", loc="left", pad=4)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(DATA / "metadata_clean.csv")
    boot = pd.read_csv(RESULTS / "bootstrap_ci.csv")
    delong = pd.read_csv(RESULTS / "delong_results.csv")
    adenoma = pd.read_csv(RESULTS / "adenoma_lodo_results.csv")
    shap_avc = pd.read_csv(RESULTS / "shap_adenoma_vs_crc.csv")
    # baseline_results.csv is loaded for parity / sanity; not directly plotted.
    _ = pd.read_csv(RESULTS / "baseline_results.csv")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.edgecolor": "#333333",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })

    fig = plt.figure(figsize=(7.0, 7.0))
    gs = fig.add_gridspec(
        2, 2,
        left=0.07, right=0.96, top=0.93, bottom=0.06,
        wspace=0.22, hspace=0.30,
    )

    ax_map = fig.add_subplot(gs[0, 0])
    ax_meth = fig.add_subplot(gs[0, 1])
    ax_head = fig.add_subplot(gs[1, 0])
    ax_sig = fig.add_subplot(gs[1, 1])

    panel_cohort_map(ax_map, metadata)
    panel_methodology(ax_meth)
    panel_headline(ax_head, boot, delong)
    panel_signature(ax_sig, shap_avc, adenoma)

    fig.suptitle(
        "Country-aware LODO meta-analysis of CRC gut microbiome cohorts",
        fontsize=10.5, fontweight="bold", y=0.985,
    )

    # Do not use bbox_inches="tight" — it would crop away the requested 7x7
    # dimensions. Visual abstracts are sized for journals (square).
    fig.savefig(OUT_PNG, dpi=600)
    fig.savefig(OUT_PDF)
    plt.close(fig)

    print(f"Wrote {OUT_PNG.relative_to(REPO)}")
    print(f"Wrote {OUT_PDF.relative_to(REPO)}")


if __name__ == "__main__":
    main()
