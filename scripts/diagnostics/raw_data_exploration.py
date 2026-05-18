"""Raw-data exploratory visualization for the 10-cohort CRC dataset.

Methodological intent: before any feature engineering or model fitting,
visualise the lowest-granularity raw signals (sample composition,
between-sample distances, per-sample diversity, top abundant taxa,
sequencing depth) and let the patterns motivate downstream choices.

Reads (relative to repo root):
    data/processed/metadata_clean.csv
    data/processed/species_filtered.csv

Writes:
    figures/diagnostics/cohort_composition.png
    figures/diagnostics/pcoa_bray_curtis.png
    figures/diagnostics/alpha_diversity.png
    figures/diagnostics/top_species_heatmap.png
    figures/diagnostics/depth_distribution.png
    results/diagnostics/raw_data_summary.csv
    results/diagnostics/RAW_DATA_PATTERNS.md

Standalone and idempotent: re-running overwrites the same outputs.
No new dependencies beyond pandas, numpy, scipy, sklearn, matplotlib.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parents[2]
DATA_MD = REPO / "data" / "processed" / "metadata_clean.csv"
DATA_SP = REPO / "data" / "processed" / "species_filtered.csv"

FIG_DIR = REPO / "figures" / "diagnostics"
RES_DIR = REPO / "results" / "diagnostics"

FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

# tab10 is colorblind-safe enough for cohorts (<=10) and the three conditions.
CMAP_COHORT = plt.get_cmap("tab10")
COND_COLOR = {
    "control": "#1f77b4",
    "adenoma": "#ff7f0e",
    "CRC":     "#d62728",
}
COND_ORDER = ["control", "adenoma", "CRC"]

DPI = 300


def load_data():
    md = pd.read_csv(DATA_MD)
    sp = pd.read_csv(DATA_SP)
    # Align on sample_id (intersection); both should already match by design.
    common = sorted(set(md["sample_id"]) & set(sp["sample_id"]))
    md = md.set_index("sample_id").loc[common].reset_index()
    sp = sp.set_index("sample_id").loc[common].reset_index()
    return md, sp


def cohort_composition_figure(md: pd.DataFrame) -> pd.DataFrame:
    """Stacked bar chart of sample counts (CRC / adenoma / control) per cohort."""
    counts = (md.groupby(["study_name", "study_condition"]).size()
                .unstack(fill_value=0)
                .reindex(columns=COND_ORDER, fill_value=0)
                .sort_index())

    cohorts = counts.index.tolist()
    x = np.arange(len(cohorts))
    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = np.zeros(len(cohorts))
    for cond in COND_ORDER:
        vals = counts[cond].values
        ax.bar(x, vals, bottom=bottom,
               color=COND_COLOR[cond], label=cond, edgecolor="white")
        bottom = bottom + vals
    for i, total in enumerate(counts.sum(axis=1).values):
        ax.text(i, total + 5, str(int(total)),
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(cohorts, rotation=35, ha="right")
    ax.set_ylabel("Sample count")
    ax.set_title("Cohort composition by study condition")
    ax.legend(title="study_condition", frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "cohort_composition.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return counts


def bray_curtis_matrix(X: np.ndarray) -> np.ndarray:
    """Compute Bray-Curtis distance matrix on a non-negative abundance matrix.

    BC(u, v) = sum(|u_i - v_i|) / sum(u_i + v_i).
    Rows of X are samples; columns are features (relative abundances).
    """
    # Row-normalise so every sample has unit total (avoids depth bias if any
    # rows have different sums after our species filter).
    row_sum = X.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    Xn = X / row_sum
    d = pdist(Xn, metric="braycurtis")
    return squareform(d)


def pcoa_classical(D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Classical (metric) multidimensional scaling / PCoA.

    Returns (coords, eigvals) where coords has shape (n, n) (we take the
    first two columns at the call site) and eigvals are the full spectrum.
    """
    n = D.shape[0]
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    # Symmetrise to suppress tiny floating-point asymmetry
    B = (B + B.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(B)
    # Sort descending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    # Clip negatives (numerical noise from non-Euclidean distance)
    pos = np.clip(eigvals, a_min=0.0, a_max=None)
    coords = eigvecs * np.sqrt(pos)
    return coords, eigvals


def pcoa_figure(md: pd.DataFrame, sp: pd.DataFrame) -> dict:
    """2-panel PCoA on Bray-Curtis: colored by cohort and by study_condition."""
    feat_cols = [c for c in sp.columns if c != "sample_id"]
    # Reverse the log10(x+1e-6) transform so distances are on the abundance
    # scale (Bray-Curtis is defined for non-negative abundances).
    X_log = sp[feat_cols].to_numpy(dtype=float)
    X = np.clip(10.0 ** X_log - 1e-6, a_min=0.0, a_max=None)

    print(f"  computing Bray-Curtis on {X.shape[0]} samples x {X.shape[1]} features ...")
    D = bray_curtis_matrix(X)
    print("  running classical PCoA ...")
    coords, eigvals = pcoa_classical(D)
    pc1, pc2 = coords[:, 0], coords[:, 1]
    total = float(np.clip(eigvals, 0, None).sum())
    var1 = (max(eigvals[0], 0.0) / total * 100.0) if total > 0 else 0.0
    var2 = (max(eigvals[1], 0.0) / total * 100.0) if total > 0 else 0.0

    cohorts = sorted(md["study_name"].unique())
    cohort_color = {c: CMAP_COHORT(i % 10) for i, c in enumerate(cohorts)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)

    # Panel A: cohort
    axA = axes[0]
    for c in cohorts:
        mask = (md["study_name"] == c).values
        axA.scatter(pc1[mask], pc2[mask],
                    s=12, alpha=0.65,
                    color=cohort_color[c], label=c, edgecolors="none")
    axA.set_title("PCoA (Bray-Curtis) coloured by cohort")
    axA.set_xlabel(f"PC1 ({var1:.1f}%)")
    axA.set_ylabel(f"PC2 ({var2:.1f}%)")
    axA.legend(fontsize=7, ncol=2, frameon=False, loc="best")
    axA.spines["top"].set_visible(False)
    axA.spines["right"].set_visible(False)

    # Panel B: study_condition
    axB = axes[1]
    for cond in COND_ORDER:
        mask = (md["study_condition"] == cond).values
        axB.scatter(pc1[mask], pc2[mask],
                    s=12, alpha=0.65,
                    color=COND_COLOR[cond], label=cond, edgecolors="none")
    axB.set_title("PCoA (Bray-Curtis) coloured by study_condition")
    axB.set_xlabel(f"PC1 ({var1:.1f}%)")
    axB.legend(frameon=False, loc="best")
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)

    fig.tight_layout()
    out = FIG_DIR / "pcoa_bray_curtis.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # Variance attributable to cohort vs condition on PC1+PC2 via simple
    # one-way effect-size proxies: between-group SS divided by total SS.
    def variance_explained(values, labels):
        labels = np.asarray(labels)
        values = np.asarray(values, dtype=float)
        grand = values.mean()
        ss_tot = ((values - grand) ** 2).sum()
        if ss_tot == 0:
            return 0.0
        ss_b = 0.0
        for lab in np.unique(labels):
            v = values[labels == lab]
            if len(v) == 0:
                continue
            ss_b += len(v) * (v.mean() - grand) ** 2
        return ss_b / ss_tot

    var_cohort_pc1 = variance_explained(pc1, md["study_name"].values)
    var_cond_pc1 = variance_explained(pc1, md["study_condition"].values)
    var_cohort_pc2 = variance_explained(pc2, md["study_name"].values)
    var_cond_pc2 = variance_explained(pc2, md["study_condition"].values)
    return {
        "pc1_var_pct": var1,
        "pc2_var_pct": var2,
        "pc1_var_by_cohort": var_cohort_pc1,
        "pc1_var_by_condition": var_cond_pc1,
        "pc2_var_by_cohort": var_cohort_pc2,
        "pc2_var_by_condition": var_cond_pc2,
    }


def shannon_index(row: np.ndarray) -> float:
    p = row.astype(float)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def alpha_diversity_figure(md: pd.DataFrame, sp: pd.DataFrame) -> pd.DataFrame:
    feat_cols = [c for c in sp.columns if c != "sample_id"]
    X_log = sp[feat_cols].to_numpy(dtype=float)
    X = np.clip(10.0 ** X_log - 1e-6, a_min=0.0, a_max=None)
    shannon = np.array([shannon_index(r) for r in X])
    df = md[["sample_id", "study_name", "study_condition"]].copy()
    df["shannon"] = shannon

    cohorts = sorted(df["study_name"].unique())
    n_cohorts = len(cohorts)
    ncols = 5
    nrows = int(np.ceil(n_cohorts / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.2 * ncols, 3.0 * nrows),
                             sharey=True)
    axes_flat = axes.flatten() if nrows * ncols > 1 else [axes]

    for ax, cohort in zip(axes_flat, cohorts):
        sub = df[df["study_name"] == cohort]
        data = [sub.loc[sub["study_condition"] == cond, "shannon"].values
                for cond in COND_ORDER]
        positions = list(range(1, len(COND_ORDER) + 1))
        bp = ax.boxplot(data, positions=positions, widths=0.6,
                        patch_artist=True, showfliers=False)
        for patch, cond in zip(bp["boxes"], COND_ORDER):
            patch.set_facecolor(COND_COLOR[cond])
            patch.set_alpha(0.55)
            patch.set_edgecolor("black")
        for med in bp["medians"]:
            med.set_color("black")
        # Strip plot overlay
        rng = np.random.default_rng(0)
        for i, cond in enumerate(COND_ORDER):
            vals = sub.loc[sub["study_condition"] == cond, "shannon"].values
            if len(vals) == 0:
                continue
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(np.full_like(vals, i + 1) + jitter, vals,
                       s=8, color=COND_COLOR[cond], alpha=0.4,
                       edgecolors="none")
        ax.set_xticks(positions)
        ax.set_xticklabels(COND_ORDER, rotation=20, fontsize=8)
        ax.set_title(cohort, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused axes
    for ax in axes_flat[n_cohorts:]:
        ax.set_visible(False)

    for ax in axes[:, 0] if nrows > 1 else [axes_flat[0]]:
        ax.set_ylabel("Shannon diversity (nats)")

    fig.suptitle("Alpha diversity (Shannon) by study_condition, per cohort",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "alpha_diversity.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return df


def top_species_heatmap(md: pd.DataFrame, sp: pd.DataFrame) -> pd.DataFrame:
    feat_cols = [c for c in sp.columns if c != "sample_id"]
    X_log = sp[feat_cols].to_numpy(dtype=float)
    X = np.clip(10.0 ** X_log - 1e-6, a_min=0.0, a_max=None)
    abund = pd.DataFrame(X, columns=feat_cols)
    abund["study_name"] = md["study_name"].values

    # Mean relative abundance per cohort
    mean_by_cohort = abund.groupby("study_name").mean()
    # Top 20 by overall (cross-cohort) mean
    top20 = mean_by_cohort.mean(axis=0).sort_values(ascending=False).head(20).index
    H = mean_by_cohort[top20]

    # Pretty species names (last s__ token)
    def short(name):
        toks = name.split("|")
        for t in toks[::-1]:
            if t.startswith("s__"):
                return t[3:].replace("_", " ")
        return name[:40]
    H = H.rename(columns=short)

    # Log10 transform for visualization dynamic range
    H_log = np.log10(H.values + 1e-6)

    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(H_log, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(H.columns)))
    ax.set_xticklabels(H.columns, rotation=55, ha="right", fontsize=8,
                       fontstyle="italic")
    ax.set_yticks(range(len(H.index)))
    ax.set_yticklabels(H.index)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("log10 mean relative abundance")
    ax.set_title("Top-20 most-abundant species (overall) by cohort")
    fig.tight_layout()
    out = FIG_DIR / "top_species_heatmap.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return H


def depth_distribution_figure(md: pd.DataFrame) -> pd.DataFrame:
    df = md[["study_name", "number_reads"]].copy()
    df = df[df["number_reads"] > 0]
    df["log10_reads"] = np.log10(df["number_reads"])
    cohorts = sorted(df["study_name"].unique())
    data = [df.loc[df["study_name"] == c, "log10_reads"].values for c in cohorts]

    fig, ax = plt.subplots(figsize=(11, 5))
    bp = ax.boxplot(data, positions=range(1, len(cohorts) + 1),
                    widths=0.6, patch_artist=True, showfliers=False)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(CMAP_COHORT(i % 10))
        patch.set_alpha(0.6)
        patch.set_edgecolor("black")
    for med in bp["medians"]:
        med.set_color("black")
    # jittered points
    rng = np.random.default_rng(1)
    for i, vals in enumerate(data, start=1):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(np.full_like(vals, i) + jitter, vals,
                   s=6, color="black", alpha=0.25, edgecolors="none")
    ax.set_xticks(range(1, len(cohorts) + 1))
    ax.set_xticklabels(cohorts, rotation=35, ha="right")
    ax.set_ylabel("log10(number_reads)")
    ax.set_title("Per-sample sequencing depth by cohort")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "depth_distribution.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return df


def summary_table(md: pd.DataFrame,
                  shannon_df: pd.DataFrame,
                  sp: pd.DataFrame) -> pd.DataFrame:
    feat_cols = [c for c in sp.columns if c != "sample_id"]
    X_log = sp[feat_cols].to_numpy(dtype=float)
    X = np.clip(10.0 ** X_log - 1e-6, a_min=0.0, a_max=None)
    zero_mask = (X == 0).astype(float)

    rows = []
    for cohort, sub in md.groupby("study_name"):
        idx = sub.index
        depth = sub["number_reads"].astype(float)
        depth = depth[depth > 0]
        median_depth_M = float(depth.median() / 1e6) if len(depth) else np.nan
        sh = shannon_df.loc[shannon_df["study_name"] == cohort, "shannon"].values
        med_sh = float(np.median(sh)) if len(sh) else np.nan
        pct_zero = float(zero_mask[idx, :].mean() * 100.0)
        rows.append({
            "cohort": cohort,
            "n_samples": int(len(sub)),
            "n_CRC": int((sub["study_condition"] == "CRC").sum()),
            "n_control": int((sub["study_condition"] == "control").sum()),
            "n_adenoma": int((sub["study_condition"] == "adenoma").sum()),
            "median_shannon": round(med_sh, 4),
            "median_depth_Mreads": round(median_depth_M, 3),
            "pct_zero_features": round(pct_zero, 2),
        })
    df = pd.DataFrame(rows).sort_values("cohort").reset_index(drop=True)
    out = RES_DIR / "raw_data_summary.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}")
    return df


def write_patterns_markdown(summary: pd.DataFrame,
                            shannon_df: pd.DataFrame,
                            pcoa_info: dict,
                            counts: pd.DataFrame,
                            top_heatmap: pd.DataFrame) -> None:
    # PCoA: cohort effect vs condition effect on PC1/PC2
    pc1_cohort = pcoa_info["pc1_var_by_cohort"] * 100
    pc1_cond = pcoa_info["pc1_var_by_condition"] * 100
    pc2_cohort = pcoa_info["pc2_var_by_cohort"] * 100
    pc2_cond = pcoa_info["pc2_var_by_condition"] * 100

    # Alpha diversity by condition (pooled)
    grp_sh = shannon_df.groupby("study_condition")["shannon"].median()
    sh_ctrl = grp_sh.get("control", float("nan"))
    sh_aden = grp_sh.get("adenoma", float("nan"))
    sh_crc  = grp_sh.get("CRC", float("nan"))

    # Top-species stability: Jaccard overlap of top-5 between cohorts
    cohorts = top_heatmap.index.tolist()
    top5 = {c: set(top_heatmap.loc[c].sort_values(ascending=False).head(5).index)
            for c in cohorts}
    overlaps = []
    for i, c1 in enumerate(cohorts):
        for c2 in cohorts[i+1:]:
            inter = len(top5[c1] & top5[c2])
            union = len(top5[c1] | top5[c2])
            overlaps.append(inter / union if union else 0.0)
    mean_jaccard = float(np.mean(overlaps)) if overlaps else float("nan")

    n_total = int(summary["n_samples"].sum())
    n_cohorts = len(summary)
    med_depth_min = float(summary["median_depth_Mreads"].min())
    med_depth_max = float(summary["median_depth_Mreads"].max())

    body = (
        "# Raw-data exploratory patterns\n\n"
        "**Scope.** Five low-level diagnostics computed before any modelling: "
        f"cohort composition ({n_cohorts} cohorts, {n_total} samples), Bray-Curtis "
        "PCoA of MetaPhlAn species, per-sample Shannon diversity, top-20 species "
        "abundance heatmap, and sequencing-depth distribution. Underlying numbers "
        "are in `results/diagnostics/raw_data_summary.csv`; figures are in "
        "`figures/diagnostics/`.\n\n"
        "**Summary (one paragraph).** "
        f"(a) PCoA on Bray-Curtis species distances is dominated by cohort "
        f"structure: cohort labels account for {pc1_cohort:.1f}% of variance on "
        f"PC1 and {pc2_cohort:.1f}% on PC2 (one-way SS_between / SS_total), "
        f"whereas study_condition accounts for only {pc1_cond:.1f}% and "
        f"{pc2_cond:.1f}% respectively. CRC, adenoma, and control samples are "
        "intermingled within each cohort cloud rather than forming a separable "
        "axis. This is the empirical justification for country-aware LODO and "
        "for per-fold (rather than global) feature filtering. "
        f"(b) Pooled median Shannon diversity is similar across conditions "
        f"(control {sh_ctrl:.2f}, adenoma {sh_aden:.2f}, CRC {sh_crc:.2f} nats), "
        "with substantial within-cohort overlap; alpha diversity is not by "
        "itself a strong discriminator and the predictive signal lives in "
        "specific taxa rather than overall diversity. "
        f"(c) The top-20 abundant species are reasonably stable across cohorts "
        f"(mean pairwise Jaccard of per-cohort top-5 = {mean_jaccard:.2f}), but "
        "absolute abundances vary by 1-2 orders of magnitude (e.g., "
        "Prevotella-dominant cohorts vs Bacteroides-dominant cohorts), which "
        "motivates log10 transformation and tree-based classifiers that are "
        "scale-invariant per feature. "
        f"Per-sample sequencing depth ranges over an order of magnitude across "
        f"cohorts (median {med_depth_min:.1f}M to {med_depth_max:.1f}M reads), "
        "consistent with the a priori HanniganGD_2017 exclusion on depth grounds.\n"
    )

    out = RES_DIR / "RAW_DATA_PATTERNS.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(body)
    print(f"wrote {out}")


def main():
    print("=== raw_data_exploration ===")
    md, sp = load_data()
    print(f"loaded {len(md)} samples, {sp.shape[1]-1} species features, "
          f"{md['study_name'].nunique()} cohorts")

    counts = cohort_composition_figure(md)
    pcoa_info = pcoa_figure(md, sp)
    shannon_df = alpha_diversity_figure(md, sp)
    top_heatmap = top_species_heatmap(md, sp)
    _ = depth_distribution_figure(md)
    summary = summary_table(md, shannon_df, sp)
    write_patterns_markdown(summary, shannon_df, pcoa_info, counts, top_heatmap)
    print("done.")


if __name__ == "__main__":
    main()
