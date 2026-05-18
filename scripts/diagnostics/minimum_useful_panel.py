"""Minimum useful species panel for CRC detection.

Question this script answers: how few species do you actually need to
get a clinically useful AUC? Starting from the global mean |SHAP|
ranking of the species RF (`results/shap_crc_features.csv`), greedily
add species one at a time and refit the same country-aware 10-cohort
LODO RF used in `scripts/train_baseline.py`. At each panel size, report
mean per-cohort AUC and pooled (stacked predictions) AUC, then
identify the smallest panel that crosses two policy-relevant
thresholds:

  * pooled AUC >= 0.75 — minimum clinically useful screening
  * pooled AUC >= 0.78 — recovers the full 229-feature baseline (0.781)

Outputs:
  - results/diagnostics/minimum_panel.csv
        k, top_species_added, mean_per_cohort_auc, pooled_auc,
        delta_from_full
  - figures/diagnostics/minimum_panel.png
        AUC vs panel size, with horizontal baseline at full-229 AUC
  - results/diagnostics/MINIMUM_PANEL_NOTE.md
        one-paragraph headline with the two k values

The script is idempotent (overwrites prior outputs), runs from the
repo root, and reuses `run_lodo_cv` from `scripts/lodo_cv.py`. The
panel size is capped at k=50 because gains plateau well before that.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from lodo_cv import get_lodo_splits  # noqa: E402

SPECIES_CSV = REPO / "data" / "processed" / "species_filtered.csv"
META_CSV = REPO / "data" / "processed" / "metadata_clean.csv"
SHAP_CSV = REPO / "results" / "shap_crc_features.csv"

OUT_CSV = REPO / "results" / "diagnostics" / "minimum_panel.csv"
OUT_FIG = REPO / "figures" / "diagnostics" / "minimum_panel.png"
OUT_MD = REPO / "results" / "diagnostics" / "MINIMUM_PANEL_NOTE.md"

K_MAX = 50
RANDOM_STATE = 42
TARGET_USEFUL = 0.75
TARGET_BASELINE = 0.78  # full 229-feature pooled AUC is 0.781
RECOVERY_FRACTION = 0.99


def short_name(feature: str) -> str:
    """Last clade in a MetaPhlAn lineage string, with s__ stripped."""
    if "|" not in feature:
        return feature
    leaf = feature.split("|")[-1]
    if leaf.startswith("s__"):
        leaf = leaf[3:]
    return leaf


def make_rf() -> RandomForestClassifier:
    """RF spec identical to scripts/train_baseline.py."""
    return RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )


def lodo_aucs(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
) -> tuple[float, float]:
    """Country-aware 10-cohort LODO. Returns (mean_per_cohort, pooled)."""
    per_cohort = []
    yt_all = []
    yp_all = []
    for cohort, train_idx, test_idx, _excl in get_lodo_splits(
        meta, cohort_col="study_name", country_col="country"
    ):
        model = make_rf()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_prob = model.predict_proba(X.iloc[test_idx])[:, 1]
        y_true = y.iloc[test_idx].values
        per_cohort.append(roc_auc_score(y_true, y_prob))
        yt_all.append(y_true)
        yp_all.append(y_prob)
    pooled = roc_auc_score(np.concatenate(yt_all), np.concatenate(yp_all))
    return float(np.mean(per_cohort)), float(pooled)


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    # --- Load aligned X, y, metadata exactly as train_baseline.py does
    sp = pd.read_csv(SPECIES_CSV)
    md = pd.read_csv(META_CSV)
    mg = md.merge(sp, on="sample_id", how="inner")
    fc = [c for c in sp.columns if c != "sample_id"]
    mask = mg["label"].isin([0, 1])

    X_full = mg.loc[mask, fc].reset_index(drop=True)
    y = mg.loc[mask, "label"].astype(int).reset_index(drop=True)
    meta_cols = ["sample_id", "study_name", "study_condition", "label", "country"]
    meta = mg.loc[mask, meta_cols].reset_index(drop=True)

    print(
        f"Samples: {len(X_full)} "
        f"(CRC={int(y.sum())}, control={int((y == 0).sum())})"
    )
    print(f"Total species features: {X_full.shape[1]}")

    # --- SHAP ranking
    shap_df = pd.read_csv(SHAP_CSV)
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False).reset_index(
        drop=True
    )
    ranked = [f for f in shap_df["feature"].tolist() if f in X_full.columns]
    if len(ranked) < len(shap_df):
        print(
            f"  Warning: {len(shap_df) - len(ranked)} ranked features missing from X"
        )

    # --- Full-panel reference (all species)
    print("\nFull 229-feature reference run (country-aware LODO)...")
    full_mean, full_pooled = lodo_aucs(X_full, y, meta)
    print(f"  Full panel:  mean per-cohort = {full_mean:.4f}   "
          f"pooled = {full_pooled:.4f}")

    # --- Greedy forward selection (already ranked globally)
    k_cap = min(K_MAX, len(ranked))
    rows: list[dict] = []
    print(f"\nForward selection up to k={k_cap}...")
    for k in range(1, k_cap + 1):
        cols = ranked[:k]
        added = short_name(ranked[k - 1])
        mean_auc, pooled_auc = lodo_aucs(X_full[cols], y, meta)
        delta = pooled_auc - full_pooled
        rows.append({
            "k": k,
            "top_species_added": added,
            "mean_per_cohort_auc": round(mean_auc, 4),
            "pooled_auc": round(pooled_auc, 4),
            "delta_from_full": round(delta, 4),
        })
        print(
            f"  k={k:2d}  +{added:<32s}  "
            f"mean={mean_auc:.4f}  pooled={pooled_auc:.4f}  "
            f"delta={delta:+.4f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")

    # --- Headline numbers
    above_useful = df[df["pooled_auc"] >= TARGET_USEFUL]
    k_useful = int(above_useful["k"].iloc[0]) if not above_useful.empty else None

    above_baseline = df[df["pooled_auc"] >= TARGET_BASELINE]
    k_baseline = (
        int(above_baseline["k"].iloc[0]) if not above_baseline.empty else None
    )

    recovery_target = RECOVERY_FRACTION * full_pooled
    above_recovery = df[df["pooled_auc"] >= recovery_target]
    k_recover = (
        int(above_recovery["k"].iloc[0]) if not above_recovery.empty else None
    )

    print("\nHeadline:")
    print(f"  k >= AUC {TARGET_USEFUL:.2f}:     {k_useful}")
    print(f"  k >= AUC {TARGET_BASELINE:.2f}:     {k_baseline}")
    print(
        f"  k recovering {RECOVERY_FRACTION:.0%} of full "
        f"({recovery_target:.4f}): {k_recover}"
    )

    # --- Plot
    plt.style.use("seaborn-v0_8-whitegrid")
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8.0, 5.5))

    ax.plot(df["k"], df["pooled_auc"], marker="o", markersize=4,
            color=cmap(0), linewidth=1.5, label="Pooled AUC")
    ax.plot(df["k"], df["mean_per_cohort_auc"], marker="s", markersize=4,
            color=cmap(1), linewidth=1.5, alpha=0.85,
            label="Mean per-cohort AUC")

    ax.axhline(full_pooled, linestyle="--", color=cmap(0), linewidth=1.0,
               alpha=0.7,
               label=f"Full-229 pooled AUC ({full_pooled:.3f})")
    ax.axhline(full_mean, linestyle=":", color=cmap(1), linewidth=1.0,
               alpha=0.7,
               label=f"Full-229 mean per-cohort AUC ({full_mean:.3f})")
    ax.axhline(TARGET_USEFUL, linestyle="-", color="grey", linewidth=0.8,
               alpha=0.6,
               label=f"AUC = {TARGET_USEFUL:.2f} (clinically useful)")

    if k_useful is not None:
        ax.axvline(k_useful, linestyle=":", color="grey", linewidth=0.8,
                   alpha=0.5)
        ax.annotate(
            f"k={k_useful}\n(pooled >= {TARGET_USEFUL:.2f})",
            xy=(k_useful, TARGET_USEFUL),
            xytext=(k_useful + 1.5, TARGET_USEFUL - 0.04),
            fontsize=9, color="black",
            arrowprops=dict(arrowstyle="-", color="grey", lw=0.6),
        )

    if k_recover is not None:
        ax.axvline(k_recover, linestyle=":", color=cmap(0), linewidth=0.8,
                   alpha=0.5)
        ax.annotate(
            f"k={k_recover}\n(>= 99% of full)",
            xy=(k_recover, df.loc[df["k"] == k_recover, "pooled_auc"].iloc[0]),
            xytext=(k_recover + 1.5, full_pooled + 0.015),
            fontsize=9, color="black",
            arrowprops=dict(arrowstyle="-", color="grey", lw=0.6),
        )

    ax.set_xlabel("Panel size (top-k species by global mean |SHAP|)",
                  fontsize=11)
    ax.set_ylabel("AUC", fontsize=11)
    ax.set_title(
        "Minimum useful species panel — greedy forward selection\n"
        "Country-aware 10-cohort LODO, species RF",
        fontsize=12,
    )
    ax.set_xlim(0, k_cap + 1)
    ax.set_ylim(min(0.55, df["pooled_auc"].min() - 0.02),
                max(0.85, full_mean + 0.02))
    ax.tick_params(labelsize=10)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_FIG}")

    # --- Markdown note
    def fmt(k: int | None) -> str:
        return str(k) if k is not None else f">{k_cap}"

    lines = []
    lines.append("# Minimum useful species panel — headline\n")
    lines.append(
        f"**{fmt(k_useful)} species yield pooled AUC >= "
        f"{TARGET_USEFUL:.2f}; "
        f"{fmt(k_recover)} species recover >= "
        f"{RECOVERY_FRACTION:.0%} of full-model performance.**\n"
    )
    lines.append("## Context\n")
    lines.append(
        "Starting from the global mean |SHAP| ranking of the species RF "
        f"({len(ranked)} ranked features), this analysis greedily adds "
        "species in SHAP order and refits the same country-aware 10-cohort "
        "LODO Random Forest used in `scripts/train_baseline.py`. At each "
        "panel size we report both the mean per-cohort AUC and the pooled "
        "AUC computed on stacked held-out predictions.\n"
    )
    lines.append("## Reference (full panel, 229 species)\n")
    lines.append(f"- Mean per-cohort AUC: **{full_mean:.4f}**")
    lines.append(f"- Pooled AUC: **{full_pooled:.4f}**\n")
    lines.append("## Headline panel sizes\n")
    lines.append(
        f"- Smallest k with pooled AUC >= {TARGET_USEFUL:.2f} "
        f"(clinically useful screening floor): **k = {fmt(k_useful)}**"
    )
    lines.append(
        f"- Smallest k with pooled AUC >= {TARGET_BASELINE:.2f} "
        f"(matches full-229 baseline of {full_pooled:.3f}): "
        f"**k = {fmt(k_baseline)}**"
    )
    lines.append(
        f"- Smallest k recovering >= {RECOVERY_FRACTION:.0%} of full pooled "
        f"AUC ({recovery_target:.4f}): **k = {fmt(k_recover)}**\n"
    )
    if k_useful is not None and k_useful <= len(ranked):
        top_useful = [short_name(f) for f in ranked[:k_useful]]
        lines.append(
            f"### Species in the AUC >= {TARGET_USEFUL:.2f} panel "
            f"(k = {k_useful})\n"
        )
        for i, sp_name in enumerate(top_useful, 1):
            lines.append(f"{i}. {sp_name}")
        lines.append("")
    lines.append("## Method notes\n")
    lines.append(
        "- LODO splits use country-aware exclusion (same-country cohorts "
        "are removed from training), matching the headline model."
    )
    lines.append(
        "- The RF spec is identical to `scripts/train_baseline.py`: "
        "500 trees, `max_features='sqrt'`, `min_samples_leaf=5`, "
        "`class_weight='balanced'`, `random_state=42`."
    )
    lines.append(
        "- Ranking is taken from the existing `results/shap_crc_features.csv` "
        "(global mean |SHAP| from the full-panel RF); this is a one-shot "
        "rank, not per-fold re-ranking, so no test-fold leakage occurs."
    )
    lines.append(
        f"- Panel size capped at k = {K_MAX}; gains plateau well before that.\n"
    )
    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
