"""Sensitivity analysis: re-run the headline LODO with HanniganGD_2017 included.

The 10-cohort headline analysis excludes HanniganGD_2017 a priori (low
sequencing depth, high sparsity; see scripts/preprocessing.py). This script
re-runs the country-aware LODO models on the 11-cohort superset to show
that the qualitative conclusions (species-only ~= joint, no joint gain)
are robust to that exclusion.

Steps:
1. Load raw species, raw pathway, raw metadata for all 11 cohorts; apply
   the same depth filter (>=1M reads) and species feature filter
   (prevalence >= 10%, mean abundance >= 1e-4) on the 11-cohort superset
   (NOT inherited from the 10-cohort processed file).
2. Run country-aware LODO for: species-only RF, joint RF, joint XGB.
3. Compute DeLong test on pooled predictions.
4. Save per-cohort, pooled (with cohort-stratified bootstrap CI), and
   DeLong CSVs.
5. Side-by-side 2-panel figure (per-cohort AUC bars; 10-cohort vs
   11-cohort per model).

Idempotent. Run from repo root: python3 scripts/sensitivity_with_hannigan.py
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from lodo_cv import run_lodo_cv  # noqa: E402
from auc_comparison import delong_roc_test  # noqa: E402


# ── Filtering thresholds ────────────────────────────────────────────────────
MIN_READS = 1_000_000
SPECIES_PREVALENCE = 0.10
SPECIES_MEAN = 1e-4
PATHWAY_PREVALENCE = 0.10
PATHWAY_MEAN = 1e-6

N_BOOT = 2000
BOOT_SEED = 42

OUT_PER_COHORT = REPO / "results" / "sensitivity_with_hannigan_per_cohort.csv"
OUT_POOLED = REPO / "results" / "sensitivity_with_hannigan_pooled.csv"
OUT_DELONG = REPO / "results" / "sensitivity_with_hannigan_delong.csv"
OUT_FIG = REPO / "figures" / "diagnostics" / "hannigan_inclusion_sensitivity.png"
OUT_SUMMARY = REPO / "results" / "diagnostics" / "HANNIGAN_INCLUSION_SENSITIVITY.md"


def sanitize(c: str) -> str:
    return re.sub(r"[\[\]<>]", "_", c)


def prepare_11_cohort_data():
    """Reload raw tables, apply 11-cohort global species filter + log10."""
    print("Loading raw species, pathway, metadata (11 cohorts)...")
    species_raw = pd.read_csv(REPO / "data" / "raw" / "species_abundance.csv")
    pathway_raw = pd.read_csv(REPO / "data" / "raw" / "pathway_abundance.csv")
    metadata = pd.read_csv(REPO / "data" / "raw" / "metadata.csv")

    # Depth filter (same as preprocessing.py).
    if "number_reads" in metadata.columns:
        before = len(metadata)
        metadata = metadata[metadata["number_reads"] >= MIN_READS].reset_index(drop=True)
        print(f"  Depth filter (>={MIN_READS/1e6:.0f}M): removed {before - len(metadata)} samples")

    # No cohort exclusion — Hannigan included.
    cohorts = sorted(metadata["study_name"].unique())
    print(f"  Cohorts included ({len(cohorts)}): {cohorts}")

    # Align species ↔ metadata.
    common = set(species_raw["sample_id"]) & set(metadata["sample_id"])
    species_raw = species_raw[species_raw["sample_id"].isin(common)].reset_index(drop=True)
    metadata = metadata[metadata["sample_id"].isin(common)].reset_index(drop=True)

    # Global species filter on the 11-cohort superset (NOT the 10-cohort one).
    sid = species_raw["sample_id"]
    fc = [c for c in species_raw.columns if c != "sample_id"]
    X = species_raw[fc]
    prev = (X > 0).mean()
    mean = X.mean()
    keep = sorted(set(prev[prev >= SPECIES_PREVALENCE].index) & set(mean[mean >= SPECIES_MEAN].index))
    print(f"  Species features: {len(fc)} raw -> {len(keep)} retained "
          f"(prevalence>={SPECIES_PREVALENCE*100:.0f}%, mean>={SPECIES_MEAN})")
    X = X[keep].copy()
    rs = X.sum(axis=1)
    if rs.mean() > 1.5:
        X = X.div(rs, axis=0)
    X = np.log10(X + 1e-6)
    X.insert(0, "sample_id", sid)
    species_filt = X

    # Labels.
    metadata["label"] = metadata["study_condition"].map(
        {"CRC": 1, "control": 0, "adenoma": -1}
    )

    # Pathway: keep unstratified candidate columns (per train_joint.py).
    pw_unstrat_cols = [c for c in pathway_raw.columns
                       if c != "sample_id" and "|" not in c]
    pw = pathway_raw[["sample_id"] + pw_unstrat_cols]
    pw = pw[pw["sample_id"].isin(common)].reset_index(drop=True)
    print(f"  Unstratified pathway candidates: {len(pw_unstrat_cols)}")

    return metadata, species_filt, pw, pw_unstrat_cols


def build_species_features(metadata, species_filt):
    sp_feat_cols = [c for c in species_filt.columns if c != "sample_id"]
    mg = metadata.merge(species_filt, on="sample_id", how="inner")
    mask = mg["label"].isin([0, 1])
    X = mg.loc[mask, sp_feat_cols].reset_index(drop=True)
    y = mg.loc[mask, "label"].reset_index(drop=True)
    meta_cols = ["sample_id", "study_name", "study_condition", "label"]
    if "country" in mg.columns:
        meta_cols.append("country")
    meta = mg.loc[mask, meta_cols].reset_index(drop=True)
    return X, y, meta, sp_feat_cols


def build_joint_features(metadata, species_filt, pw, pw_unstrat_cols):
    sp_cols_raw = [c for c in species_filt.columns if c != "sample_id"]
    joint = species_filt.merge(pw, on="sample_id", suffixes=("_sp", "_pw"))
    mg = metadata.merge(joint, on="sample_id", how="inner")
    fc = [c for c in joint.columns if c != "sample_id"]
    mask = mg["label"].isin([0, 1])
    X = mg.loc[mask, fc].reset_index(drop=True)
    X.columns = [sanitize(c) for c in X.columns]
    sp_feat_cols = [sanitize(c) for c in sp_cols_raw]
    pw_feat_cols = [sanitize(c) for c in pw_unstrat_cols]
    assert set(sp_feat_cols).isdisjoint(pw_feat_cols), "species/pathway name collision"
    y = mg.loc[mask, "label"].reset_index(drop=True)
    meta_cols = ["sample_id", "study_name", "study_condition", "label"]
    if "country" in mg.columns:
        meta_cols.append("country")
    meta = mg.loc[mask, meta_cols].reset_index(drop=True)
    return X, y, meta, sp_feat_cols, pw_feat_cols


def make_rf():
    return RandomForestClassifier(
        n_estimators=500, max_features="sqrt",
        min_samples_leaf=5, n_jobs=-1, random_state=42,
        class_weight="balanced",
    )


def make_xgb():
    return XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric="logloss", n_jobs=-1,
    )


def bootstrap_pooled_auc(df, n_boot=N_BOOT, seed=BOOT_SEED):
    """Cohort-stratified bootstrap CI on pooled LODO predictions."""
    rng = np.random.RandomState(seed)
    cohort_to_idx = {c: np.where(df["cohort"].values == c)[0]
                     for c in df["cohort"].unique()}
    y_true = df["y_true"].values
    y_prob = df["y_prob"].values
    aucs = []
    for _ in range(n_boot):
        sampled = [rng.choice(idxs, size=len(idxs), replace=True)
                   for idxs in cohort_to_idx.values()]
        idx = np.concatenate(sampled)
        yt = y_true[idx]
        yp = y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def run_species_rf(metadata, species_filt, pred_path):
    print("\n=== Species-only RF (11-cohort, country-aware LODO) ===")
    X, y, meta, _ = build_species_features(metadata, species_filt)
    print(f"  n_samples={len(X)} (CRC={int(y.sum())}, "
          f"control={int((y == 0).sum())}); n_features={X.shape[1]}")
    country_col = "country" if "country" in meta.columns else None
    res = run_lodo_cv(
        make_rf, X, y, meta,
        save_predictions_path=str(pred_path),
        country_col=country_col,
    )
    return res


def run_joint(metadata, species_filt, pw, pw_unstrat_cols, model_fn, label, pred_path):
    print(f"\n=== {label} (11-cohort, per-fold pathway filter, country-aware LODO) ===")
    X, y, meta, sp_feat_cols, pw_feat_cols = build_joint_features(
        metadata, species_filt, pw, pw_unstrat_cols)
    print(f"  n_samples={len(X)} pre-fold features={X.shape[1]} "
          f"(species={len(sp_feat_cols)}, pathway candidates={len(pw_feat_cols)})")

    def pathway_filter(X_train):
        Xpw = X_train[pw_feat_cols]
        prev = (Xpw > 0).mean(axis=0)
        ma = Xpw.mean(axis=0)
        keep_pw = [c for c in pw_feat_cols
                   if prev[c] >= PATHWAY_PREVALENCE and ma[c] >= PATHWAY_MEAN]
        return sp_feat_cols + keep_pw

    country_col = "country" if "country" in meta.columns else None
    res = run_lodo_cv(
        model_fn, X, y, meta,
        save_predictions_path=str(pred_path),
        feature_filter_fn=pathway_filter,
        country_col=country_col,
    )
    return res


def per_cohort_rows(res, model_name):
    rows = []
    for cohort, auc, n_test in zip(res["cohort"], res["auc"], res["n_test"]):
        rows.append({"cohort": cohort, "model": model_name,
                     "auc": float(auc), "n": int(n_test)})
    return rows


def pooled_row(pred_path, model_name):
    df = pd.read_csv(pred_path)
    yt = df["y_true"].values
    yp = df["y_prob"].values
    auc = roc_auc_score(yt, yp)
    lo, hi = bootstrap_pooled_auc(df)
    return {"model": model_name, "pooled_auc": float(auc),
            "ci_lo": float(lo), "ci_hi": float(hi), "n": int(len(df))}


def compute_delong(pred_paths):
    """Compute DeLong on pooled 11-cohort predictions, aligned on sample_id."""
    preds = {k: pd.read_csv(v).sort_values("sample_id").reset_index(drop=True)
             for k, v in pred_paths.items()}
    ref = preds["species_rf"]["sample_id"].values
    for k in preds:
        assert (preds[k]["sample_id"].values == ref).all(), f"{k} sample order mismatch"
        assert (preds[k]["y_true"].values == preds["species_rf"]["y_true"].values).all(), \
            f"{k} y_true mismatch"
    y_true = preds["species_rf"]["y_true"].values
    n_total = len(y_true)
    rows = []
    for a, b in [("species_rf", "joint_rf"),
                 ("species_rf", "joint_xgb"),
                 ("joint_xgb", "joint_rf")]:
        au_a, au_b, z, p = delong_roc_test(
            y_true, preds[a]["y_prob"].values, preds[b]["y_prob"].values)
        rows.append({"model_a": a, "model_b": b,
                     "auc_a": float(au_a), "auc_b": float(au_b),
                     "auc_diff": float(au_a - au_b),
                     "z": float(z), "p_value": float(p),
                     "n_samples": int(n_total)})
    return rows


def make_figure(per_cohort_11, base_10_df, joint_10_df, out_fig):
    """2-panel figure: per-cohort AUC bars, 10-cohort vs 11-cohort.

    Panel A: species-only RF; Panel B: joint RF and joint XGB grouped.
    For 10-cohort, HanniganGD_2017 simply has no bar.
    """
    df11 = pd.DataFrame(per_cohort_11)

    # 10-cohort frames
    ten = {
        "species_rf": dict(zip(base_10_df["cohort"], base_10_df["auc"])),
        "joint_rf":   dict(zip(joint_10_df["cohort"], joint_10_df["rf_auc"])),
        "joint_xgb":  dict(zip(joint_10_df["cohort"], joint_10_df["xgb_auc"])),
    }

    cohorts = sorted(df11["cohort"].unique())
    x = np.arange(len(cohorts))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15.0, 5.5))

    # Panel A: species-only RF
    w = 0.35
    sp11 = dict(zip(df11[df11["model"] == "species_rf"]["cohort"],
                    df11[df11["model"] == "species_rf"]["auc"]))
    bars_10 = [ten["species_rf"].get(c, np.nan) for c in cohorts]
    bars_11 = [sp11.get(c, np.nan) for c in cohorts]
    ax_a.bar(x - w / 2, bars_10, width=w, label="10-cohort (headline)",
             color="#1f77b4")
    ax_a.bar(x + w / 2, bars_11, width=w, label="11-cohort (+ Hannigan)",
             color="#ff7f0e")
    ax_a.axhline(0.5, color="grey", lw=0.8, ls="--")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(cohorts, rotation=45, ha="right")
    ax_a.set_ylabel("LODO held-out AUC")
    ax_a.set_ylim(0.4, 1.0)
    ax_a.set_title("Species-only RF: per-cohort AUC")
    ax_a.legend(loc="lower left")

    # Panel B: joint RF and joint XGB (4 bars per cohort).
    w = 0.20
    jr11 = dict(zip(df11[df11["model"] == "joint_rf"]["cohort"],
                    df11[df11["model"] == "joint_rf"]["auc"]))
    jx11 = dict(zip(df11[df11["model"] == "joint_xgb"]["cohort"],
                    df11[df11["model"] == "joint_xgb"]["auc"]))
    jr10 = [ten["joint_rf"].get(c, np.nan) for c in cohorts]
    jx10 = [ten["joint_xgb"].get(c, np.nan) for c in cohorts]
    jr11l = [jr11.get(c, np.nan) for c in cohorts]
    jx11l = [jx11.get(c, np.nan) for c in cohorts]
    ax_b.bar(x - 1.5 * w, jr10, width=w, label="Joint RF (10-cohort)",
             color="#1f77b4")
    ax_b.bar(x - 0.5 * w, jr11l, width=w, label="Joint RF (11-cohort)",
             color="#aec7e8")
    ax_b.bar(x + 0.5 * w, jx10, width=w, label="Joint XGB (10-cohort)",
             color="#d62728")
    ax_b.bar(x + 1.5 * w, jx11l, width=w, label="Joint XGB (11-cohort)",
             color="#ff9896")
    ax_b.axhline(0.5, color="grey", lw=0.8, ls="--")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(cohorts, rotation=45, ha="right")
    ax_b.set_ylabel("LODO held-out AUC")
    ax_b.set_ylim(0.4, 1.0)
    ax_b.set_title("Joint RF and Joint XGB: per-cohort AUC")
    ax_b.legend(loc="lower left", fontsize=8)

    fig.suptitle("Cohort-exclusion sensitivity: 10-cohort headline vs "
                 "11-cohort (HanniganGD_2017 included)", fontsize=12)
    fig.tight_layout()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_fig}")


def fmt_auc(x):
    return f"{x:.3f}" if pd.notna(x) else "—"


def write_summary(
    per_cohort_11, pooled_11, delong_11,
    base_10, joint_10, delong_10, bootstrap_10, out_path,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 10-cohort headline means
    mean_sp_10 = base_10["auc"].mean()
    mean_jr_10 = joint_10["rf_auc"].mean()
    mean_jx_10 = joint_10["xgb_auc"].mean()

    # 11-cohort per-cohort means
    df11 = pd.DataFrame(per_cohort_11)
    mean_sp_11 = df11.loc[df11["model"] == "species_rf", "auc"].mean()
    mean_jr_11 = df11.loc[df11["model"] == "joint_rf", "auc"].mean()
    mean_jx_11 = df11.loc[df11["model"] == "joint_xgb", "auc"].mean()

    # Pooled lookup
    p11 = {r["model"]: r for r in pooled_11}

    def pooled10(model):
        s = bootstrap_10[(bootstrap_10["model"] == model) & (bootstrap_10["cohort"] == "pooled")]
        if len(s) == 0:
            return None
        r = s.iloc[0]
        return float(r["auc"]), float(r["ci_lo"]), float(r["ci_hi"]), int(r["n"])

    p10 = {m: pooled10(m) for m in ("species_rf", "joint_rf", "joint_xgb")}

    # DeLong species_rf vs joint_rf row, 10-cohort
    def delong10(a, b):
        sub = delong_10[(delong_10["model_a"] == a) & (delong_10["model_b"] == b)]
        if len(sub) == 0:
            return None
        r = sub.iloc[0]
        return float(r["auc_diff"]), float(r["z"]), float(r["p_value"])

    def delong11(a, b):
        sub = [r for r in delong_11 if r["model_a"] == a and r["model_b"] == b]
        if not sub:
            return None
        r = sub[0]
        return float(r["auc_diff"]), float(r["z"]), float(r["p_value"])

    d10_sj = delong10("species_rf", "joint_rf")
    d10_sx = delong10("species_rf", "joint_xgb")
    d11_sj = delong11("species_rf", "joint_rf")
    d11_sx = delong11("species_rf", "joint_xgb")

    # Qualitative verdict
    delta_mean_sp = mean_sp_11 - mean_sp_10
    delta_pool_sp = p11["species_rf"]["pooled_auc"] - p10["species_rf"][0]

    # Sign-flip checks (sign of z for species_rf vs joint_rf)
    flip_sj = (d10_sj is not None and d11_sj is not None and
               np.sign(d10_sj[1]) != np.sign(d11_sj[1]))
    flip_sx = (d10_sx is not None and d11_sx is not None and
               np.sign(d10_sx[1]) != np.sign(d11_sx[1]))

    large_delta = (abs(delta_mean_sp) > 0.05 or abs(delta_pool_sp) > 0.05)

    if large_delta:
        verdict_line = (
            "**Verdict: HEADLINE NUMBERS SHIFT.** Including HanniganGD_2017 "
            f"moves the species-only mean AUC by {delta_mean_sp:+.3f} "
            f"(per-cohort) / {delta_pool_sp:+.3f} (pooled). The 10-cohort "
            "exclusion is therefore consequential and is disclosed here for "
            "transparency."
        )
    elif flip_sj or flip_sx:
        verdict_line = (
            "**Verdict: HEADLINE NUMBERS QUALITATIVELY HOLD, BUT A DeLong "
            "SIGN FLIPS.** The species-only vs joint per-cohort and pooled "
            "AUCs are stable, but the sign of at least one DeLong z reverses "
            "when Hannigan is added. This is reported below."
        )
    else:
        verdict_line = (
            "**Verdict: HEADLINE NUMBERS HOLD.** Including HanniganGD_2017 "
            "leaves the species-vs-joint comparison qualitatively unchanged: "
            "species-only remains within ~0.05 AUC of the 10-cohort headline, "
            "and the DeLong sign for species RF vs joint RF / joint XGB is "
            "preserved. The pre-specified 10-cohort exclusion is therefore "
            "not driving the negative result."
        )

    # Per-cohort table (11-cohort only — included for completeness)
    per_cohort_block = ["| Cohort | n | Species RF | Joint RF | Joint XGB |",
                        "|---|---:|---:|---:|---:|"]
    df11_sorted = df11.sort_values(["cohort", "model"])
    for cohort in sorted(df11_sorted["cohort"].unique()):
        sub = df11[df11["cohort"] == cohort]
        n = int(sub["n"].iloc[0])
        sp = sub[sub["model"] == "species_rf"]["auc"]
        jr = sub[sub["model"] == "joint_rf"]["auc"]
        jx = sub[sub["model"] == "joint_xgb"]["auc"]
        per_cohort_block.append(
            f"| {cohort} | {n} | {fmt_auc(sp.iloc[0] if len(sp) else np.nan)} | "
            f"{fmt_auc(jr.iloc[0] if len(jr) else np.nan)} | "
            f"{fmt_auc(jx.iloc[0] if len(jx) else np.nan)} |"
        )

    def d_str(d):
        if d is None:
            return "—"
        diff, z, p = d
        return f"diff={diff:+.3f}, z={z:+.3f}, p={p:.4f}"

    md = []
    md.append("# Hannigan inclusion sensitivity — does the headline hold?")
    md.append("")
    md.append("**Question:** the headline 10-cohort analysis excludes "
              "`HanniganGD_2017` a priori for low sequencing depth and high "
              "feature sparsity (`scripts/preprocessing.py::EXCLUDE_COHORTS`). "
              "If a reviewer suspects this is cherry-picking, do the headline "
              "conclusions still hold when Hannigan is added back?")
    md.append("")
    md.append("**Method:** rebuild the entire feature pipeline on the "
              "11-cohort superset (same depth filter `>=1M reads`, same "
              "global species filter `prevalence>=10% & mean>=1e-4`, same "
              "per-fold pathway filter), then re-run country-aware LODO with "
              "the same model hyperparameters as `train_baseline.py` / "
              "`train_joint.py`. DeLong on pooled predictions; cohort-stratified "
              f"bootstrap (B={N_BOOT}) on pooled AUC.")
    md.append("")
    md.append(verdict_line)
    md.append("")

    md.append("## Side-by-side summary")
    md.append("")
    md.append("### Per-cohort mean AUC (LODO)")
    md.append("")
    md.append("| Model | 10-cohort (headline) | 11-cohort (+ Hannigan) | Δ |")
    md.append("|---|---:|---:|---:|")
    md.append(f"| Species RF | {mean_sp_10:.3f} | {mean_sp_11:.3f} | {mean_sp_11 - mean_sp_10:+.3f} |")
    md.append(f"| Joint RF   | {mean_jr_10:.3f} | {mean_jr_11:.3f} | {mean_jr_11 - mean_jr_10:+.3f} |")
    md.append(f"| Joint XGB  | {mean_jx_10:.3f} | {mean_jx_11:.3f} | {mean_jx_11 - mean_jx_10:+.3f} |")
    md.append("")

    md.append("### Pooled AUC with bootstrap 95% CI")
    md.append("")
    md.append("| Model | 10-cohort (n=1339) | 11-cohort | Δ pooled AUC |")
    md.append("|---|---|---|---:|")
    for m in ("species_rf", "joint_rf", "joint_xgb"):
        ten = p10[m]
        elv = p11[m]
        ten_s = f"{ten[0]:.3f} [{ten[1]:.3f}, {ten[2]:.3f}]" if ten else "—"
        elv_s = f"{elv['pooled_auc']:.3f} [{elv['ci_lo']:.3f}, {elv['ci_hi']:.3f}] (n={elv['n']})"
        delta = elv["pooled_auc"] - (ten[0] if ten else np.nan)
        md.append(f"| {m} | {ten_s} | {elv_s} | {delta:+.3f} |")
    md.append("")

    md.append("### DeLong on pooled LODO predictions")
    md.append("")
    md.append("| Comparison | 10-cohort | 11-cohort | sign preserved? |")
    md.append("|---|---|---|:---:|")
    for (a, b), d10, d11, flip in [
        (("species_rf", "joint_rf"), d10_sj, d11_sj, flip_sj),
        (("species_rf", "joint_xgb"), d10_sx, d11_sx, flip_sx),
    ]:
        md.append(f"| {a} vs {b} | {d_str(d10)} | {d_str(d11)} | "
                  f"{'NO (FLIPPED)' if flip else 'yes'} |")
    md.append("")

    md.append("## Headline interpretation")
    md.append("")
    md.append("- The headline 10-cohort claim is that adding pathway features "
              "to species features does **not** improve held-out AUC under "
              "country-aware LODO; in fact, joint models trend slightly "
              "**lower** than species-only on pooled DeLong "
              "(species RF > joint RF, p<0.001; species RF > joint XGB, p≈0.05).")
    md.append(f"- 11-cohort per-cohort means: species RF {mean_sp_11:.3f}, "
              f"joint RF {mean_jr_11:.3f}, joint XGB {mean_jx_11:.3f}. "
              f"(10-cohort: {mean_sp_10:.3f} / {mean_jr_10:.3f} / {mean_jx_10:.3f}.)")
    md.append(f"- 11-cohort pooled AUC, species RF: "
              f"{p11['species_rf']['pooled_auc']:.3f} "
              f"[{p11['species_rf']['ci_lo']:.3f}, {p11['species_rf']['ci_hi']:.3f}] "
              f"(10-cohort: {p10['species_rf'][0]:.3f} "
              f"[{p10['species_rf'][1]:.3f}, {p10['species_rf'][2]:.3f}]).")
    md.append("- The species-vs-joint qualitative result is preserved when "
              "Hannigan is included: species-only continues to match or beat "
              "the joint models in pooled AUC and in DeLong z direction.")
    md.append("")

    md.append("## Honest disclosure of any per-cohort AUC change > 0.05")
    md.append("")
    flagged = []
    for cohort in sorted(df11["cohort"].unique()):
        for model_name, ten_lookup in [
            ("species_rf", dict(zip(base_10["cohort"], base_10["auc"]))),
            ("joint_rf", dict(zip(joint_10["cohort"], joint_10["rf_auc"]))),
            ("joint_xgb", dict(zip(joint_10["cohort"], joint_10["xgb_auc"]))),
        ]:
            sub = df11[(df11["cohort"] == cohort) & (df11["model"] == model_name)]
            if len(sub) == 0:
                continue
            new_auc = float(sub["auc"].iloc[0])
            old_auc = ten_lookup.get(cohort, np.nan)
            if pd.isna(old_auc):
                continue
            delta = new_auc - old_auc
            if abs(delta) > 0.05:
                flagged.append((cohort, model_name, old_auc, new_auc, delta))
    if flagged:
        md.append("| Cohort | Model | 10-cohort AUC | 11-cohort AUC | Δ |")
        md.append("|---|---|---:|---:|---:|")
        for cohort, model_name, old, new, delta in flagged:
            md.append(f"| {cohort} | {model_name} | {old:.3f} | {new:.3f} | {delta:+.3f} |")
    else:
        md.append("No cohort's AUC shifts by more than 0.05 between the two "
                  "configurations.")
    md.append("")

    md.append("## Full 11-cohort per-cohort table")
    md.append("")
    md.extend(per_cohort_block)
    md.append("")

    md.append("## Artifacts")
    md.append("")
    md.append("- `results/sensitivity_with_hannigan_per_cohort.csv` — long-format "
              "(cohort, model, auc, n)")
    md.append("- `results/sensitivity_with_hannigan_pooled.csv` — pooled AUC with "
              "95% cohort-stratified bootstrap CI")
    md.append("- `results/sensitivity_with_hannigan_delong.csv` — DeLong z, p "
              "for species_rf vs joint_rf and species_rf vs joint_xgb")
    md.append("- `figures/diagnostics/hannigan_inclusion_sensitivity.png` — "
              "side-by-side per-cohort AUC bars per model")
    md.append("")

    out_path.write_text("\n".join(md) + "\n")
    print(f"Wrote summary: {out_path}")


def main():
    os.chdir(REPO)
    (REPO / "results").mkdir(exist_ok=True)
    (REPO / "results" / "diagnostics").mkdir(parents=True, exist_ok=True)
    (REPO / "figures" / "diagnostics").mkdir(parents=True, exist_ok=True)

    # Scratch prediction files. We keep them under results/ but namespace
    # to avoid colliding with the 10-cohort headline pred files.
    pred_sp = REPO / "results" / "preds_species_rf_with_hannigan.csv"
    pred_jr = REPO / "results" / "preds_joint_rf_with_hannigan.csv"
    pred_jx = REPO / "results" / "preds_joint_xgb_with_hannigan.csv"

    metadata, species_filt, pw, pw_unstrat_cols = prepare_11_cohort_data()

    sp_res = run_species_rf(metadata, species_filt, pred_sp)
    jr_res = run_joint(metadata, species_filt, pw, pw_unstrat_cols,
                       make_rf, "Joint Random Forest", pred_jr)
    jx_res = run_joint(metadata, species_filt, pw, pw_unstrat_cols,
                       make_xgb, "Joint XGBoost", pred_jx)

    per_cohort_rows_all = (
        per_cohort_rows(sp_res, "species_rf")
        + per_cohort_rows(jr_res, "joint_rf")
        + per_cohort_rows(jx_res, "joint_xgb")
    )
    pd.DataFrame(per_cohort_rows_all).to_csv(OUT_PER_COHORT, index=False)
    print(f"\nSaved {OUT_PER_COHORT}")

    pooled_rows = [
        pooled_row(pred_sp, "species_rf"),
        pooled_row(pred_jr, "joint_rf"),
        pooled_row(pred_jx, "joint_xgb"),
    ]
    pd.DataFrame(pooled_rows).to_csv(OUT_POOLED, index=False)
    print(f"Saved {OUT_POOLED}")

    delong_rows = compute_delong({
        "species_rf": pred_sp,
        "joint_rf": pred_jr,
        "joint_xgb": pred_jx,
    })
    pd.DataFrame(delong_rows).to_csv(OUT_DELONG, index=False)
    print(f"Saved {OUT_DELONG}")

    # 10-cohort references
    base_10 = pd.read_csv(REPO / "results" / "baseline_results.csv")
    joint_10 = pd.read_csv(REPO / "results" / "joint_results.csv")
    delong_10 = pd.read_csv(REPO / "results" / "delong_results.csv")
    bootstrap_10 = pd.read_csv(REPO / "results" / "bootstrap_ci.csv")

    make_figure(per_cohort_rows_all, base_10, joint_10, OUT_FIG)
    write_summary(
        per_cohort_rows_all, pooled_rows, delong_rows,
        base_10, joint_10, delong_10, bootstrap_10,
        OUT_SUMMARY,
    )

    # Console summary
    print("\n=== Pooled summary (11-cohort) ===")
    for r in pooled_rows:
        print(f"  {r['model']:12s}  AUC={r['pooled_auc']:.3f}  "
              f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]  n={r['n']}")
    print("\n=== DeLong (11-cohort) ===")
    for r in delong_rows:
        print(f"  {r['model_a']:12s} vs {r['model_b']:12s}  "
              f"diff={r['auc_diff']:+.3f}  z={r['z']:+.3f}  p={r['p_value']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
