"""Cross-disease specificity test for the species RF CRC classifier.

Goal: pre-empt the "your CRC signature is just generic gut dysbiosis"
reviewer concern. We train the species-only Random Forest on the local
CRC vs. control dataset (the same model used elsewhere in the pipeline)
and then score samples from *non-CRC* disease cohorts that were never
seen during training. If the classifier scores IBD / T2D / cirrhosis
samples just as high as CRC, the signature is generic dysbiosis. If
non-CRC disease samples score near the control distribution, the
signature is CRC-specific.

Procedure
---------
1. Train species RF on data/processed/species_filtered.csv +
   metadata_clean.csv (CRC=1 vs control=0; adenoma dropped). RF config
   matches scripts/train_baseline.py exactly:
     n_estimators=500, max_features='sqrt', min_samples_leaf=5,
     class_weight='balanced', random_state=42.
2. Pull non-CRC disease cohorts from curatedMetagenomicData via an R
   helper (Rscript subprocess). Cohorts:
       NielsenHB_2014        IBD (Denmark/Spain)
       HMP_2019_ibdmdb       IBD (USA, IBDMDB)
       KarlssonFH_2013       T2D + IGT  (Sweden)
       QinJ_2012             T2D        (China)
       QinN_2014             cirrhosis  (China)
       LeChatelierE_2013     obesity proxy (BMI>=30 vs <25, France)
       HMP_2012              healthy baseline (USA)
   Per-cohort species tables (relative abundance) are written to
   data/external_disease_cache/<cohort>__species.csv and the matching
   per-sample metadata to data/external_disease_cache/<cohort>__meta.csv
   so the R step is one-shot (~few minutes); subsequent runs reuse the
   cache.
3. Align each cohort's species table to the 229-feature panel used by
   the trained RF: missing features filled with 0, extra features
   dropped, then the same preprocessing as preprocessing.py is applied
   (row-sum renormalisation when needed, then log10(x + 1e-6)). This
   is the same pipeline the model saw during fitting, so probabilities
   are comparable.
4. Predict CRC probability for every sample. Compare against the
   Youden-J optimal threshold (species_rf: 0.4847 from
   results/diagnostics/confusion_matrices.csv).
5. Report per-cohort distribution stats and an interpretation label
   (CRC-specific / intermediate / generic-dysbiosis-like) using
   bright-line cutoffs on the fraction exceeding the Youden threshold.
6. Build a boxplot grouping cohorts as
       Training CRC   |   Training control   |   non-CRC disease
       (one box per cohort or per disease label).

Outputs
-------
- results/diagnostics/cross_disease_specificity.csv
- results/diagnostics/cross_disease_predictions.csv (per-sample probs)
- results/diagnostics/SPECIFICITY_NOTE.md
- figures/diagnostics/cross_disease_specificity.png

Failure handling
----------------
If Rscript / curatedMetagenomicData is not available in the current
environment, the script still trains the model and writes a CSV with
n_samples=0 for each external cohort plus an explanatory NOTE; it does
not crash. The R bridge can be re-run on a machine where cMD is
installed and the rest of the pipeline picks up the cache.

Usage
-----
    python3 scripts/diagnostics/cross_disease_specificity.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
SPECIES_PATH = ROOT / "data" / "processed" / "species_filtered.csv"
META_PATH = ROOT / "data" / "processed" / "metadata_clean.csv"
CACHE_DIR = ROOT / "data" / "external_disease_cache"
CONFUSION_PATH = ROOT / "results" / "diagnostics" / "confusion_matrices.csv"

OUT_CSV = ROOT / "results" / "diagnostics" / "cross_disease_specificity.csv"
OUT_PREDS = ROOT / "results" / "diagnostics" / "cross_disease_predictions.csv"
OUT_MD = ROOT / "results" / "diagnostics" / "SPECIFICITY_NOTE.md"
OUT_FIG = ROOT / "figures" / "diagnostics" / "cross_disease_specificity.png"

# ── Cohort plan ───────────────────────────────────────────────────────────────
# Each entry: study_name, list of (group_label, condition_filter_fn_for_R) pairs
# describing the disease groups we want to score, plus a primary disease label
# used for plotting / interpretation.
@dataclass(frozen=True)
class CohortSpec:
    study_name: str
    disease_label: str          # display label, e.g. "IBD"
    case_conditions: tuple      # study_condition values that count as cases
    control_conditions: tuple   # values counted as that cohort's own controls
    note: str                   # short note for the CSV

COHORT_PLAN = [
    CohortSpec("NielsenHB_2014",   "IBD",       ("IBD",),         ("control",), "Denmark/Spain IBD cohort"),
    CohortSpec("HMP_2019_ibdmdb",  "IBD",       ("IBD",),         ("control",), "IBDMDB longitudinal cohort (first sample per subject only)"),
    CohortSpec("KarlssonFH_2013",  "T2D",       ("T2D", "IGT"),   ("control",), "Swedish T2D + IGT cohort"),
    CohortSpec("QinJ_2012",        "T2D",       ("T2D",),         ("control",), "Chinese T2D cohort"),
    CohortSpec("QinN_2014",        "cirrhosis", ("cirrhosis",),   ("control",), "Chinese cirrhosis cohort"),
    CohortSpec("LeChatelierE_2013","obesity",   ("control",),     ("control",), "BMI-based split (>=30 obese vs <25 lean)"),
    CohortSpec("HMP_2012",         "healthy",   (),               ("control",), "USA healthy baseline (HMP1)"),
]

# Interpretation buckets on (% predicted CRC at Youden threshold) MINUS
# (% in training-control at the same threshold). A negative or near-zero
# excess is the CRC-specific case; values approaching the training-CRC
# rate are generic-dysbiosis-like.
INTERPRETATION_BUCKETS = [
    (-1.00, 0.10, "CRC-specific (non-CRC disease scores near control)"),
    (0.10, 0.30, "intermediate (some shared signal)"),
    (0.30, 1.00, "generic-dysbiosis-like (scores comparable to CRC)"),
]


def interpret(frac_above: float, control_frac_above: float, crc_frac_above: float) -> str:
    """Label a cohort given fraction above Youden threshold."""
    crc_gap = crc_frac_above - control_frac_above
    if crc_gap <= 0:
        # Pathological training metric — fall back to absolute thresholds.
        return "indeterminate"
    relative = (frac_above - control_frac_above) / crc_gap
    for lo, hi, label in INTERPRETATION_BUCKETS:
        if lo <= relative < hi:
            return label
    return INTERPRETATION_BUCKETS[-1][2]


# ── R bridge: pull species tables from curatedMetagenomicData ────────────────
R_FETCH_SCRIPT = r"""
suppressMessages(library(curatedMetagenomicData))
suppressMessages(library(SummarizedExperiment))

args <- commandArgs(trailingOnly = TRUE)
plan_json <- args[1]
out_dir   <- args[2]

if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

plan <- jsonlite::fromJSON(plan_json, simplifyVector = FALSE)
data("sampleMetadata")
sm <- sampleMetadata

for (entry in plan) {
  study <- entry$study_name
  cat("[R] fetching", study, "\n")
  sub <- sm[sm$study_name == study, ]
  if (nrow(sub) == 0) {
    cat("[R]   no samples found for", study, "\n")
    next
  }
  # Restrict to stool samples wherever a body_site column exists — the model
  # was trained on stool only, and HMP_2012 in particular mixes oral / nasal /
  # vaginal / skin samples that have radically different microbiomes.
  if ("body_site" %in% colnames(sub)) {
    n_before <- nrow(sub)
    sub <- sub[!is.na(sub$body_site) & sub$body_site == "stool", ]
    if (nrow(sub) < n_before) {
      cat("[R]   body_site=stool filter:", n_before, "->", nrow(sub), "samples\n")
    }
  }
  # For longitudinal IBDMDB, keep one sample per subject (earliest collection
  # by days_from_first_collection, treating NA as 0) so samples are independent.
  if (study == "HMP_2019_ibdmdb" && "subject_id" %in% colnames(sub) &&
      "days_from_first_collection" %in% colnames(sub)) {
    d <- sub$days_from_first_collection
    d[is.na(d)] <- 0
    sub$.dfc <- d
    sub <- sub[order(sub$subject_id, sub$.dfc), ]
    sub <- sub[!duplicated(sub$subject_id), ]
    sub$.dfc <- NULL
    cat("[R]   IBDMDB one-per-subject filter ->", nrow(sub), "samples\n")
  }
  # Write metadata
  meta_path <- file.path(out_dir, paste0(study, "__meta.csv"))
  meta_cols <- intersect(c("sample_id","subject_id","study_name","study_condition",
                           "disease","age","gender","BMI","country",
                           "sequencing_platform","number_reads"),
                         colnames(sub))
  write.csv(sub[, meta_cols, drop=FALSE], meta_path, row.names = FALSE)

  se <- tryCatch(
    returnSamples(sub, "relative_abundance"),
    error = function(e) { cat("[R]   returnSamples failed:", conditionMessage(e), "\n"); NULL }
  )
  if (is.null(se)) next

  mat <- t(SummarizedExperiment::assay(se))
  df <- as.data.frame(mat, check.names = FALSE)
  df$sample_id <- rownames(df)
  # Reorder so sample_id is first
  df <- df[, c("sample_id", setdiff(colnames(df), "sample_id"))]
  sp_path <- file.path(out_dir, paste0(study, "__species.csv"))
  write.csv(df, sp_path, row.names = FALSE)
  cat("[R]   wrote", sp_path, "(", nrow(df), "samples,", ncol(df)-1, "species)\n")
}
cat("[R] done\n")
"""


def have_cached(study: str) -> bool:
    return (CACHE_DIR / f"{study}__species.csv").exists() and \
           (CACHE_DIR / f"{study}__meta.csv").exists()


def fetch_external_cohorts_via_R(force: bool = False) -> dict:
    """Populate CACHE_DIR with one species csv + meta csv per cohort.

    Returns a dict mapping study_name -> 'cached'|'fetched'|'unavailable'.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    status = {}
    plan = [{"study_name": c.study_name} for c in COHORT_PLAN
            if force or not have_cached(c.study_name)]
    if not plan:
        return {c.study_name: "cached" for c in COHORT_PLAN}

    # Write R script to a tempfile so we don't need shell quoting acrobatics.
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False) as fh:
        fh.write(R_FETCH_SCRIPT)
        script_path = fh.name
    plan_path = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False).name
    with open(plan_path, "w") as fh:
        json.dump(plan, fh)

    try:
        proc = subprocess.run(
            ["Rscript", script_path, plan_path, str(CACHE_DIR)],
            capture_output=True, text=True, timeout=1800,
        )
        print(proc.stdout)
        if proc.returncode != 0:
            print("[fetch] Rscript failed (rc=%d). STDERR tail:" % proc.returncode)
            print(proc.stderr[-1500:])
    except FileNotFoundError:
        print("[fetch] Rscript not found in PATH; skipping external fetch.")
    except subprocess.TimeoutExpired:
        print("[fetch] Rscript timed out; partial results may exist.")
    finally:
        try:
            os.unlink(script_path)
            os.unlink(plan_path)
        except OSError:
            pass

    for c in COHORT_PLAN:
        if have_cached(c.study_name):
            status[c.study_name] = "cached"
        else:
            status[c.study_name] = "unavailable"
    return status


# ── Feature alignment ────────────────────────────────────────────────────────
def align_to_panel(raw_species: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Reproject a raw relative-abundance table onto the trained feature panel.

    Applies the same preprocessing as scripts/preprocessing.py: optional
    row-sum normalisation when values look like percentages (sum >> 1) and
    log10(x + 1e-6).
    """
    sid = raw_species["sample_id"].copy()
    X = raw_species.drop(columns=["sample_id"], errors="ignore").copy()
    # Reindex: missing features get 0 (species absent in this cohort),
    # extra features dropped.
    X = X.reindex(columns=feature_cols, fill_value=0.0)
    X = X.astype(float).fillna(0.0)
    # Row-sum normalisation if the matrix looks like percentages.
    rs = X.sum(axis=1)
    if rs.mean() > 1.5:
        # Avoid division by zero
        rs_safe = rs.replace(0, np.nan)
        X = X.div(rs_safe, axis=0).fillna(0.0)
    X = np.log10(X + 1e-6)
    X.insert(0, "sample_id", sid.values)
    return X


# ── Helpers ──────────────────────────────────────────────────────────────────
def read_youden_threshold() -> float:
    if not CONFUSION_PATH.exists():
        return 0.4847  # documented headline value
    cm = pd.read_csv(CONFUSION_PATH)
    row = cm[(cm["model"] == "species_rf") & (cm["threshold_type"] == "youden_j")]
    if row.empty:
        return 0.4847
    return float(row.iloc[0]["threshold"])


def train_species_rf():
    sp = pd.read_csv(SPECIES_PATH)
    md = pd.read_csv(META_PATH)
    mg = md.merge(sp, on="sample_id", how="inner")
    feat_cols = [c for c in sp.columns if c != "sample_id"]
    mask = mg["label"].isin([0, 1])
    X = mg.loc[mask, feat_cols].reset_index(drop=True)
    y = mg.loc[mask, "label"].reset_index(drop=True).astype(int)
    meta_train = mg.loc[mask, ["sample_id", "study_name", "study_condition", "label"]].reset_index(drop=True)
    print(f"[train] {len(X)} samples (CRC={int(y.sum())}, control={int((y == 0).sum())}), "
          f"{X.shape[1]} features")
    model = RandomForestClassifier(
        n_estimators=500, max_features="sqrt", min_samples_leaf=5,
        n_jobs=-1, random_state=42, class_weight="balanced",
    )
    model.fit(X, y)
    # In-sample training probabilities (used only to anchor the plot's
    # 'training CRC' / 'training control' boxes; the headline LODO probs
    # live in results/preds_species_rf.csv and are also overlaid).
    train_prob = model.predict_proba(X)[:, 1]
    meta_train["y_prob"] = train_prob
    return model, feat_cols, meta_train


def load_lodo_predictions() -> pd.DataFrame | None:
    p = ROOT / "results" / "preds_species_rf.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    # df columns: sample_id, cohort, y_true, y_prob
    return df


# ── Per-cohort scoring ───────────────────────────────────────────────────────
def score_cohort(spec: CohortSpec, model, feat_cols, youden: float) -> pd.DataFrame:
    """Return a per-sample DataFrame with predicted CRC probability."""
    sp_path = CACHE_DIR / f"{spec.study_name}__species.csv"
    md_path = CACHE_DIR / f"{spec.study_name}__meta.csv"
    if not sp_path.exists() or not md_path.exists():
        return pd.DataFrame(columns=["sample_id", "study_name", "group", "y_prob"])

    species = pd.read_csv(sp_path)
    meta = pd.read_csv(md_path)

    # Bucket samples into "case" / "control" groups within this cohort.
    if spec.disease_label == "obesity":
        # BMI-based grouping for LeChatelierE_2013.
        meta = meta[meta["BMI"].notna()].copy()
        meta["group"] = np.where(meta["BMI"] >= 30, "obesity",
                          np.where(meta["BMI"] < 25, "lean_control", "overweight"))
        meta = meta[meta["group"].isin(["obesity", "lean_control"])]
    elif spec.disease_label == "healthy":
        meta = meta.copy()
        meta["group"] = "healthy"
    else:
        meta = meta.copy()
        def _grp(cond):
            if cond in spec.case_conditions:
                return spec.disease_label
            if cond in spec.control_conditions:
                return f"{spec.study_name}_control"
            return None
        meta["group"] = meta["study_condition"].map(_grp)
        meta = meta[meta["group"].notna()]

    species = species[species["sample_id"].isin(meta["sample_id"])]
    if species.empty:
        return pd.DataFrame(columns=["sample_id", "study_name", "group", "y_prob"])

    aligned = align_to_panel(species, feat_cols)
    X = aligned.drop(columns=["sample_id"]).values
    probs = model.predict_proba(X)[:, 1]
    out = pd.DataFrame({
        "sample_id": aligned["sample_id"].values,
        "study_name": spec.study_name,
        "y_prob": probs,
    })
    out = out.merge(meta[["sample_id", "group"]], on="sample_id", how="left")
    out = out[["sample_id", "study_name", "group", "y_prob"]]
    return out


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-fetch", action="store_true",
                        help="Re-download external cohorts even if cached.")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Do not call Rscript; use whatever is already cached.")
    args = parser.parse_args()

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Train the RF.
    model, feat_cols, meta_train = train_species_rf()

    # 2. Pull external cohorts via R unless told to skip.
    if args.skip_fetch:
        status = {c.study_name: ("cached" if have_cached(c.study_name) else "unavailable")
                  for c in COHORT_PLAN}
    else:
        status = fetch_external_cohorts_via_R(force=args.force_fetch)
    for k, v in status.items():
        print(f"[cohort] {k:22s}  {v}")

    # 3. Read Youden threshold from existing diagnostics.
    youden = read_youden_threshold()
    print(f"[thresh] species_rf Youden-J = {youden:.4f}")

    # 4. Score each cohort.
    per_sample = []
    for spec in COHORT_PLAN:
        df = score_cohort(spec, model, feat_cols, youden)
        if df.empty:
            print(f"[score] {spec.study_name:22s} -> no samples scored")
        else:
            print(f"[score] {spec.study_name:22s} -> {len(df)} samples scored "
                  f"(groups: {sorted(df['group'].unique())})")
        per_sample.append(df)
    ext_preds = pd.concat(per_sample, ignore_index=True) if per_sample else pd.DataFrame()

    # 5. Build the headline summary table.
    # Training-control / training-CRC baselines via the held-out LODO file
    # (preferred, no in-sample optimism). If unavailable, fall back to the
    # in-sample probabilities from the freshly trained model.
    lodo = load_lodo_predictions()
    if lodo is not None and {"y_true", "y_prob"}.issubset(lodo.columns):
        train_crc_probs  = lodo.loc[lodo["y_true"] == 1, "y_prob"].values
        train_ctrl_probs = lodo.loc[lodo["y_true"] == 0, "y_prob"].values
        baseline_source = "LODO held-out (results/preds_species_rf.csv)"
    else:
        train_crc_probs  = meta_train.loc[meta_train["label"] == 1, "y_prob"].values
        train_ctrl_probs = meta_train.loc[meta_train["label"] == 0, "y_prob"].values
        baseline_source = "in-sample training (LODO predictions unavailable)"

    ctrl_frac_above = float((train_ctrl_probs >= youden).mean()) if len(train_ctrl_probs) else float("nan")
    crc_frac_above  = float((train_crc_probs  >= youden).mean()) if len(train_crc_probs)  else float("nan")

    summary_rows = []

    # Training baselines as the top rows of the table (for context).
    if len(train_crc_probs):
        summary_rows.append({
            "disease_cohort": "training_CRC", "study_name": "10-cohort_pool",
            "group": "CRC", "n_samples": int(len(train_crc_probs)),
            "mean_crc_prob": float(np.mean(train_crc_probs)),
            "median_crc_prob": float(np.median(train_crc_probs)),
            "pct_above_youden_threshold": 100.0 * crc_frac_above,
            "interpretation_label": "reference (positives)",
            "source": baseline_source,
        })
    if len(train_ctrl_probs):
        summary_rows.append({
            "disease_cohort": "training_control", "study_name": "10-cohort_pool",
            "group": "control", "n_samples": int(len(train_ctrl_probs)),
            "mean_crc_prob": float(np.mean(train_ctrl_probs)),
            "median_crc_prob": float(np.median(train_ctrl_probs)),
            "pct_above_youden_threshold": 100.0 * ctrl_frac_above,
            "interpretation_label": "reference (negatives)",
            "source": baseline_source,
        })

    # External cohorts.
    for spec in COHORT_PLAN:
        sub = ext_preds[ext_preds["study_name"] == spec.study_name]
        if sub.empty:
            summary_rows.append({
                "disease_cohort": spec.disease_label, "study_name": spec.study_name,
                "group": "(unavailable)", "n_samples": 0,
                "mean_crc_prob": float("nan"), "median_crc_prob": float("nan"),
                "pct_above_youden_threshold": float("nan"),
                "interpretation_label": "data unavailable in this environment",
                "source": spec.note,
            })
            continue
        for group, grp_df in sub.groupby("group"):
            frac_above = float((grp_df["y_prob"] >= youden).mean())
            label = interpret(frac_above, ctrl_frac_above, crc_frac_above) \
                if group in (spec.disease_label,) or group == "obesity" else "cohort-internal control"
            summary_rows.append({
                "disease_cohort": spec.disease_label,
                "study_name": spec.study_name,
                "group": group,
                "n_samples": int(len(grp_df)),
                "mean_crc_prob": float(grp_df["y_prob"].mean()),
                "median_crc_prob": float(grp_df["y_prob"].median()),
                "pct_above_youden_threshold": 100.0 * frac_above,
                "interpretation_label": label,
                "source": spec.note,
            })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_CSV, index=False)
    print(f"[write] {OUT_CSV}")

    if not ext_preds.empty:
        ext_preds.to_csv(OUT_PREDS, index=False)
        print(f"[write] {OUT_PREDS} ({len(ext_preds)} per-sample rows)")

    # 6. Figure: boxplot of CRC probability per cohort/group.
    make_figure(train_crc_probs, train_ctrl_probs, ext_preds, youden)

    # 7. Narrative note.
    write_specificity_note(summary, youden, baseline_source)


def make_figure(train_crc, train_ctrl, ext_preds, youden):
    boxes = []
    labels = []
    colors = []
    PAL = {
        "training_CRC": "#c43c39",
        "training_control": "#7393b3",
        "IBD": "#e58e26",
        "T2D": "#9b59b6",
        "cirrhosis": "#16a085",
        "obesity": "#d35400",
        "healthy": "#27ae60",
        "internal_control": "#bdc3c7",
    }

    if len(train_crc):
        boxes.append(train_crc); labels.append(f"Training CRC\n(n={len(train_crc)})")
        colors.append(PAL["training_CRC"])
    if len(train_ctrl):
        boxes.append(train_ctrl); labels.append(f"Training control\n(n={len(train_ctrl)})")
        colors.append(PAL["training_control"])

    if not ext_preds.empty:
        # Preserve the cohort plan order, and within each cohort plot cases first.
        for spec in COHORT_PLAN:
            sub = ext_preds[ext_preds["study_name"] == spec.study_name]
            if sub.empty:
                continue
            # cases first
            case_groups = []
            ctrl_groups = []
            for group in sub["group"].unique():
                if group in (spec.disease_label, "obesity") or spec.disease_label == "healthy":
                    case_groups.append(group)
                else:
                    ctrl_groups.append(group)
            for group in case_groups + ctrl_groups:
                vals = sub.loc[sub["group"] == group, "y_prob"].values
                if len(vals) == 0:
                    continue
                boxes.append(vals)
                labels.append(f"{spec.study_name}\n{group}\n(n={len(vals)})")
                colors.append(PAL.get(group, PAL.get(spec.disease_label, "#95a5a6")))

    if not boxes:
        print("[figure] No data to plot; writing placeholder.")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No external-cohort scores available\n"
                          "(Rscript / curatedMetagenomicData not run)",
                ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        fig.savefig(OUT_FIG, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return

    fig_w = max(10, 1.0 + 0.85 * len(boxes))
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    positions = np.arange(1, len(boxes) + 1)
    bp = ax.boxplot(
        boxes, positions=positions, widths=0.6, patch_artist=True,
        showmeans=True, meanline=False,
        flierprops=dict(marker=".", markersize=4, alpha=0.5),
        medianprops=dict(color="black", linewidth=1.4),
        meanprops=dict(marker="D", markerfacecolor="white",
                       markeredgecolor="black", markersize=5),
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    # Strip-plot points for small cohorts (n<=120).
    rng = np.random.default_rng(0)
    for pos, vals in zip(positions, boxes):
        if len(vals) <= 120:
            jitter = rng.uniform(-0.18, 0.18, size=len(vals))
            ax.scatter(pos + jitter, vals, s=8, alpha=0.4, color="black", zorder=2)
    ax.axhline(youden, color="red", linestyle="--", linewidth=1.0,
               label=f"Youden-J threshold = {youden:.3f}")
    ax.set_ylabel("Predicted CRC probability (species RF)")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("CRC-classifier scores on non-CRC disease cohorts\n"
                 "(specificity vs. generic-dysbiosis test)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {OUT_FIG}")


def write_specificity_note(summary: pd.DataFrame, youden: float, baseline_source: str):
    # Aggregate non-CRC disease samples (excluding cohort-internal controls
    # and the training-baseline rows we synthesised).
    disease_rows = summary[(summary["disease_cohort"].isin(["IBD", "T2D", "cirrhosis", "obesity"])) &
                           (summary["n_samples"] > 0) &
                           (summary["interpretation_label"] != "cohort-internal control")]
    healthy_rows = summary[(summary["disease_cohort"] == "healthy") & (summary["n_samples"] > 0)]
    train_crc_row = summary[summary["disease_cohort"] == "training_CRC"]
    train_ctrl_row = summary[summary["disease_cohort"] == "training_control"]

    total_disease_n = int(disease_rows["n_samples"].sum())
    if total_disease_n > 0:
        weighted_pct_above = float(
            (disease_rows["pct_above_youden_threshold"] * disease_rows["n_samples"]).sum()
            / total_disease_n
        )
    else:
        weighted_pct_above = float("nan")

    crc_pct = float(train_crc_row["pct_above_youden_threshold"].iloc[0]) if not train_crc_row.empty else float("nan")
    ctrl_pct = float(train_ctrl_row["pct_above_youden_threshold"].iloc[0]) if not train_ctrl_row.empty else float("nan")

    if total_disease_n == 0:
        verdict = ("No non-CRC disease samples could be loaded in this environment, so "
                   "the test could not be executed end-to-end. The script encodes the "
                   "exact procedure and will populate the table when run on a machine "
                   "with curatedMetagenomicData installed.")
    else:
        if np.isfinite(crc_pct) and np.isfinite(ctrl_pct) and crc_pct > ctrl_pct:
            denom = crc_pct - ctrl_pct
            relative = (weighted_pct_above - ctrl_pct) / denom if denom > 0 else float("nan")
            if relative < 0.10:
                verdict = ("supporting that the species-RF CRC signature is "
                           "**CRC-specific** (non-CRC disease samples score near "
                           "the training-control distribution).")
            elif relative < 0.30:
                verdict = ("indicating that the species-RF CRC signature is "
                           "**predominantly CRC-specific but with a modest shared "
                           "dysbiosis component**.")
            else:
                verdict = ("indicating that the species-RF CRC signature carries a "
                           "**substantial shared-dysbiosis component**; readers "
                           "should weigh the headline AUCs against this caveat.")
        else:
            verdict = ("training baselines were degenerate, so the relative-excess "
                       "interpretation could not be computed.")

    # Per-cohort sentences for the table.
    bullets = []
    for _, r in summary.iterrows():
        if r["disease_cohort"] in ("training_CRC", "training_control"):
            continue
        if r["n_samples"] == 0:
            bullets.append(f"- **{r['study_name']}** ({r['disease_cohort']}): data unavailable in this environment.")
            continue
        bullets.append(
            f"- **{r['study_name']}** ({r['group']}, n={int(r['n_samples'])}): "
            f"mean predicted CRC probability {r['mean_crc_prob']:.3f}, "
            f"{r['pct_above_youden_threshold']:.1f}% above Youden threshold "
            f"— *{r['interpretation_label']}*."
        )

    md = (
        f"# Cross-disease specificity of the species RF CRC classifier\n\n"
        f"The headline species-only Random Forest (trained on CRC vs. control across "
        f"10 cohorts; pooled LODO AUC ~0.78) was applied without retraining to samples "
        f"from non-CRC disease cohorts in curatedMetagenomicData. The question: do "
        f"those samples score *like CRC* (indicating a generic gut-dysbiosis signal) "
        f"or *like controls* (indicating a CRC-specific signal)?\n\n"
        f"**Baselines (Youden-J threshold = {youden:.4f}, source: {baseline_source}):** "
        f"training-CRC samples score above threshold {crc_pct:.1f}% of the time; "
        f"training-control samples score above threshold {ctrl_pct:.1f}% of the time.\n\n"
        f"**Headline:** Of {total_disease_n} non-CRC disease samples scored by the species RF, "
        f"{weighted_pct_above:.1f}% exceeded the Youden-J threshold (vs {crc_pct:.1f}% of "
        f"training-CRC samples and {ctrl_pct:.1f}% of training-control samples), {verdict}\n\n"
        f"**Per-cohort breakdown:**\n\n"
        + "\n".join(bullets)
        + "\n\n**Method.** Species RF (500 trees, max_features=sqrt, min_samples_leaf=5, "
        f"class_weight=balanced, random_state=42) was trained on the 1,339-sample binary "
        f"CRC-vs-control set using the same 229-feature panel and log10-relative-abundance "
        f"preprocessing as `scripts/train_baseline.py`. External cohort species tables "
        f"were fetched via `curatedMetagenomicData::returnSamples(..., 'relative_abundance')` "
        f"(MetaPhlAn taxonomy), reindexed onto the trained 229-feature panel (missing "
        f"features filled with 0, extra features dropped), renormalised, and log10-"
        f"transformed identically. Probabilities were thresholded at the species-RF "
        f"Youden-J value from `results/diagnostics/confusion_matrices.csv` "
        f"({youden:.4f}).\n\n"
        f"**Reproducibility.** External cohort tables are cached under "
        f"`data/external_disease_cache/` so subsequent runs of "
        f"`python3 scripts/diagnostics/cross_disease_specificity.py` reuse them without "
        f"re-downloading. Pass `--force-fetch` to refresh.\n"
    )
    OUT_MD.write_text(md)
    print(f"[write] {OUT_MD}")


if __name__ == "__main__":
    main()
