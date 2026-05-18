# Supplementary Diagnostics

Standalone diagnostics that supplement the main results in `results/` and
the formal supplementary tables in `results/supplementary/` (see
`results/supplementary/INDEX.csv`). Every diagnostic is derived from the
three pooled LODO prediction files already in `results/`:

- `preds_species_rf.csv` — headline species-only random forest
- `preds_joint_rf.csv` — joint (species + pathway) random forest
- `preds_joint_xgb.csv` — joint XGBoost

All numbers shown are pooled across the 10 cohorts (n = 1339). The
species RF model is the headline model and is the one used for the
per-cohort and subgroup analyses.

## Files

### 1. Calibration — `calibration_metrics.csv`, `calibration_reliability.png`

What it shows: For each model, the Brier score (mean squared error of
the predicted probability) and the Expected Calibration Error (ECE,
10 equal-width bins) on pooled LODO predictions. The figure shows
reliability curves with a per-panel histogram of predicted probabilities.

What to look for: A perfectly calibrated model lies on the diagonal. A
curve consistently above the diagonal is under-confident; below the
diagonal, over-confident. Brier rewards both calibration and resolution;
ECE isolates the calibration gap.

How it supports the paper: Reliability quantification complements the
AUC-based claims (Supplementary Table S2, S4): a model with good AUC but
poor calibration would still be unfit for clinical decision support
without recalibration. The RF models (species and joint) are noticeably
better calibrated than joint XGB, which over-confidently splits scores
toward the extremes — a useful caveat to the headline AUC comparison.

### 2. Confusion matrices — `confusion_matrices.csv`, `confusion_matrices.png`

What it shows: Counts (TP, FP, TN, FN) and derived operating
characteristics (sensitivity, specificity, PPV, NPV, F1, MCC) for each
model at two thresholds: the pooled Youden-J optimum and a fixed 0.5.

What to look for: Youden-J chooses the threshold that maximizes
TPR-FPR — a natural single-number operating point for an unbiased
screening tool. The 0.5 row exists to anchor against the model's raw
score scale. MCC is a balanced single-number summary robust to
class imbalance.

How it supports the paper: Provides explicit clinical-style operating
points to accompany the threshold-free AUC comparisons in S2 and S4.

### 3. Per-cohort operating characteristics — `per_cohort_operating_chars.csv`, `per_cohort_sens_spec.png`

What it shows: Sensitivity, specificity, PPV, NPV of the species RF
model in each of the 10 LODO cohorts, evaluated at a single pooled
Youden-J threshold (so cohorts are directly comparable). The pooled row
appears first in the CSV. Dashed lines on the figure mark the pooled
values.

What to look for: Whether the headline operating point generalizes
across cohorts. Strong heterogeneity (e.g. cohorts where specificity
collapses) would caution against a single fixed threshold.

How it supports the paper: A per-cohort companion to the per-fold AUC
analysis (Supplementary Table S2). Demonstrates that the gains shown in
S2 / S4 translate into useful operating characteristics in nearly every
cohort, while flagging the cohorts (e.g., ThomasAM_2018a) where
performance is weakest.

### 4. ROC and Precision-Recall — `auprc_pooled.csv`, `roc_pr_pooled.png`

What it shows: Pooled ROC and PR curves for all three models, with
AUROC and AUPRC annotated. The PR panel includes the positive-class
prevalence as a baseline.

What to look for: PR curves are more sensitive to class imbalance than
ROC curves; agreement between AUROC and AUPRC rankings strengthens the
model comparison.

How it supports the paper: Pairs naturally with the bootstrap CIs in
Supplementary Table S4 and the DeLong tests in S10 — same pooled
predictions, alternative summaries.

### 5. Subgroup AUC — `subgroup_auc.csv`, `subgroup_auc.png`

What it shows: Species RF AUC stratified by age band (<50, 50-65, >65),
sex (female, male), and BMI category (<25, 25-30, >30). Each subgroup
has a 95% bootstrap CI (1000 iterations). Samples with missing values
for a stratifier are dropped from that variable's analysis only; n per
subgroup is reported.

What to look for: Overlapping CIs across levels of a variable indicate
that performance is comparable across that demographic; non-overlap
flags a subgroup where the model behaves differently.

How it supports the paper: A direct readout of fairness/robustness
across the same demographic variables analyzed in the confounder
adjustment study (Supplementary Table S7). Where S7 asks "do these
covariates confound the species signal", this diagnostic asks "does the
model perform comparably within each subgroup".

## Reproducibility

Each diagnostic is produced by a standalone, idempotent script in
`scripts/diagnostics/`:

| Diagnostic | Script |
|---|---|
| Calibration | `scripts/diagnostics/calibration.py` |
| Confusion matrices | `scripts/diagnostics/confusion_matrices.py` |
| Per-cohort operating chars | `scripts/diagnostics/per_cohort_sens_spec.py` |
| ROC / PR | `scripts/diagnostics/roc_pr_curves.py` |
| Subgroup AUC | `scripts/diagnostics/subgroup_analysis.py` |

Run from repo root:

```bash
python3 scripts/diagnostics/calibration.py
python3 scripts/diagnostics/confusion_matrices.py
python3 scripts/diagnostics/per_cohort_sens_spec.py
python3 scripts/diagnostics/roc_pr_curves.py
python3 scripts/diagnostics/subgroup_analysis.py
```

Inputs are read-only: the three `results/preds_*.csv` files and
`data/processed/metadata_clean.csv`. No existing files in `results/`,
`figures/`, or `scripts/` are modified. Outputs are written to
`results/diagnostics/` and `figures/diagnostics/`.
