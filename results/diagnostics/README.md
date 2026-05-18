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

Three narrative roll-ups summarise the diagnostics produced by these
scripts:

- `ROBUSTNESS_SUMMARY.md` — TreeSHAP-bias, sequencing-depth, and
  calibration-mechanism cross-checks.
- `CLINICAL_TRANSLATION_SUMMARY.md` — sensitivity at fixed specificity,
  base-rate PPV, and head-to-head with FIT.
- `RAW_DATA_PATTERNS.md` — pre-modelling exploratory patterns
  (cohort composition, PCoA, alpha diversity, top species, depth).

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

### 2. Calibration mechanism — `brier_decomposition.csv`, `calibration_mechanism.png`

What it shows: Murphy (1973) decomposition of the Brier score into
reliability − resolution + uncertainty for each of the three models.

How it supports the paper: The joint-XGB calibration gap is driven
almost entirely by the reliability term (≈0.026 vs ≈0.006-0.007 for the
two RF models), confirming that aggressive logit pushes — not a loss of
discriminative information — explain the worse ECE. This contextualises
the RF-vs-XGB recommendation in the Discussion.

### 3. Confusion matrices — `confusion_matrices.csv`, `confusion_matrices.png`

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

### 4. Per-cohort operating characteristics — `per_cohort_operating_chars.csv`, `per_cohort_sens_spec.png`

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

### 5. ROC and Precision-Recall — `auprc_pooled.csv`, `roc_pr_pooled.png`

What it shows: Pooled ROC and PR curves for all three models, with
AUROC and AUPRC annotated. The PR panel includes the positive-class
prevalence as a baseline.

What to look for: PR curves are more sensitive to class imbalance than
ROC curves; agreement between AUROC and AUPRC rankings strengthens the
model comparison.

How it supports the paper: Pairs naturally with the bootstrap CIs in
Supplementary Table S4 and the DeLong tests in S10 — same pooled
predictions, alternative summaries.

### 6. Sensitivity at fixed specificity — `sens_at_fixed_spec.csv`, `sens_at_fixed_spec.png`

What it shows: For each of the three models, the sensitivity achieved
at the two clinically conventional specificity floors of 0.90 and 0.95,
together with PPV / NPV anchored at the US lifetime CRC prevalence of
5% and the implied number-needed-to-test = 1 / PPV.

How it supports the paper: Directly answers "what fraction of CRC
cases would this catch at FIT-like specificity?" — the headline figure
is `species RF sensitivity = 49.9%` at spec 0.90 and `39.8%` at spec
0.95. Used in the Discussion FIT-comparison paragraph and as input to
the `fit_comparison.py` table.

### 7. Base-rate PPV / NPV — `base_rate_ppv.csv`, `base_rate_ppv.png`

What it shows: PPV and NPV for the species RF (Youden-J operating
point) across a 12-step CRC-prevalence sweep from 0.5% to 50%, using
Bayes' rule with the pooled-LODO sensitivity and specificity held
fixed.

How it supports the paper: Anchors the central translational caveat:
at population CRC prevalence (≈5%), PPV is only 11.4% even though the
balanced-class operating point looks favourable. This is the empirical
basis for the "stratifier, not standalone screen" framing in the
Discussion.

### 8. FIT comparison — `fit_vs_microbiome.csv`

What it shows: Head-to-head table comparing the species RF (at
FIT-matched specificity floors of 0.94 and 0.96) against published FIT
performance for CRC and the advanced-adenoma sub-endpoint (Imperiale
2014, NEJM 370:1287).

How it supports the paper: Anchors the Discussion paragraph on where
microbiome screening sits relative to FIT and provides the
specificity-matched sensitivity gap.

### 9. Permutation importance — `permutation_importance_species_rf.csv`, `permutation_vs_shap_correlation.csv`, `figures/diagnostics/permutation_vs_shap.png`

What it shows: Permutation importance (`n_repeats=30`,
`scoring='roc_auc'`) on the species RF, alongside a paired-rank
comparison with the TreeSHAP rankings from `results/shap_crc_features.csv`.

How it supports the paper: Addresses the standard critique that
TreeSHAP can be biased toward high-cardinality features. 16 of the top
20 species by TreeSHAP also rank in the top 20 by permutation; three of
the four oral pathobionts hold top-4 rank under both measures.

### 10. Depth confounding — `depth_confound_shap.csv`, `figures/diagnostics/depth_vs_fnucleatum_shap.png`

What it shows: Spearman correlation between per-cohort SHAP rank and
per-cohort median sequencing depth for the top-20 species, with a
focused scatter for *Fusobacterium nucleatum*.

How it supports the paper: Rules out the alternative explanation that
the oral-pathobiont signal is an artifact of deeper-sequenced cohorts
detecting rare taxa more often (no top-20 species survives a
Bonferroni-style threshold; *F. nucleatum* itself has ρ = −0.19,
p = 0.59).

### 11. Subgroup AUC — `subgroup_auc.csv`, `subgroup_auc.png`

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

### 12. Raw-data exploration — `raw_data_summary.csv`, `RAW_DATA_PATTERNS.md`, `figures/diagnostics/{cohort_composition,pcoa_bray_curtis,alpha_diversity,top_species_heatmap,depth_distribution}.png`

What it shows: Five pre-modelling exploratory plots and a per-cohort
summary table (n, condition breakdown, median Shannon diversity,
median sequencing depth, percent zero features) computed before any
classifier fitting.

How it supports the paper: Justifies the country-aware LODO and
per-fold filtering design empirically (cohort structure dominates
condition structure on PC1/PC2 of Bray-Curtis), and motivates the
log-transform / tree-based modelling choice (top-species abundance
varies by 1-2 orders of magnitude across cohorts).

## Reproducibility

Each diagnostic is produced by a standalone, idempotent script in
`scripts/diagnostics/`:

| Diagnostic | Script | Outputs (under `results/diagnostics/` and `figures/diagnostics/`) |
|---|---|---|
| Calibration | `scripts/diagnostics/calibration.py` | `calibration_metrics.csv`, `calibration_reliability.png` |
| Calibration mechanism | `scripts/diagnostics/calibration_mechanism.py` | `brier_decomposition.csv`, `calibration_mechanism.png` |
| Confusion matrices | `scripts/diagnostics/confusion_matrices.py` | `confusion_matrices.csv`, `confusion_matrices.png` |
| Per-cohort operating chars | `scripts/diagnostics/per_cohort_sens_spec.py` | `per_cohort_operating_chars.csv`, `per_cohort_sens_spec.png` |
| ROC / PR | `scripts/diagnostics/roc_pr_curves.py` | `auprc_pooled.csv`, `roc_pr_pooled.png` |
| Sensitivity at fixed spec | `scripts/diagnostics/sens_at_fixed_specificity.py` | `sens_at_fixed_spec.csv`, `sens_at_fixed_spec.png` |
| Base-rate PPV / NPV | `scripts/diagnostics/base_rate_ppv.py` | `base_rate_ppv.csv`, `base_rate_ppv.png` |
| FIT comparison | `scripts/diagnostics/fit_comparison.py` | `fit_vs_microbiome.csv` |
| Permutation importance | `scripts/diagnostics/permutation_importance.py` | `permutation_importance_species_rf.csv`, `permutation_vs_shap_correlation.csv`, `permutation_vs_shap.png` |
| Depth confounding | `scripts/diagnostics/depth_confound_check.py` | `depth_confound_shap.csv`, `depth_vs_fnucleatum_shap.png` |
| Subgroup AUC | `scripts/diagnostics/subgroup_analysis.py` | `subgroup_auc.csv`, `subgroup_auc.png` |
| Raw-data exploration | `scripts/diagnostics/raw_data_exploration.py` | `raw_data_summary.csv`, `RAW_DATA_PATTERNS.md`, `{cohort_composition,pcoa_bray_curtis,alpha_diversity,top_species_heatmap,depth_distribution}.png` |

Run from repo root:

```bash
python3 scripts/diagnostics/calibration.py
python3 scripts/diagnostics/calibration_mechanism.py
python3 scripts/diagnostics/confusion_matrices.py
python3 scripts/diagnostics/per_cohort_sens_spec.py
python3 scripts/diagnostics/roc_pr_curves.py
python3 scripts/diagnostics/sens_at_fixed_specificity.py
python3 scripts/diagnostics/base_rate_ppv.py
python3 scripts/diagnostics/fit_comparison.py
python3 scripts/diagnostics/permutation_importance.py
python3 scripts/diagnostics/depth_confound_check.py
python3 scripts/diagnostics/subgroup_analysis.py
python3 scripts/diagnostics/raw_data_exploration.py
```

Inputs are read-only: the three `results/preds_*.csv` files,
`data/processed/metadata_clean.csv`, `data/processed/species_filtered.csv`,
and (for permutation_importance / depth_confound_check)
`results/shap_crc_features.csv`. No existing files in `results/`,
`figures/`, or `scripts/` are modified. Outputs are written to
`results/diagnostics/` and `figures/diagnostics/`.
