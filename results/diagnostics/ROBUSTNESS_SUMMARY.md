# Robustness diagnostics — summary

Three pre-emptive sensitivity analyses for the species RF model and the
joint-model calibration gap. Each addresses a specific anticipated
reviewer critique.

## 1. TreeSHAP vs permutation importance (TreeSHAP-bias critique)

Refitting the species RF on all 1,339 labeled samples and computing
sklearn `permutation_importance` (n_repeats=30, scoring='roc_auc'),
**16 of the top 20 SHAP-ranked species are also among the top 20 by
permutation importance**. Three of the four oral pathobionts
(G. morbillorum, P. stomatis, F. nucleatum) hold top-4 rank under both
measures; P. micra ranks 2nd by SHAP and 12th by permutation, still
inside the top 20. This rules out the possibility that the SHAP-driven
biological interpretation is an artifact of TreeSHAP's known bias toward
high-cardinality / high-split-opportunity features.

  - `results/diagnostics/permutation_importance_species_rf.csv`
  - `results/diagnostics/permutation_vs_shap_correlation.csv`
  - `figures/diagnostics/permutation_vs_shap.png`

## 2. Sequencing-depth confounding (depth-driven SHAP critique)

Per-cohort TreeSHAP (model refit on the other 9 cohorts, SHAP computed
on the held-out cohort) was regressed against per-cohort median read
depth. **F. nucleatum's per-cohort SHAP rank shows no significant
correlation with cohort sequencing depth (Spearman rho = -0.194,
p = 0.591, n = 10 cohorts).** Across the top-20 SHAP species, none
survive a Bonferroni-style threshold, indicating the species-level
signal is not a sequencing-depth artifact. This addresses the concern
that "deeper cohorts detect rare oral pathobionts and the classifier is
really learning depth."

  - `results/diagnostics/depth_confound_shap.csv`
  - `figures/diagnostics/depth_vs_fnucleatum_shap.png`

## 3. Joint-XGB calibration mechanism (calibration-gap critique)

A Brier-score decomposition (Murphy, 1973) on the pooled LODO
predictions shows that the calibration gap is driven almost entirely by
the **reliability term**:

  - Species RF: reliability = 0.0067
  - Joint RF:   reliability = 0.0057
  - Joint XGB:  reliability = 0.0260  (~4x larger)

Resolution (0.049–0.061) and uncertainty (0.250) are essentially
matched across the three models, confirming that the joint XGB's worse
ECE reflects probability-calibration miscalibration — the aggressive
logit pushes characteristic of gradient boosting on tabular data — and
not a loss of discriminative information. This contextualizes the
modest joint-XGB AUC difference and motivates the manuscript's
recommendation to use the RF as the primary classifier.

  - `results/diagnostics/brier_decomposition.csv`
  - `figures/diagnostics/calibration_mechanism.png`
