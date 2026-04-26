# Reproducing the Analyses

See README.md and results/decisions_addendum.md for the canonical pipeline. Detailed step-by-step reproduction instructions are in this commit message and the script docstrings.

## Quick start

1. Rscript scripts/export_data.R
2. Rscript scripts/audit_subject_ids.R
3. python3 scripts/merge_pathways.py
4. python3 scripts/validate_pathways.py
5. python3 scripts/filter_pathways.py
6. python3 scripts/preprocessing.py
7. python3 scripts/generate_table1.py
8. python3 scripts/adenoma_counts.py
9. python3 scripts/train_baseline.py    # expect AUC 0.803
10. python3 scripts/train_joint.py      # expect Joint RF 0.785, XGB 0.784
11. python3 scripts/train_adenoma.py    # expect H-vs-A 0.681/0.709, A-vs-CRC 0.787/0.809
12. python3 scripts/auc_comparison.py
13. python3 scripts/shap_analysis.py
14. python3 scripts/shap_adenoma.py
15. python3 scripts/shap_xgb.py
16. python3 scripts/add_covariates.py
17. python3 scripts/external_validation.py
18. python3 scripts/generate_figures.py

## Sanity checks

- python3 scripts/sanity_check.py
- python3 scripts/find_nans.py
- python3 scripts/check_label_dist.py

## Reproducibility

All scripts use random_state=42. Joint model uses unstratified pathway features filtered per LODO fold (540 candidates; 402-406 retained per fold from train-only prevalence/mean). See results/decisions_addendum.md for SMOTE, DeLong, normalization, tuning, and LODO leakage decisions.

## Robustness and sensitivity analyses

19. python3 scripts/sensitivity_analysis.py   # filter threshold sweep
20. python3 scripts/confounder_adjustment.py   # age/sex/BMI confounding
21. python3 scripts/adenoma_lodo.py            # cross-cohort adenoma LODO
22. python3 scripts/bootstrap_ci.py            # 95% bootstrap CIs on AUCs
23. python3 scripts/seed_sensitivity.py        # random seed stability
24. python3 scripts/batch_correction.py        # per-fold ComBat (requires pycombat)
25. python3 scripts/verify_results.py          # smoke-test headline numbers
