# README and REPRODUCING link / code-block audit

Date: 2026-05-18

Scope: `README.md` and `REPRODUCING.md` (repo root). Verifies (a) every
markdown `[text](path)` relative link, (b) every script / file path
referenced from a fenced code block or prose, (c) syntactic plausibility
of every fenced `bash` and `python` block, and (d) every "expect:"
output file referenced in REPRODUCING.md against the on-disk
`results/` / `data/` tree.

## Headline

- Total relative markdown links checked: 7
- Total in-prose / code-block file references checked: 88
  (10 scripts in README quick-summary, 53 distinct scripts in
  REPRODUCING, 14 "Key files" table entries in README,
  11 expected output CSVs in REPRODUCING)
- BROKEN: 0
- FIXED: 0
- External (http/https) links: 0 (neither file contains any external
  URLs to test)

All references resolve. No edits to `README.md` or `REPRODUCING.md` were
required.

## Code-block correctness summary

| File | Fenced blocks | Languages | Notes |
|------|---------------|-----------|-------|
| README.md | 3 | bash (2), python (1) | All commands parse; `pip install -r requirements.lock`, `pip install -e .`, and 10 `python3 scripts/*.py` / `Rscript scripts/*.R` invocations. Python block imports 3 symbols (`run_lodo_cv`, `per_fold_pathway_filter`, `bootstrap_pooled_ci`) all confirmed exported from `src/crc_lodo_bench/__init__.py`. |
| REPRODUCING.md | 14 | bash (14) | All commands parse; ~50 `python3 scripts/*.py`, 2 `Rscript scripts/*.R`, 1 `bash scripts/run_robustness.sh`, 1 `pytest tests/ -v`. No unmatched fences, no obvious typos. |

## Cross-check: scripts referenced vs. scripts on disk

Every script referenced from either file exists at the path written.
Sampled diagnostics path families (`scripts/diagnostics/*.py`) all
present. The `scripts/` directory additionally contains unreferenced
files (`sensitivity_with_hannigan.py`, `wirbel_replication.py`, and
several `scripts/diagnostics/` modules such as
`cross_disease_specificity.py`, `cv_methodology_comparison.py`,
`decision_curves.py`, `generate_visual_abstract.py`, `lift_curves.py`,
`minimum_useful_panel.py`, `per_cohort_ppv.py`, `power_analysis.py`) —
these are *orphans* and out of scope here; flagged for the orphan-audit
agent only.

## Cross-check: "expect:" output files referenced in REPRODUCING.md

All output paths mentioned in `# expect:` / `# outputs:` comments exist:
`results/baseline_results.csv`, `preds_species_rf.csv`,
`results/joint_results.csv`, `preds_joint_rf.csv`, `preds_joint_xgb.csv`,
`results/model_comparison.csv`, `delong_results.csv`,
`results/adenoma_lodo_results.csv`,
`results/adenoma_rebalanced_lodo.csv`,
`results/adenoma_rebalanced_summary.csv`,
`results/bio_pathway_results.csv`, `preds_bio_pathway_rf.csv`,
`results/bio_pathway_shortlist.txt`,
`results/stratified_pathway_pilot.csv`, `results/bootstrap_ci.csv`,
`results/seed_sensitivity.csv`, `results/sensitivity_thresholds.csv`,
`results/confounder_results.csv`, `results/covariate_comparison.csv`,
`results/combat_results.csv`, `results/external_validation.csv`,
`results/table1.csv`, `results/adenoma_counts_per_cohort.csv`,
`results/diagnostics/raw_data_summary.csv`,
`results/diagnostics/auprc_pooled.csv`.

## Cross-check: 49/49 verify_results assertions

REPRODUCING.md line 143 documents "49 smoke-test assertions". Static
count of `check(...)` / `check_near(...)` invocations in
`scripts/verify_results.py`, with loop bodies expanded for the
non-degenerate path (all three pred files present; all four adenoma
tasks present; all four confounder methods present):

- Standalone `check` / `check_near` calls in `main`: 32
- `for pf, expected_n in pred_files` (3 files x 3 checks each, file
  present branch): 9
- `for task, expected_folds in [...]` (4 tasks x 1 check): 4
- `for method in (...)` (4 methods x 1 check): 4

Total: 32 + 9 + 4 + 4 = 49. Matches the documented `49/49 passes`
contract.

## Per-link / per-reference detail

### README.md — relative markdown links (7)

| file:line | link | status | action |
|-----------|------|--------|--------|
| README.md:108 | results/diagnostics/README.md | OK | none |
| README.md:109 | results/diagnostics/ROBUSTNESS_SUMMARY.md | OK | none |
| README.md:110 | results/diagnostics/CLINICAL_TRANSLATION_SUMMARY.md | OK | none |
| README.md:111 | results/diagnostics/RAW_DATA_PATTERNS.md | OK | none |
| README.md:112 | results/CITATION_AUDIT.md | OK | none |
| README.md:113 | results/decisions_addendum.md | OK | none |
| README.md:114 | results/baseline_results.md | OK | none |

### README.md — code-block script references (10)

| file:line | reference | status | action |
|-----------|-----------|--------|--------|
| README.md:36 | requirements.lock | OK | none |
| README.md:37 | scripts/export_data.R | OK | none |
| README.md:38 | scripts/preprocessing.py | OK | none |
| README.md:39 | scripts/train_baseline.py | OK | none |
| README.md:40 | scripts/train_joint.py | OK | none |
| README.md:41 | scripts/auc_comparison.py | OK | none |
| README.md:42 | scripts/bootstrap_ci.py | OK | none |
| README.md:43 | scripts/shap_analysis.py | OK | none |
| README.md:44 | scripts/bio_pathway_shortlist.py | OK | none |
| README.md:45 | scripts/adenoma_lodo.py | OK | none |
| README.md:46 | scripts/verify_results.py | OK | none |

### README.md — "Key files" table (14)

All 14 paths in the `data/processed/`, `data/raw/`, `results/`, and
`src/crc_lodo_bench/` tree exist as written (rows 89-101 of README.md
plus the `src/crc_lodo_bench/README.md` reference on row 67). Status:
OK / no action.

### REPRODUCING.md — code-block script references (53 distinct)

All scripts referenced exist on disk. Sample (full list verified):
`scripts/export_data.R`, `scripts/audit_subject_ids.R`,
`scripts/merge_pathways.py`, `scripts/validate_pathways.py`,
`scripts/filter_pathways.py`, `scripts/preprocessing.py`,
`scripts/generate_table1.py`, `scripts/adenoma_counts.py`,
`scripts/add_covariates.py`, `scripts/train_baseline.py`,
`scripts/train_joint.py`, `scripts/auc_comparison.py`,
`scripts/shap_analysis.py`, `scripts/shap_xgb.py`,
`scripts/shap_adenoma.py`, `scripts/adenoma_lodo.py`,
`scripts/rebalanced_adenoma_lodo.py`, `scripts/train_adenoma.py`,
`scripts/bio_pathway_shortlist.py`,
`scripts/stratified_pathway_pilot.py`, `scripts/bootstrap_ci.py`,
`scripts/seed_sensitivity.py`, `scripts/sensitivity_analysis.py`,
`scripts/confounder_adjustment.py`, `scripts/batch_correction.py`,
`scripts/external_validation.py`, `scripts/run_robustness.sh`,
`scripts/generate_figures.py`, `scripts/figure1_forest_plot.py`,
`scripts/figure5_shap_three_panel.py`, `scripts/verify_results.py`,
`scripts/diagnostics/calibration.py`,
`scripts/diagnostics/calibration_mechanism.py`,
`scripts/diagnostics/confusion_matrices.py`,
`scripts/diagnostics/roc_pr_curves.py`,
`scripts/diagnostics/per_cohort_sens_spec.py`,
`scripts/diagnostics/sens_at_fixed_specificity.py`,
`scripts/diagnostics/base_rate_ppv.py`,
`scripts/diagnostics/fit_comparison.py`,
`scripts/diagnostics/permutation_importance.py`,
`scripts/diagnostics/depth_confound_check.py`,
`scripts/diagnostics/subgroup_analysis.py`,
`scripts/diagnostics/raw_data_exploration.py`,
`scripts/lodo_cv.py`, `scripts/sanity_check.py`,
`scripts/find_nans.py`, `scripts/check_label_dist.py`,
`manuscript/markdown/_build_docx.py`,
`scripts/build_supplementary_tables.py`,
`scripts/build_submission.py`, `scripts/build_biorxiv_pdf.py`,
`scripts/test_submission_build.py`, `tests/test_lodo_cv.py`.
Status: all OK / no action.

## External links

None present in either file.

## Edits applied

None. Both files passed the audit with zero broken internal links.
