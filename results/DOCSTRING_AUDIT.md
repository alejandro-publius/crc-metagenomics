# Docstring audit — scripts/ and src/crc_lodo_bench/

Procedure: every Python file under `scripts/` (including `scripts/diagnostics/`) and
`src/crc_lodo_bench/` was read. For each module docstring and every top-level
function/class docstring, the prose was compared against the actual code to check
that inputs, outputs, side effects, and key algorithmic steps were accurately
described. P0 = misleading; P1 = incomplete; P2 = could be more precise; PASS =
docstring accurately describes the code.

Counts:
- P0 found: **1**     | P0 fixed: **1**
- P1 found: **2**     | P1 fixed: **2**
- P2 found: **1**     | P2 fixed: **1** (improved opportunistically — easy win)
- PASS: all other files
- Forbidden author attributions found (constraint violation, separate from
  docstring accuracy): **6 files**, all stripped.

`scripts/verify_results.py` still passes 49/49 after edits.

## Per-file audit table

| File | Function | Status | Issue summary |
|---|---|---|---|
| `src/crc_lodo_bench/__init__.py` | (module) | PASS | Re-exports match `__all__`; canonical-vs-vendored prose matches `_try_import_canonical`. |
| `src/crc_lodo_bench/filters.py` | (module) | PASS | Accurate. |
| `src/crc_lodo_bench/filters.py` | `per_fold_pathway_filter` | PASS | Parameters / Returns / Example all match code. |
| `src/crc_lodo_bench/lodo.py` | (module) | PASS | Accurate. |
| `src/crc_lodo_bench/lodo.py` | `_try_import_canonical` | PASS | Walks two dirs up; matches code. |
| `src/crc_lodo_bench/lodo.py` | `_vendored_get_lodo_splits` | PASS | 4-tuple yield documented correctly. |
| `src/crc_lodo_bench/lodo.py` | `_vendored_run_lodo_cv` | PASS | Parameters and Returns match. |
| `src/crc_lodo_bench/stats.py` | (module) | PASS | Accurate. |
| `src/crc_lodo_bench/stats.py` | `_midrank` | PASS | Accurate. |
| `src/crc_lodo_bench/stats.py` | `delong_test` | PASS | Accurate. |
| `src/crc_lodo_bench/stats.py` | `bootstrap_pooled_ci` | PASS | Accurate. |
| `scripts/add_covariates.py` | (module) | FIXED | Stripped author-attribution line (constraint violation). Docstring content accurate. |
| `scripts/add_covariates.py` | `run_species_plus_cov_lodo` | PASS | One-liner matches code. |
| `scripts/adenoma_counts.py` | (module) | PASS | Accurate. |
| `scripts/adenoma_lodo.py` | (module) | PASS | Accurate. |
| `scripts/adenoma_lodo.py` | `run_adenoma_lodo` | PASS | Accurate. |
| `scripts/auc_comparison.py` | (module) | PASS | Accurate. |
| `scripts/auc_comparison.py` | `delong_roc_test` | PASS | Accurate. |
| `scripts/batch_correction.py` | (module) | PASS | Accurate. |
| `scripts/bio_pathway_shortlist.py` | (module) | PASS | Accurate. |
| `scripts/bio_pathway_shortlist.py` | `select_pathways` | PASS | Accurate. |
| `scripts/bootstrap_ci.py` | (module) | PASS | Accurate. |
| `scripts/bootstrap_ci.py` | `bootstrap_auc_iid` / `bootstrap_auc_stratified` | PASS | Accurate. |
| `scripts/build_biorxiv_pdf.py` | (module) | FIXED | Stripped author-attribution line. Content otherwise accurate. |
| `scripts/build_biorxiv_pdf.py` | other helpers | PASS | All match code. |
| `scripts/build_submission.py` | (module) | FIXED | Stripped author-attribution line. Content otherwise accurate. |
| `scripts/build_submission.py` | other helpers | PASS | All match code. |
| `scripts/build_supplementary_tables.py` | (module) | PASS | Accurate. |
| `scripts/check_label_dist.py` | (module) | PASS | Accurate. |
| `scripts/confounder_adjustment.py` | (module) | PASS | Accurate. |
| `scripts/confounder_adjustment.py` | `prepare_covariates_per_fold` | PASS | Accurate. |
| `scripts/external_validation.py` | (module) | FIXED | Stripped author-attribution line. Content otherwise accurate. |
| `scripts/figure1_forest_plot.py` | (module) | PASS | Accurate. |
| `scripts/figure5_shap_three_panel.py` | (module) | PASS | Panel ordering matches `SHAP_FILES` dict iteration. |
| `scripts/figure5_shap_three_panel.py` | `clean_feature_name` / `load_shap` | PASS | Accurate. |
| `scripts/filter_pathways.py` | (module) | PASS | Accurate. |
| `scripts/find_nans.py` | (module) | PASS | Accurate. |
| `scripts/generate_figures.py` | (module) | FIXED (P2) | Original docstring was a single sentence (`"Generate all figures"`). Expanded to enumerate the four output PNGs, the inputs read, and note that the headline manuscript figures live in dedicated scripts. |
| `scripts/generate_table1.py` | (module) | FIXED | Stripped author-attribution line. Content otherwise accurate. |
| `scripts/lodo_cv.py` | (module) | PASS | Accurate. |
| `scripts/lodo_cv.py` | `get_lodo_splits` | FIXED (P1) | Old yield signature documented as 3-tuple `(cohort, train, test)`, code actually yields 4-tuple `(cohort, train, test, excluded_cohorts)`. Updated to match. |
| `scripts/lodo_cv.py` | `run_lodo_cv` | FIXED (P1) | Parameter block was correct but Returns section was missing. Added explicit Returns dict spec to match the `results` dict the code builds (keys: `cohort, auc, n_train, n_test, n_features, excluded_cohorts, mean_auc, std_auc`). |
| `scripts/merge_pathways.py` | (module) | FIXED (P1) | Docstring claimed the script produced a single `pathway_abundance.csv`. Code also writes a sidecar `pathway_unstratified_full.csv` consumed by `sensitivity_analysis.py`. Updated docstring to disclose both outputs. |
| `scripts/preprocessing.py` | (module) | PASS | Accurate. |
| `scripts/rebalanced_adenoma_lodo.py` | (module) | PASS | Accurate. |
| `scripts/rebalanced_adenoma_lodo.py` | inner helpers | PASS | Accurate. |
| `scripts/sanity_check.py` | (module) | PASS | Accurate. |
| `scripts/seed_sensitivity.py` | (module) | PASS | Accurate. |
| `scripts/sensitivity_analysis.py` | (module) | PASS | Accurate. |
| `scripts/sensitivity_with_hannigan.py` | (module) | PASS | Accurate. |
| `scripts/sensitivity_with_hannigan.py` | helpers | PASS | Accurate. |
| `scripts/shap_adenoma.py` | (module) | PASS | Accurate. |
| `scripts/shap_analysis.py` | (module) | PASS | Accurate. |
| `scripts/shap_xgb.py` | (module) | PASS | Accurate. |
| `scripts/stratified_pathway_pilot.py` | (module) | PASS | Accurate. |
| `scripts/test_submission_build.py` | (module) | FIXED | Stripped author-attribution line. Content otherwise accurate. |
| `scripts/train_adenoma.py` | (module) | PASS | Accurate; correctly flagged as DEPRECATED. |
| `scripts/train_baseline.py` | (module) | PASS | Accurate; expected AUC numbers match `verify_results.py` tolerances. |
| `scripts/train_joint.py` | (module) | PASS | Accurate; expected per-fold pathway count 402-406 matches verify tolerance [400, 410]. |
| `scripts/train_joint.py` | `pathway_filter` | PASS | Accurate. |
| `scripts/validate_pathways.py` | (module) | PASS | Accurate. |
| `scripts/verify_results.py` | (module) | PASS | Accurate. |
| `scripts/wirbel_replication.py` | (module) | PASS | Accurate. |
| `scripts/diagnostics/base_rate_ppv.py` | (module) | PASS | Accurate. |
| `scripts/diagnostics/calibration.py` | (module) | PASS | Accurate. |
| `scripts/diagnostics/calibration_mechanism.py` | (module) | PASS | Accurate. |
| `scripts/diagnostics/confusion_matrices.py` | (module) | PASS | Accurate. |
| `scripts/diagnostics/cross_disease_specificity.py` | (module) | PASS | Accurate. |
| `scripts/diagnostics/cv_methodology_comparison.py` | (module) | FIXED (P0) | Strategy 4 was described as "Stratified GroupKFold by cohort (10 folds; each cohort held out exactly once with class-stratified label balance)". The implementation in `splits_group_kfold_cohort` does no class stratification — it is plain LeaveOneGroupOut. Rewrote the bullet to say "GroupKFold by cohort (equivalent to LeaveOneGroupOut at n_groups=n_folds; ... no extra class stratification and no country-awareness applied)". |
| `scripts/diagnostics/cv_methodology_comparison.py` | inner functions | PASS | Function-level docstring already correctly described LOGO-equivalence. |
| `scripts/diagnostics/decision_curves.py` | (module) | PASS | Accurate. |
| `scripts/diagnostics/depth_confound_check.py` | (module) | PASS | Accurate. |
| `scripts/diagnostics/fit_comparison.py` | (module) | PASS | Accurate. |
| `scripts/diagnostics/generate_visual_abstract.py` | (module) | PASS | Accurate. |
| `scripts/diagnostics/lift_curves.py` | (module) | PASS | Accurate. |
| `scripts/diagnostics/minimum_useful_panel.py` | (module) | PASS | Accurate. |
| `scripts/diagnostics/per_cohort_ppv.py` | (module) | PASS | Accurate. |
| `scripts/diagnostics/per_cohort_sens_spec.py` | (module) | PASS | Accurate. |
| `scripts/diagnostics/permutation_importance.py` | (module) | PASS | Accurate. |
| `scripts/diagnostics/power_analysis.py` | (module) | PASS | Accurate. |
| `scripts/diagnostics/raw_data_exploration.py` | (module) | PASS | Accurate. |
| `scripts/diagnostics/roc_pr_curves.py` | (module) | PASS | Accurate. |
| `scripts/diagnostics/sens_at_fixed_specificity.py` | (module) | PASS | Accurate. |
| `scripts/diagnostics/subgroup_analysis.py` | (module) | PASS | Accurate. |

## Fix summary

Edits applied (Python files in `scripts/` and `src/` only — no edits to
`tests/`, `manuscript/`, `submission/`, `results/`, or config files):

1. `scripts/lodo_cv.py` — `get_lodo_splits` yield signature corrected from
   3-tuple to 4-tuple; `run_lodo_cv` Returns block added.
2. `scripts/merge_pathways.py` — module docstring now mentions the second
   output file `pathway_unstratified_full.csv`.
3. `scripts/diagnostics/cv_methodology_comparison.py` — strategy 4 redescribed
   to remove the false "stratified ... class-stratified label balance" claim
   that did not reflect the LOGO-equivalent implementation.
4. `scripts/generate_figures.py` — single-sentence module docstring expanded
   to enumerate the four outputs and disclose that headline manuscript
   figures live elsewhere.
5. Six scripts had author-attribution lines naming individuals; those lines
   were removed (constraint compliance — separate from docstring accuracy).

Post-edit, `python3 scripts/verify_results.py` exits 0 with all 49 checks
passing.
