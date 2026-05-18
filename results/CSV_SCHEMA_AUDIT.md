# CSV Schema Audit

Read-only audit of every CSV under `results/`, `results/diagnostics/`,
`results/supplementary/`, and `data/processed/`.

- Files audited: **75 / 75** (4 in `data/processed/`, 38 in `results/`,
  20 in `results/diagnostics/`, 13 in `results/supplementary/`).
- No file failed to parse.
- All NaN counts are either zero or explained (see per-file table).
- Cross-file checks pass; one P0 column-naming inconsistency, one P0
  label-vocabulary inconsistency, and a handful of P1 polish items
  found.

## Cross-checks (status)

| Check | Result |
|---|---|
| `preds_species_rf.csv` ↔ `preds_joint_rf.csv` sample_id sets | identical, n=1339 |
| `preds_species_rf.csv` ↔ `preds_joint_xgb.csv` sample_id sets | identical, n=1339 |
| `preds_*_with_hannigan.csv` sample_id sets (RF species / RF joint / XGB joint) | identical, n=1392 (1339 + 53 Hannigan) |
| `bootstrap_ci.csv` row count = 10 cohorts × 3 models + 3 pooled = 33 | exactly **33** rows, models = {joint_rf, joint_xgb, species_rf}, cohorts = 10 + `pooled` |
| `supplementary/S4_bootstrap_ci.csv` row count | **33** rows, same schema as `bootstrap_ci.csv` |

## Per-file table

Columns are listed in declaration order. `NaN` is the sum across all
columns (per-column counts are noted only where non-zero). `id_col`
flags the canonical key (or the first column when it is unique).
Feature columns of the three big abundance / metadata tables are
elided.

### `data/processed/`

| File | n_rows | n_cols | id_col | Columns | dtypes | NaN |
|---|---:|---:|---|---|---|---|
| `metadata_clean.csv` | 1522 | 10 | `sample_id` | sample_id, study_name, study_condition, age, gender, BMI, country, sequencing_platform, number_reads, label | object×5, float64×2, int64×2, object | 17 (age=1, BMI=16) |
| `species_filtered.csv` | 1522 | 230 | `sample_id` | sample_id + 229 species relative abundance cols | object, float64×229 | 0 |
| `pathway_unstratified.csv` | 1604 | 404 | `sample_id` | sample_id + 403 pathway cols | object, float64×403 | 0 |
| `pathway_abundance_filtered.csv` | 1604 | 4817 | `sample_id` | sample_id + 4816 stratified pathway cols (incl. `UNMAPPED`, `UNINTEGRATED`, `…|unclassified`, `…|g__X.s__Y`) | object, float64×4816 | 0 |

### `results/` (main results)

| File | n_rows | n_cols | id_col | Columns | dtypes |
|---|---:|---:|---|---|---|
| `adenoma_counts_per_cohort.csv` | 4 | 2 | `study_name` | study_name, n_adenoma | object, int64 |
| `adenoma_lodo_results.csv` | 4 | 3 | `task` | task, mean_lodo_auc, n_folds | object, float64, int64 |
| `adenoma_rebalanced_lodo.csv` | 64 | 9 | composite (task, strategy, cohort) | task, strategy, cohort, auc, n_train, n_test, n_train_minority, n_train_majority, mean_lodo_auc | object×3, float64, int64×4, float64 |
| `adenoma_rebalanced_summary.csv` | 16 | 3 | composite (task, strategy) | task, strategy, mean_lodo_auc | object×2, float64 |
| `adenoma_results.csv` | 2 | 3 | `task` | task, rf_auc, xgb_auc | object, float64×2 |
| `baseline_results.csv` | 10 | 4 | `cohort` | cohort, auc, n_train, n_test | object, float64, int64×2 |
| `bio_pathway_results.csv` | 10 | 3 | `cohort` | cohort, bio_pw_auc, n_features | object, float64, int64 |
| `bootstrap_ci.csv` | 33 | 6 | composite (model, cohort) | model, cohort, auc, ci_lo, ci_hi, n | object×2, float64×3, int64 |
| `combat_results.csv` | 10 | 2 | `cohort` | cohort, auc | object, float64 |
| `confounder_results.csv` | 4 | 2 | `method` | method, mean_auc | object, float64 |
| `covariate_comparison.csv` | 10 | 4 | `cohort` | cohort, species_auc, species_cov_auc, difference | object, float64×3 |
| `delong_results.csv` | 3 | 8 | composite (model_a, model_b) | model_a, model_b, auc_a, auc_b, auc_diff, z, p_value, n_samples | object×2, float64×5, int64 |
| `external_validation.csv` | 3 | 5 | `cohort` | cohort, auc, n_samples, n_crc, n_control | object, float64, int64×3 |
| `joint_results.csv` | 10 | 5 | `cohort` | cohort, rf_auc, xgb_auc, rf_n_features, xgb_n_features | object, float64×2, int64×2 |
| `model_comparison.csv` | 3 | 11 | `comparison` | comparison, mean_a, mean_b, mean_diff, ci_low, ci_high, t_stat, t_pvalue, wilcoxon_stat, wilcoxon_pvalue, n_folds | object, float64×9, int64 |
| `preds_bio_pathway_rf.csv` | 1339 | 4 | `sample_id` | sample_id, cohort, y_true, y_prob | object×2, int64, float64 |
| `preds_joint_rf.csv` | 1339 | 4 | `sample_id` | sample_id, cohort, y_true, y_prob | object×2, int64, float64 |
| `preds_joint_rf_with_hannigan.csv` | 1392 | 4 | `sample_id` | sample_id, cohort, y_true, y_prob | object×2, int64, float64 |
| `preds_joint_xgb.csv` | 1339 | 4 | `sample_id` | sample_id, cohort, y_true, y_prob | object×2, int64, float64 |
| `preds_joint_xgb_with_hannigan.csv` | 1392 | 4 | `sample_id` | sample_id, cohort, y_true, y_prob | object×2, int64, float64 |
| `preds_species_rf.csv` | 1339 | 4 | `sample_id` | sample_id, cohort, y_true, y_prob | object×2, int64, float64 |
| `preds_species_rf_with_hannigan.csv` | 1392 | 4 | `sample_id` | sample_id, cohort, y_true, y_prob | object×2, int64, float64 |
| `seed_sensitivity.csv` | 5 | 3 | `seed` | seed, mean_auc, std_auc | int64, float64×2 |
| `sensitivity_thresholds.csv` | 20 | 6 | composite (prev_threshold, mean_threshold) | prev_threshold, mean_threshold, n_pathways_mean, n_features_mean, mean_auc, std_auc | float64×6 |
| `sensitivity_with_hannigan_delong.csv` | 3 | 8 | composite (model_a, model_b) | model_a, model_b, auc_a, auc_b, auc_diff, z, p_value, n_samples | object×2, float64×5, int64 |
| `sensitivity_with_hannigan_per_cohort.csv` | 33 | 4 | composite (cohort, model) | cohort, model, auc, n | object×2, float64, int64 |
| `sensitivity_with_hannigan_pooled.csv` | 3 | 5 | `model` | model, pooled_auc, ci_lo, ci_hi, n | object, float64×3, int64 |
| `shap_adenoma_vs_crc.csv` | 632 | 2 | `feature` | feature, mean_abs_shap | object, float64 |
| `shap_adenoma_vs_crc_xgb.csv` | 632 | 2 | `feature` | feature, mean_abs_shap | object, float64 |
| `shap_crc_features.csv` | 229 | 2 | `feature` | feature, mean_abs_shap | object, float64 |
| `shap_crc_xgb.csv` | 632 | 2 | `feature` | feature, mean_abs_shap | object, float64 |
| `shap_healthy_vs_adenoma.csv` | 632 | 2 | `feature` | feature, mean_abs_shap | object, float64 |
| `shap_healthy_vs_adenoma_xgb.csv` | 632 | 2 | `feature` | feature, mean_abs_shap | object, float64 |
| `stratified_pathway_pilot.csv` | 80 | 6 | composite (feature_set, model, cohort) | feature_set, model, cohort, auc, n_features, mean_auc | object×3, float64, int64, float64 |
| `table1.csv` | 11 | 9 | `Cohort` | Cohort, Country, N (total), N (CRC), N (adenoma), N (control), Age (mean ± SD), Female %, BMI (mean ± SD) | object×2, int64×4, object×3 |
| `wirbel_replication.csv` | 6 | 6 | `cohort` | cohort, n_train, n_test, auc_ours, auc_wirbel_2019, delta | object, float64×5 — note `n_train`/`n_test` stored as **float**, with 1 NaN each for the pooled row |

### `results/diagnostics/`

| File | n_rows | n_cols | id_col | Columns | dtypes |
|---|---:|---:|---|---|---|
| `auprc_pooled.csv` | 3 | 4 | `model` | model, auroc, auprc, n | object, float64×2, int64 |
| `base_rate_ppv.csv` | 12 | 5 | `prevalence` | prevalence, ppv, npv, sens_fixed, spec_fixed | float64×5 |
| `brier_decomposition.csv` | 3 | 5 | `model` | model, brier, reliability, resolution, uncertainty | object, float64×4 |
| `calibration_metrics.csv` | 3 | 4 | `model` | model, brier, ece, n | object, float64×2, int64 |
| `confusion_matrices.csv` | 6 | 14 | composite (model, threshold_type) | model, threshold_type, threshold, tp, fp, tn, fn, sensitivity, specificity, ppv, npv, f1, mcc, n | object×2, float64, int64×4, float64×5, int64 |
| `cross_disease_predictions.csv` | 1663 | 4 | `sample_id` | sample_id, **study_name**, **group**, y_prob | object×3, float64 |
| `cross_disease_specificity.csv` | 15 | 9 | composite (disease_cohort, study_name, group) | disease_cohort, **study_name**, **group**, n_samples, mean_crc_prob, median_crc_prob, pct_above_youden_threshold, interpretation_label, source | object×3, int64, float64×3, object×2 |
| `cv_methodology_comparison.csv` | 39 | 4 | composite | strategy, fold_or_pooled, auc, n | object×2, float64, int64 |
| `cv_methodology_summary.csv` | 4 | 6 | `strategy` | strategy, n_folds, mean_per_fold_auc, pooled_auc, delong_z_vs_country_aware, delong_p_vs_country_aware | object, int64, float64×4 — 2 NaN expected (country-aware row is the reference, no z/p against itself) |
| `decision_curves.csv` | 250 | 6 | composite (model, threshold_pt, strategy) | model, threshold_pt, net_benefit, n_tp, n_fp, strategy | object, float64×2, int64×2, object |
| `depth_confound_shap.csv` | 20 | 3 | `species` | species, rho_depth_vs_rank, p_value | object, float64×2 |
| `fit_vs_microbiome.csv` | 5 | 7 | `assay` | assay, sensitivity, specificity, ppv_5pct, npv_5pct, source, notes | object, float64×4, object×2 |
| `lift_curves.csv` | 30 | 8 | composite (model, frac_screened) | model, frac_screened, cumulative_n, cumulative_tp, total_positives, cumulative_gains, lift, prevalence | object, float64, int64×3, float64×3 |
| `minimum_panel.csv` | 50 | 5 | `k` | k, top_species_added, mean_per_cohort_auc, pooled_auc, delta_from_full | int64, object, float64×3 |
| `per_cohort_operating_chars.csv` | 11 | 13 | `cohort` | cohort, threshold, n, n_pos, n_neg, tp, fp, tn, fn, sensitivity, specificity, ppv, npv | object, float64, int64×7, float64×4 |
| `per_cohort_ppv.csv` | 10 | 10 | `cohort` | cohort, n, sens_observed, spec_observed, ppv_at_0.5pct, ppv_at_2pct, ppv_at_5pct, ppv_at_10pct, npv_at_5pct, nnt_at_5pct | object, int64, float64×8 |
| `permutation_importance_species_rf.csv` | 229 | 4 | `feature` | feature, perm_imp_mean, perm_imp_std, perm_imp_rank | object, float64×2, int64 |
| `permutation_vs_shap_correlation.csv` | 229 | 4 | `feature` | feature, perm_rank, shap_rank, both_top20 | object, int64×2, bool |
| `power_analysis.csv` | 7 | 3 | `true_auc_diff` | true_auc_diff, power, n | float64×2, int64 |
| `raw_data_summary.csv` | 10 | 8 | `cohort` | cohort, n_samples, n_CRC, n_control, n_adenoma, median_shannon, median_depth_Mreads, pct_zero_features | object, int64×4, float64×3 |
| `sens_at_fixed_spec.csv` | 6 | 8 | composite (model, target_spec) | model, target_spec, achieved_spec, threshold, sensitivity, ppv_5pct, npv_5pct, number_needed_to_test_5pct | object, float64×7 |
| `subgroup_auc.csv` | 9 | 8 | composite (variable, level) | variable, level, n, n_pos, n_neg, auc, ci_lo, ci_hi | object×2, int64×3, float64×3 |

### `results/supplementary/`

| File | n_rows | n_cols | id_col | Columns | dtypes |
|---|---:|---:|---|---|---|
| `INDEX.csv` | 12 | 3 | `table` | table, file, description | object×3 |
| `S1_cohort_overview.csv` | 10 | 15 | `Cohort` | **Cohort**, Country, N_total, N_CRC, N_control, N_adenoma, Sequencing_platform, Reads_median_Mreads, Reads_min_Mreads, Reads_max_Mreads, Age_mean, Age_SD, BMI_mean, BMI_SD, Pct_female | object×2, int64×4, object, float64×8 |
| `S2_per_fold_aucs.csv` | 11 | 8 | `cohort` | cohort, species_rf_auc, n_train, n_test, joint_rf_auc, joint_xgb_auc, bio_pathway_rf_auc, combat_species_rf_auc | object, float64×7 — `n_train`/`n_test` again typed float, not int |
| `S3_top_shap_features.csv` | 120 | 5 | composite (task, rank) | task, rank, species, feature, mean_abs_shap | object, int64, object×2, float64 |
| `S4_bootstrap_ci.csv` | 33 | 6 | composite (model, cohort) | model, cohort, auc, ci_lo, ci_hi, n | object×2, float64×3, int64 |
| `S5_sensitivity_grid.csv` | 20 | 6 | composite | prev_threshold, mean_threshold, n_pathways_mean, n_features_mean, mean_auc, std_auc | float64×6 |
| `S6_adenoma_lodo.csv` | 4 | 3 | `task` | task, mean_lodo_auc, n_folds | object, float64, int64 |
| `S7_confounder_adjustment.csv` | 4 | 2 | `method` | method, mean_auc | object, float64 |
| `S8_seed_sensitivity.csv` | 5 | 3 | `seed` | seed, mean_auc, std_auc | int64, float64×2 |
| `S8b_seed_sensitivity_summary.csv` | 1 | 7 | `metric` | metric, n_seeds, grand_mean, across_seed_std, min, max, spread | object, int64, float64×5 |
| `S9_external_validation.csv` | 3 | 5 | `cohort` | cohort, auc, n_samples, n_crc, n_control | object, float64, int64×3 |
| `S10_delong.csv` | 3 | 8 | composite (model_a, model_b) | model_a, model_b, auc_a, auc_b, auc_diff, z, p_value, n_samples | object×2, float64×5, int64 |
| `S11_methods_comparison.csv` | 21 | 5 | `axis` | axis, thomas_2019, wirbel_2019, piccinno_2025, this_work | object×5 |

## P0 inconsistencies (worth fixing for cross-file joins / merges)

1. **`cohort` vs `study_name` vs `Cohort` column-name split.** The same
   semantic key — the FengQ_2015 / ZellerG_2014 / … study identifier —
   appears under three different column headers:
   - `study_name` (4 files): `adenoma_counts_per_cohort.csv`,
     `diagnostics/cross_disease_predictions.csv`,
     `diagnostics/cross_disease_specificity.csv`,
     `data/processed/metadata_clean.csv`.
   - `cohort` (24 files, the de-facto standard).
   - `Cohort` (capitalised, 2 files): `table1.csv`,
     `supplementary/S1_cohort_overview.csv`.

   The capitalised `Cohort` variant in `table1.csv` and
   `S1_cohort_overview.csv` is a presentation table where capitalisation
   is intentional, but the `study_name` variant in
   `cross_disease_predictions.csv` / `cross_disease_specificity.csv`
   pointlessly breaks `concat` / `merge` against the other 22 results
   files. Picking one of `cohort` or `study_name` and renaming the
   minority would let a downstream consumer pd-merge predictions on
   `[sample_id, cohort]` without per-file renames.

2. **Label / outcome column-name split.** The supervised target column
   is called `y_true` in all 7 `preds_*.csv` files, `label` in
   `metadata_clean.csv`, and `group` in
   `diagnostics/cross_disease_predictions.csv` and
   `diagnostics/cross_disease_specificity.csv`. The three names also
   encode different vocabularies:
   - `y_true`: {0, 1} (binary CRC vs control).
   - `label`: {-1, 0, 1} where `-1` = adenoma (intentionally masked,
     not a NaN problem; this is the only file with that convention).
   - `group`: free-text strings (`CRC`, `control`, `IBD`, `T2D`, …).

   This is also intentional in places (the cross-disease file
   genuinely has more than two classes), but a downstream user joining
   `metadata_clean.csv` (`label`) to `preds_species_rf.csv` (`y_true`)
   will silently produce two columns. Documenting the three-way
   relationship in `data/processed/README.md` (or just renaming `label`
   → `y_true_3class` in metadata) would help.

## P1 polish (cosmetic, no impact on numbers)

- **`auc` column-name proliferation.** Across 75 files there are 22
  distinct AUC-flavoured column names (`auc`, `mean_auc`, `mean_lodo_auc`,
  `pooled_auc`, `auroc`, `rf_auc`, `xgb_auc`, `bio_pw_auc`,
  `species_auc`, `species_cov_auc`, `auc_a`, `auc_b`, `auc_diff`,
  `std_auc`, `auc_ours`, `auc_wirbel_2019`, `mean_per_fold_auc`,
  `mean_per_cohort_auc`, `true_auc_diff`, `species_rf_auc`,
  `joint_rf_auc`, `joint_xgb_auc`, `bio_pathway_rf_auc`,
  `combat_species_rf_auc`). Most are semantically distinct and the
  variety is justified, but some pairs collapse trivially — e.g. in
  `confounder_results.csv` / `S7_confounder_adjustment.csv` the column
  is named `mean_auc`, whereas the parallel `seed_sensitivity.csv` /
  `S8_seed_sensitivity.csv` also use `mean_auc`, and these are fine,
  but `bio_pathway_results.csv` calls it `bio_pw_auc` for what is
  effectively the same column shape as every other per-cohort
  `cohort, auc, …` table. Consider standardising one-AUC-per-row files
  on `auc`.

- **`n_train` / `n_test` typed as float, not int.**
  - `wirbel_replication.csv` (cols are `float64`; 1 NaN in each for the
    pooled `Overall` row — that NaN is what is forcing the float dtype).
  - `supplementary/S2_per_fold_aucs.csv` (cols are `float64`, no NaN —
    purely a write-out polish issue).

  Either drop the pooled row from `wirbel_replication.csv` (or replace
  the NaN with the sum of the per-cohort cells) and cast to `int64`,
  or document that NaN means "not applicable for pooled".

- **Float-precision mismatch.** Several float columns mix
  short-and-long decimal expansions in the same column:
  - `bootstrap_ci.csv` `auc`: 2–12 decimals (most rows are 16-digit
    floats; a handful look like exact terminations).
  - `covariate_comparison.csv` `difference`: 2–14 decimals.
  - `preds_joint_xgb.csv` / `..._with_hannigan.csv` `y_prob`: 9–15
    decimals (XGB's `predict_proba` output written raw).
  - `sensitivity_with_hannigan_per_cohort.csv` `auc`: 4–12 decimals.
  - `shap_adenoma_vs_crc_xgb.csv`, `shap_crc_xgb.csv`,
    `shap_healthy_vs_adenoma.csv`, `shap_healthy_vs_adenoma_xgb.csv`
    `mean_abs_shap`: 0–10/11/15/10 decimals (lots of values written as
    bare integers when they happened to land on `0.0`).
  - `stratified_pathway_pilot.csv` `auc`: 2–12 decimals.
  - `diagnostics/decision_curves.csv` `net_benefit`: 0–6 decimals.

  A single `df.to_csv(float_format='%.6f')` (or `%.4f` for AUC tables,
  `%.8f` for raw probabilities) at write time would homogenise these.
  None of these affect the published numbers — they affect only file
  diff stability and look-and-feel.

- **Hidden whitespace in `feature` column of five SHAP files** and the
  `species` / `feature` columns of `supplementary/S3_top_shap_features.csv`:
  exactly one row in each has trailing whitespace inside the feature
  name. The same row in `shap_adenoma_vs_crc.csv`,
  `shap_adenoma_vs_crc_xgb.csv`, `shap_crc_xgb.csv`,
  `shap_healthy_vs_adenoma.csv`, `shap_healthy_vs_adenoma_xgb.csv`,
  `S3_top_shap_features.csv` is affected — likely a single MetaPhlAn
  feature name with a stray space that propagated through. A
  `df["feature"] = df["feature"].str.strip()` before write would fix
  it; downstream joins on `feature` between files can currently
  silently drop this row.

- **`BMI` mixed precision in `metadata_clean.csv`** (0–10 decimals,
  16 NaN values). Expected: BMI is reported with whatever precision
  the source cohort reported it. No action needed, just flagged.

- **Documented-but-absent / present-but-undocumented columns.** I
  diff'd the column lists against `README.md`, `REPRODUCING.md`,
  `results/decisions_addendum.md`, and `results/diagnostics/README.md`.
  No column documented in markdown is missing from its CSV. The
  markdown files describe files at the file level rather than at the
  per-column level, so this check is mostly vacuous — only
  `study_condition` (called out in `README.md` for using `control`
  instead of `healthy`) is referenced by column name, and that column
  is present with the documented vocabulary.

## Read-only confirmation

- No CSV under `results/`, `results/diagnostics/`,
  `results/supplementary/`, or `data/processed/` was modified or
  written by this audit.
- The only file created is this report (`results/CSV_SCHEMA_AUDIT.md`).
- The test suite is unaffected (the audit only read CSVs).
